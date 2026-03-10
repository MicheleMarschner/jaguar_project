"""
after each model is trained, extract embeddings for the relevant images, normalize them, and then compute similarities.
1. Comparable protocol and evaluation
2. Save:  a) per-image embeddings, b)similarity matrix on val, c) per-query rankings, d) summary score distributions
3. analyze error overlap to find right candidates: For each query on validation, record who gets it right and who gets it wrong (so good score overall but coplementary) -> different mistakes
4. ensemble baseline: Score fusion where each each model scores independently, then scores are combined for the same query-gallery pair. ( equal and than manually weighted)

-> currently the gallery is only val_ds (if noisy change to train+val)
"""

import os
from pathlib import Path

from jaguar.utils.utils_datasets import build_processing_fn, load_split_jaguar_from_FO_export
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaguar.utils.utils import read_toml_from_path, resolve_path
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from jaguar.config import EXPERIMENTS_STORE, PATHS, DEVICE
from jaguar.models.jaguarid_models import JaguarIDModel


def query_expansion(emb, top_k=3):
    print("Applying Query Expansion...")
    sims = emb @ emb.T
    indices = np.argsort(-sims, axis=1)[:, :top_k]

    new_emb = np.zeros_like(emb)
    for i in range(len(emb)):
        new_emb[i] = np.mean(emb[indices[i]], axis=0)

    return new_emb / (np.linalg.norm(new_emb, axis=1, keepdims=True) + 1e-12)


def k_reciprocal_rerank(prob, k1=20, k2=6, lambda_value=0.3):
    print("Applying Re-ranking...")

    q_g_dist = 1 - prob
    original_dist = q_g_dist.copy()
    initial_rank = np.argsort(original_dist, axis=1)

    nn_k1 = []
    for i in range(prob.shape[0]):
        forward_k1 = initial_rank[i, :k1 + 1]
        backward_k1 = initial_rank[forward_k1, :k1 + 1]
        fi = np.where(backward_k1 == i)[0]
        nn_k1.append(forward_k1[fi])

    jaccard_dist = np.zeros_like(original_dist)

    for i in range(prob.shape[0]):
        ind_non_zero = np.where(original_dist[i, :] < 0.6)[0]
        ind_images = [
            inv for inv in ind_non_zero
            if len(np.intersect1d(nn_k1[i], nn_k1[inv])) > 0
        ]

        for j in ind_images:
            intersection = len(np.intersect1d(nn_k1[i], nn_k1[j]))
            union = len(np.union1d(nn_k1[i], nn_k1[j]))
            if union > 0:
                jaccard_dist[i, j] = 1 - intersection / union

    return 1 - (jaccard_dist * lambda_value + original_dist * (1 - lambda_value))


def _resolve_from_project_root(path_str):
    p = Path(path_str)
    return p if p.is_absolute() else PATHS.project_root / p

def load_model(member_cfg: dict, num_classes: int):

    # absolute or relative - do we have a function for that?
    # ! TODO need to retrieve the base_config not only the experiment one
    model_cfg = read_toml_from_path(_resolve_from_project_root(member_cfg["experiment_toml"]))
    checkpoint_path = _resolve_from_project_root(model_cfg["training"]["save_dir"])

    print(f"Loading model '{member_cfg['name']}' from {checkpoint_path}...")

    model = JaguarIDModel(
        backbone_name=model_cfg['model']['backbone_name'],
        num_classes=num_classes,
        head_type=model_cfg['model']['head_type'],
        device=DEVICE,
        emb_dim=model_cfg['model']['emb_dim'],
        freeze_backbone=model_cfg['model']['freeze_backbone'],
        loss_s=model_cfg["model"].get("s", 30.0),
        loss_m=model_cfg["model"].get("m", 0.5),
        use_projection=model_cfg['model']['use_projection'],
        use_forward_features=model_cfg['model']['use_forward_features'],
    )

    checkpoint = torch.load(checkpoint_path / "best_model.pth", map_location=DEVICE, weights_only=False)

    ## besonderheiten per model??
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def extract_embeddings(model, dataloader, use_tta=False):
    print(f"\nExtracting embeddings for {model.backbone_wrapper.name}...")
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs = batch["img"].to(DEVICE)

            # extracts already normalized embeddings
            feats = model.get_embeddings(imgs)

            if use_tta:
                flipped = torch.flip(imgs, dims=[3])
                feats_flip = model.get_embeddings(flipped)
                feats = (feats + feats_flip) / 2.0

            feats = torch.nn.functional.normalize(feats, dim=1)
            embeddings.append(feats.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings


def cosine_similarity_matrix(embeddings, use_qe=False, use_rerank=False):
    
    sim_matrix = embeddings @ embeddings.T

    if use_qe:
        sim_matrix = query_expansion(embeddings) @ query_expansion(embeddings).T

    if use_rerank:
        sim_matrix = k_reciprocal_rerank(sim_matrix)

    # would diffuse the min_max later -> moved after normalization of all sim_matrices
    #sim_matrix = np.clip(sim_matrix, 0.0, 1.0)

    return sim_matrix


def minmax_norm(mat, eps=1e-12):
    return (mat - mat.min()) / (mat.max() - mat.min() + eps)


def fuse_similarity_matrices(sim_mats, weights, use_minmax=True, square=True):
    # Source: https://www.kaggle.com/competitions/shopee-product-matching/writeups/watercooled-4th-place-solution
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    fused_sim_matrix = np.zeros_like(sim_mats[0], dtype=np.float64)

    for w, sim in zip(weights, sim_mats):
        # normalizing individual sim_matrices to [0,1]
        # helps more than ((cos + 1) / 2) when models have very different score distributions.
        if use_minmax:
            sim = minmax_norm(sim)
            if square:
                sim = sim ** 2
        fused_sim_matrix += w * sim

    return np.clip(fused_sim_matrix, 0.0, 1.0)
    

def validate_submission(submission_df, test_df):
    assert len(submission_df) == len(test_df)
    assert list(submission_df.columns) == ["row_id", "similarity"]
    assert submission_df["row_id"].tolist() == test_df["row_id"].tolist()
    assert np.isfinite(submission_df["similarity"].values).all()
    assert (submission_df["similarity"] >= 0).all()
    assert (submission_df["similarity"] <= 1).all()


def create_simple_ensemble(config, save_dir):

    # either branch in main due to base_name or directly enter here
    #base_config = load_toml_config(args.base_config)
    #experiment_config = load_toml_config(args.experiment_config)

    parquet_root = resolve_path(config["data"]["split_data_path"], EXPERIMENTS_STORE)
    data_path = PATHS.data_export / "splits_curated"

    members = config["members"]
    weights = config["fusion"]["weights"]

    # Shopee-style fusion
    USE_MINMAX = True
    SQUARE_BEFORE_FUSION = True

    train_processing_fn = build_processing_fn(config, split="train")
    val_processing_fn = build_processing_fn(config, split="val")

    _, train_ds, val_ds = load_split_jaguar_from_FO_export(
        data_path,
        overwrite_db=False,
        parquet_path=parquet_root,
        train_processing_fn=train_processing_fn,
        val_processing_fn=val_processing_fn,
        include_duplicates=config["split"]["include_duplicates"],
        use_fiftyone=config["data"]["use_fiftyone"]
    )

    # Calculate Identities
    num_classes = len(train_ds.label_to_idx)

    # --------------------------------------------------
    # 4. EXTRACT PER-MODEL EMBEDDINGS / SIM MATRICES
    # --------------------------------------------------
    sim_mats = []
    member_outputs = {}

    # for name in MODEL_CONFIGS():
    for i, member in enumerate(members):
        print(f"\n========== Model {i+1}/{len(members)}: {member['name']} ==========")
        
        model = load_model(member, num_classes=num_classes)

        # important: each model must use its own transform
        val_ds.transform = model.backbone_wrapper.transform

        val_loader = DataLoader(
            val_ds,
            batch_size=config["inference"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )

        embeddings = extract_embeddings(
            model=model,
            dataloader=val_loader,
            use_tta=config["inference"]["use_tta"]
        )

        sim_matrix = cosine_similarity_matrix(
            embeddings,
            use_qe=config["inference"]["use_qe"],
            use_rerank=config["inference"]["use_rerank"]
        )

        print("\nSimilarity statistics:")
        print(f"  Min:  {sim_matrix.min():.4f}")
        print(f"  Max:  {sim_matrix.max():.4f}")
        print(f"  Mean: {sim_matrix.mean():.4f}")
        print(f"  Std:  {sim_matrix.std():.4f}")

        
        member_outputs[member["name"]] = {
            "embeddings": embeddings,
            "sim_matrix": sim_matrix,
            "weights": config["fusion"]["weights"][i]
        }

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------
    # 5. FUSE SIMILARITY MATRICES
    # --------------------------------------------------

    sim_mats = [member_outputs[member["name"]]["sim_matrix"] for member in members]

    print("\nFusing similarity matrices...")
    
    fused_sim_matrix = fuse_similarity_matrices(sim_mats, weights, USE_MINMAX, SQUARE_BEFORE_FUSION)

    print("\nFused similarity statistics:")
    print(f"  Min:  {fused_sim_matrix.min():.4f}")
    print(f"  Max:  {fused_sim_matrix.max():.4f}")
    print(f"  Mean: {fused_sim_matrix.mean():.4f}")
    print(f"  Std:  {fused_sim_matrix.std():.4f}")

    return {
        "member_outputs": member_outputs,
        "fused_sim_matrix": fused_sim_matrix,
        "labels": np.asarray(val_ds.labels),
    }

   
if __name__ == "__main__":
    create_simple_ensemble()