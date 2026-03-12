"""
after each model is trained, extract embeddings for the relevant images, normalize them, and then compute similarities.
1. Comparable protocol and evaluation
2. Save:  a) per-image embeddings, b)similarity matrix on val, c) per-query rankings, d) summary score distributions
3. analyze error overlap to find right candidates: For each query on validation, record who gets it right and who gets it wrong (so good score overall but coplementary) -> different mistakes
4. ensemble baseline: Score fusion where each each model scores independently, then scores are combined for the same query-gallery pair. ( equal and than manually weighted)

-> currently the gallery is only val_ds (if noisy change to train+val)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from jaguar.config import EXPERIMENTS_STORE, PATHS, DEVICE
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.utils.utils_models import load_or_extract_jaguarid_embeddings
from jaguar.utils.utils_experiments import read_toml_from_path
from jaguar.utils.utils_evaluation import build_local_to_emb_row
from jaguar.utils.utils_datasets import build_processing_fn, load_split_jaguar_from_FO_export
from jaguar.utils.utils import ensure_dir, resolve_path


def get_embedding_cache_path(
    config,
    member_name: str,
    split_name: str,
    cache_prefix: str | None = None,
) -> Path:
    ensemble_name = config["ensemble"]["name"]
    cache_dir = PATHS.runs / "ensemble_cache" / ensemble_name / member_name
    ensure_dir(cache_dir)

    tta_tag = "tta" if config["inference"]["use_tta"] else "no_tta"
    prefix = f"{cache_prefix}__" if cache_prefix else ""

    return cache_dir / f"{prefix}{split_name}_{tta_tag}_embeddings.npy"


def load_or_extract_jaguarid_embeddings_cached(
    model,
    torch_ds,
    config,
    member_name: str,
    split_name: str,
    cache_prefix: str | None = None,
):
    cache_path = get_embedding_cache_path(
        config=config,
        member_name=member_name,
        split_name=split_name,
        cache_prefix=cache_prefix,
    )

    if cache_path.exists():
        print(f"[Cache] Loading embeddings from {cache_path}")
        emb = np.load(cache_path)
        print(f"[Cache] Loaded shape: {emb.shape}")
        return emb

    emb = load_or_extract_jaguarid_embeddings(
        model=model,
        torch_ds=torch_ds,
        split=split_name,
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix=cache_prefix,
        folder=cache_path.parent,
    )
    return emb



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
    return p if p.is_absolute() else PATHS.checkpoints / p

def load_model(member_cfg: dict, num_classes: int):

    # absolute or relative - do we have a function for that?
    # ! TODO need to retrieve the base_config not only the experiment one
    model_cfg = read_toml_from_path(_resolve_from_project_root(member_cfg["config_path"]))
    checkpoint_dir = _resolve_from_project_root(member_cfg["checkpoint_path"])

    print(f"Loading model '{member_cfg['name']}' from {checkpoint_dir}...")

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
    print(checkpoint_dir)

    checkpoint = torch.load(checkpoint_dir / "best_model.pth", map_location=DEVICE, weights_only=False)

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


def cosine_similarity_matrix_square(embeddings, use_qe=False, use_rerank=False):
    
    sim_matrix = embeddings @ embeddings.T

    if use_qe:
        sim_matrix = query_expansion(embeddings) @ query_expansion(embeddings).T

    if use_rerank:
        sim_matrix = k_reciprocal_rerank(sim_matrix)

    # would diffuse the min_max later -> moved after normalization of all sim_matrices
    #sim_matrix = np.clip(sim_matrix, 0.0, 1.0)

    return sim_matrix

def cosine_similarity_matrix_rect(query_embeddings, gallery_embeddings, use_qe=False, use_rerank=False):
    if use_qe:
        raise NotImplementedError("QE for rectangular setup not implemented yet.")
    if use_rerank:
        raise NotImplementedError("Reranking for rectangular setup not implemented yet.")

    return query_embeddings @ gallery_embeddings.T


def minmax_norm(mat, eps=1e-12):
    return (mat - mat.min()) / (mat.max() - mat.min() + eps)


def normalize_none(sim: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return sim

# Source: https://www.kaggle.com/competitions/shopee-product-matching/writeups/watercooled-4th-place-solution
# normalizing individual sim_matrices to [0,1]
 # helps more than ((cos + 1) / 2) when models have very different score distributions.
def normalize_global_minmax(sim: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    sim_min = sim.min()
    sim_max = sim.max()
    return (sim - sim_min) / (sim_max - sim_min + eps)


def normalize_row_minmax(sim: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_min = sim.min(axis=1, keepdims=True)
    row_max = sim.max(axis=1, keepdims=True)
    return (sim - row_min) / (row_max - row_min + eps)


def normalize_row_zscore(sim: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_mean = sim.mean(axis=1, keepdims=True)
    row_std = sim.std(axis=1, keepdims=True)
    return (sim - row_mean) / (row_std + eps)


NORMALIZERS = {
    "none": normalize_none,
    "global_minmax": normalize_global_minmax,
    "row_minmax": normalize_row_minmax,
    "row_zscore": normalize_row_zscore,
}


def fuse_similarity_matrices(
    sim_mats: list[np.ndarray],
    weights: list[float],
    normalize_mode: str = "global_minmax",
    square_before_fusion: bool = True,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    normalize_fn = NORMALIZERS[normalize_mode]

    fused = np.zeros_like(sim_mats[0], dtype=np.float64)

    for w, sim in zip(weights, sim_mats):
        sim = normalize_fn(np.asarray(sim, dtype=np.float64))
        if square_before_fusion:
            sim = sim ** 2
        fused += w * sim

    return fused        # np.clip(fused_sim_matrix, 0.0, 1.0)

    

def fuse_embeddings_concat(embeddings_list, weights=None):
    """
    Embedding fusion via weighted concatenation.

    Steps:
    1) normalize each model embedding
    2) optionally scale by model weight
    3) concatenate along feature dimension
    4) normalize fused embedding again
    """
    if len(embeddings_list) == 0:
        raise ValueError("embeddings_list must not be empty")

    n = embeddings_list[0].shape[0]
    for emb in embeddings_list:
        if emb.shape[0] != n:
            raise ValueError("All embedding arrays must have the same number of images")

    if weights is None:
        weights = np.ones(len(embeddings_list), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != len(embeddings_list):
            raise ValueError("weights must match number of embedding arrays")

    parts = []
    for emb, w in zip(embeddings_list, weights):
        emb = np.asarray(emb, dtype=np.float64)

        # per-model normalize
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

        # optional weighting on embedding level
        emb = emb * w
        parts.append(emb)

    fused_emb = np.concatenate(parts, axis=1)

    # final normalize
    fused_emb = fused_emb / (np.linalg.norm(fused_emb, axis=1, keepdims=True) + 1e-12)
    return fused_emb



def create_simple_ensemble(config, save_dir):

    # either branch in main due to base_name or directly enter here
    #base_config = load_toml_config(args.base_config)
    #experiment_config = load_toml_config(args.experiment_config)

    parquet_root = resolve_path(config["data"]["split_data_path"], EXPERIMENTS_STORE)
    data_path = PATHS.data_export / "splits_curated"
    split_df = pd.read_parquet(parquet_root)

    members = config["members"]
    weights = config["fusion"]["weights"]

    normalize_mode = config["fusion"].get("normalize_mode", "global_minmax")
    square_before_fusion = config["fusion"].get("square_before_fusion", True)

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

    train_local_to_emb_row = build_local_to_emb_row(train_ds, split_df, split="train")
    val_local_to_emb_row = build_local_to_emb_row(val_ds, split_df, split="val")

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
        train_ds.transform = model.backbone_wrapper.transform

        cache_prefix = member.get("cache_prefix", member["name"])

        query_embeddings = load_or_extract_jaguarid_embeddings_cached(
            model=model,
            torch_ds=val_ds,
            config=config,
            member_name=member["name"],
            split_name="val",
            cache_prefix=cache_prefix,
        )

        train_embeddings = load_or_extract_jaguarid_embeddings_cached(
            model=model,
            torch_ds=train_ds,
            config=config,
            member_name=member["name"],
            split_name="train",
            cache_prefix=cache_prefix,
        )

        gallery_embeddings = np.concatenate([train_embeddings, query_embeddings], axis=0)
        query_labels = np.asarray(val_ds.labels)
        gallery_labels = np.concatenate([np.asarray(train_ds.labels), np.asarray(val_ds.labels)], axis=0)

        query_global_indices = val_local_to_emb_row
        gallery_global_indices = np.concatenate(
            [train_local_to_emb_row, val_local_to_emb_row],
            axis=0,
        )

        #sim_matrix_square = cosine_similarity_matrix_square(
        #    embeddings,
        #    use_qe=config["inference"]["use_qe"],
        #    use_rerank=config["inference"]["use_rerank"]
        #)

        sim_matrix_rect = cosine_similarity_matrix_rect(query_embeddings, gallery_embeddings)
        

        """
        print("\nSimilarity statistics:")
        print(f"  Min:  {sim_matrix_square.min():.4f}")
        print(f"  Max:  {sim_matrix_square.max():.4f}")
        print(f"  Mean: {sim_matrix_square.mean():.4f}")
        print(f"  Std:  {sim_matrix_square.std():.4f}")

        
        member_outputs[member["name"]] = {
            "embeddings": embeddings,
            "sim_matrix": sim_matrix,
            "weights": config["fusion"]["weights"][i]
        }
        """

        print(f"  Min:  {sim_matrix_rect.min():.4f}")
        print(f"  Max:  {sim_matrix_rect.max():.4f}")
        print(f"  Mean: {sim_matrix_rect.mean():.4f}")
        print(f"  Std:  {sim_matrix_rect.std():.4f}")

        member_outputs[member["name"]] = {
            "query_embeddings": query_embeddings,
            "gallery_embeddings": gallery_embeddings,
            "sim_matrix": sim_matrix_rect,
            "weight": config["fusion"]["weights"][i],
        }


        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------
    # 5. FUSE SIMILARITY MATRICES
    # --------------------------------------------------

    sim_mats = [member_outputs[member["name"]]["sim_matrix"] for member in members]

    print("\nFusing similarity matrices...")
    
    fused_sim_matrix = fuse_similarity_matrices(
        sim_mats=sim_mats,
        weights=weights,
        normalize_mode=normalize_mode,
        square_before_fusion=square_before_fusion,
    )


    print("\nFused similarity statistics:")
    print(f"  Min:  {fused_sim_matrix.min():.4f}")
    print(f"  Max:  {fused_sim_matrix.max():.4f}")
    print(f"  Mean: {fused_sim_matrix.mean():.4f}")
    print(f"  Std:  {fused_sim_matrix.std():.4f}")

    # --------------------------------------------------
    # 6. EMBEDDING FUSION (concat + normalize)
    # --------------------------------------------------
    
    """
    embedding_list = [member_outputs[member["name"]]["embeddings"] for member in members]

    fused_embeddings = fuse_embeddings_concat(
        embeddings_list=embedding_list,
        weights=weights,
    )

    fused_embedding_sim_matrix = fused_embeddings @ fused_embeddings.T
    """

    query_embedding_list = [member_outputs[member["name"]]["query_embeddings"] for member in members]
    gallery_embedding_list = [member_outputs[member["name"]]["gallery_embeddings"] for member in members]

    fused_query_embeddings = fuse_embeddings_concat(
        embeddings_list=query_embedding_list,
        weights=weights,
    )

    fused_gallery_embeddings = fuse_embeddings_concat(
        embeddings_list=gallery_embedding_list,
        weights=weights,
    )

    fused_embedding_sim_matrix = fused_query_embeddings @ fused_gallery_embeddings.T

    print("\nFused embedding similarity statistics:")
    print(f"  Min:  {fused_embedding_sim_matrix.min():.4f}")
    print(f"  Max:  {fused_embedding_sim_matrix.max():.4f}")
    print(f"  Mean: {fused_embedding_sim_matrix.mean():.4f}")
    print(f"  Std:  {fused_embedding_sim_matrix.std():.4f}")
    

    return {
        "member_outputs": member_outputs,
        "fused_sim_matrix": fused_sim_matrix,
        "fused_query_embeddings": fused_query_embeddings,
        "fused_gallery_embeddings": fused_gallery_embeddings,
        "fused_embedding_sim_matrix": fused_embedding_sim_matrix,
        "query_labels": query_labels,
        "gallery_labels": gallery_labels,
        "query_global_indices": query_global_indices,
        "gallery_global_indices": gallery_global_indices,
        "split_df": split_df,
    }