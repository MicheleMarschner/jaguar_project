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

from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.utils.utils_evaluation import build_local_to_emb_row
from jaguar.utils.utils_models import load_or_extract_embeddings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
from jaguar.utils.utils import ensure_dir, read_toml_from_path, resolve_path, save_parquet
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from jaguar.config import EXPERIMENTS_STORE, PATHS, DEVICE
from jaguar.models.jaguarid_models import JaguarIDModel




def get_embedding_cache_path(
    config,
    member_name: str,
    split_name: str,
    cache_prefix: str | None = None,
) -> Path:
    ensemble_name = config["ensemble"]["name"]
    cache_dir = PATHS.runs / "ensemble_cache" / ensemble_name / member_name
    ensure_dir(cache_dir)
    return cache_dir / f"{split_name}_embeddings.npy"


def load_or_extract_embeddings_cached(
    model,
    dataloader,
    config,
    member_name: str,
    split_name: str,
):
    cache_path = get_embedding_cache_path(config, member_name, split_name)

    if cache_path.exists():
        print(f"[Cache] Loading embeddings from {cache_path}")
        emb = np.load(cache_path)
        print(f"[Cache] Loaded shape: {emb.shape}")
        return emb

    emb = extract_embeddings(
        model=model,
        dataloader=dataloader,
        use_tta=config["inference"]["use_tta"],
    )

    np.save(cache_path, emb)
    print(f"[Cache] Saved embeddings to {cache_path}")
    return emb


def evaluate_query_gallery_retrieval(
    torch_ds,
    query_indices: np.ndarray,
    gallery_indices: np.ndarray,
    split_df,
    model_wrapper: FoundationModelWrapper,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate retrieval on a rectangular query->gallery setup using the shared core logic.

    Returns:
    - query_df: one row per query with AP / rank1 info
    - summary: aggregate metrics
    """
    retrieval = prepare_query_gallery_retrieval(
        torch_ds=torch_ds,
        query_indices=query_indices,
        gallery_indices=gallery_indices,
        split_df=split_df,
        model_wrapper=model_wrapper,
    )

    query_rows = []
    ap_list = []
    rank1_hits = []

    n_queries = len(retrieval["q_global"])

    for i in tqdm(range(n_queries), desc="Evaluating Retrieval"):
        q_idx_global, q_label, ranked_candidates = get_ranked_candidates_for_query(retrieval, i)

        if len(ranked_candidates) == 0:
            continue

        rels = np.array([int(c["is_same_id"]) for c in ranked_candidates], dtype=np.int64)
        sims = np.array([c["sim"] for c in ranked_candidates], dtype=np.float64)

        num_rel = rels.sum()
        if num_rel == 0:
            continue

        precision_at_k = np.cumsum(rels) / (np.arange(len(rels)) + 1)
        ap = float((precision_at_k * rels).sum() / num_rel)

        rank1_correct = bool(ranked_candidates[0]["is_same_id"])
        first_pos_rank = int(np.where(rels == 1)[0][0] + 1)

        ap_list.append(ap)
        rank1_hits.append(rank1_correct)

        query_rows.append({
            "query_idx": q_idx_global,
            "query_label": q_label,
            "n_gallery_valid": len(ranked_candidates),
            "n_relevant": int(num_rel),
            "rank1_correct": rank1_correct,
            "ap": ap,
            "first_pos_rank": first_pos_rank,
            "top1_idx": ranked_candidates[0]["gallery_global_idx"],
            "top1_label": ranked_candidates[0]["gallery_label"],
            "top1_sim": ranked_candidates[0]["sim"],
        })

    query_df = pd.DataFrame(query_rows)

    summary = {
        "mAP": float(np.mean(ap_list)) if ap_list else 0.0,
        "rank1": float(np.mean(rank1_hits)) if rank1_hits else 0.0,
        "n_queries_eval": len(ap_list),
    }

    return query_df, summary


def prepare_query_gallery_retrieval(
    torch_ds,
    query_indices: np.ndarray,
    gallery_indices: np.ndarray,
    split_df,
    model_wrapper: FoundationModelWrapper,
):
    """
    Build rectangular query-gallery retrieval state.
    """
    all_embeddings = load_or_extract_embeddings(model_wrapper, torch_ds, num_workers=0)

    q_global = np.asarray(query_indices, dtype=np.int64)
    g_global = np.asarray(gallery_indices, dtype=np.int64)

    emb_q = all_embeddings[q_global]
    emb_g = all_embeddings[g_global]

    emb_q = emb_q / (np.linalg.norm(emb_q, axis=1, keepdims=True) + 1e-12)
    emb_g = emb_g / (np.linalg.norm(emb_g, axis=1, keepdims=True) + 1e-12)

    sim_matrix = emb_q @ emb_g.T

    all_labels = np.asarray(torch_ds.labels)
    labels_q = all_labels[q_global]
    labels_g = all_labels[g_global]

    bg = split_df.set_index("emb_row")["burst_group_id"]
    burst_q = bg.reindex(q_global).fillna(-1).to_numpy()
    burst_g = bg.reindex(g_global).fillna(-1).to_numpy()

    return {
        "q_global": q_global,
        "g_global": g_global,
        "sim_matrix": sim_matrix,
        "labels_q": labels_q,
        "labels_g": labels_g,
        "burst_q": burst_q,
        "burst_g": burst_g,
    }


def get_ranked_candidates_for_query(retrieval: dict, i: int):
    """
    Shared core: return valid ranked gallery candidates for one query.
    Keeps the exact filtering logic from your original function.
    """
    sims_i = retrieval["sim_matrix"][i]
    ranked_g_indices = np.argsort(-sims_i)

    q_idx_global = int(retrieval["q_global"][i])
    q_label = retrieval["labels_q"][i]

    rows = []
    valid_rank = 0

    for g_idx in ranked_g_indices:
        g_idx_global = int(retrieval["g_global"][g_idx])

        # Skip Self-Match
        if q_idx_global == g_idx_global:
            continue

        # Skip same burst group
        if retrieval["burst_q"][i] != -1 and retrieval["burst_g"][g_idx] == retrieval["burst_q"][i]:
            continue

        valid_rank += 1

        rows.append({
            "gallery_local_idx": int(g_idx),
            "gallery_global_idx": g_idx_global,
            "gallery_label": retrieval["labels_g"][g_idx],
            "sim": float(sims_i[g_idx]),
            "rank_in_gallery": valid_rank,
            "is_same_id": bool(q_label == retrieval["labels_g"][g_idx]),
        })

    return q_idx_global, q_label, rows


### !TODO for xai_similarity
def mine_references_from_gallery(
    torch_ds,
    query_indices: np.ndarray,
    gallery_indices: np.ndarray,
    split_df,
    model_wrapper: FoundationModelWrapper,
    out_root: Path,
    split: str,
    pair_types: Sequence[str],
) -> pd.DataFrame:
    """
    Mines reference pairs by searching for Queries within the Gallery.
    
    This stage defines the *cases we want to explain* (pair taxonomy):
    - easy_pos: “model should clearly match” (sanity case)
    - hard_neg: “model is likely to confuse” (failure-analysis case)
    - hard_pos: 
    """
    n_queries = len(query_indices)
    out_path = out_root / f"refs_n{n_queries}.parquet"

    if out_path.exists():
        print(f"[Info] Loading existing refs from {out_path}")
        return pd.read_parquet(out_path)
    
    print(f"[Mining] Queries: {n_queries} vs Gallery: {len(gallery_indices)}")

    retrieval = prepare_query_gallery_retrieval(
        torch_ds=torch_ds,
        query_indices=query_indices,
        gallery_indices=gallery_indices,
        split_df=split_df,
        model_wrapper=model_wrapper,
    )

    rows = []

    for i in tqdm(range(n_queries), desc="Mining Refs"):
        q_idx_global, q_label, ranked_candidates = get_ranked_candidates_for_query(retrieval, i)

        found_pairs = {pt: False for pt in pair_types}
        hard_pos_candidate = None
        need_tail_scan = ("hard_pos" in pair_types)

        for cand in ranked_candidates:
            if (not need_tail_scan) and all(found_pairs.values()):
                break

            current_sim = cand["sim"]
            g_idx_global = cand["gallery_global_idx"]
            is_same_id = cand["is_same_id"]

            if "easy_pos" in pair_types and not found_pairs.get("easy_pos", False):
                if is_same_id:
                    rows.append({
                        "pair_type": "easy_pos",
                        "query_idx": q_idx_global,
                        "ref_idx": g_idx_global,
                        "pair_sim": current_sim,
                        "rank_in_gallery": cand["rank_in_gallery"]
                    })
                    found_pairs["easy_pos"] = True

            if "hard_neg" in pair_types and not found_pairs.get("hard_neg", False):
                if not is_same_id:
                    rows.append({
                        "pair_type": "hard_neg",
                        "query_idx": q_idx_global,
                        "ref_idx": g_idx_global,
                        "pair_sim": current_sim,
                        "rank_in_gallery": cand["rank_in_gallery"]
                    })
                    found_pairs["hard_neg"] = True

            if "hard_pos" in pair_types and is_same_id:
                hard_pos_candidate = {
                    "pair_type": "hard_pos",
                    "query_idx": q_idx_global,
                    "ref_idx": g_idx_global,
                    "pair_sim": current_sim,
                    "rank_in_gallery": cand["rank_in_gallery"]
                }
                found_pairs["hard_pos"] = True

        if "hard_pos" in pair_types and hard_pos_candidate is not None:
            rows.append(hard_pos_candidate)

    ref_df = pd.DataFrame(rows)

    expected_count = n_queries * len(pair_types)
    if len(ref_df) < expected_count:
        print(f"[Warning] Mined {len(ref_df)} pairs, expected {expected_count}. Some queries lack positives.")
        
    ensure_dir(out_path.parent)
    save_parquet(out_path, ref_df)
    return ref_df




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

        val_loader = DataLoader(
            val_ds,
            batch_size=config["inference"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config["inference"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )

        query_embeddings = load_or_extract_embeddings_cached(
            model=model,
            dataloader=val_loader,
            config=config,
            member_name=member["name"],
            split_name="val",
        )

        train_embeddings = load_or_extract_embeddings_cached(
            model=model,
            dataloader=train_loader,
            config=config,
            member_name=member["name"],
            split_name="train",
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