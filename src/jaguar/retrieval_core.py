
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from captum.attr import IntegratedGradients
from pytorch_grad_cam import GradCAM

from typing import Dict, Any, Sequence, Tuple, List

from jaguar.config import DATA_STORE, EXPERIMENTS_STORE, PATHS, DEVICE, SEED 
from jaguar.utils.utils import ensure_dir, resolve_path, save_npy, save_parquet
from jaguar.utils.utils_datasets import load_full_jaguar_from_FO_export, load_or_extract_embeddings 
from jaguar.models.foundation_models import FoundationModelWrapper  
from jaguar.utils.utils_xai import CosineSimilarityTarget, EmbeddingForwardWrapper, SimilarityForward, find_module_name  
from jaguar.utils.utils_xai import ig_saliency_batched_similarity 

# ============================================================
# Deterministic query selection (curated val subset)
# ============================================================


def get_curated_indices(split_df: pd.DataFrame, splits: Sequence[str]) -> np.ndarray:
    # training/eval/XAI should only use kept images.
    df = split_df[
        split_df["split_final"].isin(list(splits))
        & split_df["keep_curated"].fillna(False).astype(bool)
    ]

    return df["emb_row"].astype(np.int64).to_numpy()


def sample_indices(indices: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    # Deterministic sampling
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    
    if len(indices) == 0:
        return indices
    
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=min(n_samples, len(indices)), replace=False)

    return np.sort(chosen)


def get_val_query_indices(
    split_df: pd.DataFrame,
    out_root: Path,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Cache the exact query subset so explainers/models can be compared on the same images 
    across repeated runs (reproducible qualitative analysis)
    """
    idx_path = out_root / f"xai_val_idx_n{n_samples}.npy"
    if idx_path.exists():
        return np.load(idx_path)

    val_pool = get_curated_indices(split_df, splits=["val"])
    val_chosen = sample_indices(val_pool, n_samples=n_samples, seed=seed)

    ensure_dir(idx_path.parent)
    save_npy(idx_path, val_chosen)
    return val_chosen




# ============================================================
# Mine references for pair types 
# ============================================================
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



