import numpy as np
import pandas as pd
from typing import Any

from jaguar.utils.utils_evaluation import l2_normalize


def _build_member_dicts(
    out: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
]:
    """Collect per-member query embeddings, gallery embeddings, similarity matrices, and weights."""
    query_embs = {}
    gallery_embs = {}
    sim_mats = {}
    weights = {}

    for name, member_out in out["member_outputs"].items():
        query_embs[name] = member_out["query_embeddings"]
        gallery_embs[name] = member_out["gallery_embeddings"]
        sim_mats[name] = member_out["sim_matrix"]
        weights[name] = float(member_out["weight"])

    return query_embs, gallery_embs, sim_mats, weights


def normalize_none(sim: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Leave scores unchanged."""
    return sim


def normalize_global_minmax(sim: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Global min-max normalization."""
    sim_min = sim.min()
    sim_max = sim.max()
    return (sim - sim_min) / (sim_max - sim_min + eps)


def normalize_row_minmax(sim: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise min-max normalization."""
    row_min = sim.min(axis=1, keepdims=True)
    row_max = sim.max(axis=1, keepdims=True)
    return (sim - row_min) / (row_max - row_min + eps)


def normalize_row_zscore(sim: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise z-score normalization."""
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
    sim_mats: dict[str, np.ndarray],
    weights: dict[str, float],
    normalize_mode: str = "global_minmax",
    square_before_fusion: bool = True,
) -> np.ndarray:
    """Fuse member similarity matrices by weighted sum after optional normalization."""
    if not sim_mats:
        raise ValueError("sim_mats must not be empty")
    if normalize_mode not in NORMALIZERS:
        raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

    model_names = list(sim_mats.keys())
    normalize_fn = NORMALIZERS[normalize_mode]

    weight_vec = np.asarray([weights[name] for name in model_names], dtype=np.float64)
    weight_vec = weight_vec / weight_vec.sum()

    fused = np.zeros_like(sim_mats[model_names[0]], dtype=np.float64)

    for w, name in zip(weight_vec, model_names):
        sim = normalize_fn(np.asarray(sim_mats[name], dtype=np.float64))
        if square_before_fusion:
            sim = sim ** 2
        fused += w * sim

    return fused


def run_score_fusion(out: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Build weighted score fusion from member similarity matrices."""
    _, _, sim_mats, weights = _build_member_dicts(out)

    normalize_mode = config["fusion"].get("normalize_mode", "global_minmax")
    square_before_fusion = config["fusion"].get("square_before_fusion", True)

    fused_sim = fuse_similarity_matrices(
        sim_mats=sim_mats,
        weights=weights,
        normalize_mode=normalize_mode,
        square_before_fusion=square_before_fusion,
    )

    return {
        "name": "score_fusion",
        "sim_matrix": fused_sim,
        "meta": {
            "family": "score",
            "normalize_mode": normalize_mode,
            "square_before_fusion": bool(square_before_fusion),
            "n_members_used": len(sim_mats),
        },
        "artifacts": {},
    }


def fuse_embeddings_concat(
    embeddings_list: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Fuse embeddings by weighted concatenation followed by final L2 normalization."""
    if not embeddings_list:
        raise ValueError("embeddings_list must not be empty")

    n = embeddings_list[0].shape[0]
    for emb in embeddings_list:
        if emb.shape[0] != n:
            raise ValueError("All embedding arrays must have the same number of rows")

    if weights is None:
        weights = np.ones(len(embeddings_list), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != len(embeddings_list):
            raise ValueError("weights must match number of embedding arrays")

    parts = []
    for emb, w in zip(embeddings_list, weights):
        emb = np.asarray(emb, dtype=np.float64)
        emb = l2_normalize(emb)
        parts.append(emb * w)

    fused_emb = np.concatenate(parts, axis=1)
    fused_emb = l2_normalize(fused_emb)
    return fused_emb


def run_embedding_concat_fusion(
    out: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build weighted concatenation fusion from member embeddings."""
    query_embs, gallery_embs, _, weights = _build_member_dicts(out)

    member_names = list(query_embs.keys())
    weight_list = [weights[name] for name in member_names]

    fused_query = fuse_embeddings_concat(
        embeddings_list=[query_embs[name] for name in member_names],
        weights=weight_list,
    )
    fused_gallery = fuse_embeddings_concat(
        embeddings_list=[gallery_embs[name] for name in member_names],
        weights=weight_list,
    )

    fused_sim = fused_query @ fused_gallery.T

    return {
        "name": "embedding_concat",
        "sim_matrix": fused_sim,
        "meta": {
            "family": "embedding_concat",
            "members_used": "|".join(member_names),
            "n_members_used": len(member_names),
            "embedding_dim": int(fused_query.shape[1]),
        },
        "artifacts": {},
    }


def run_same_dim_mean_embedding_fusion(
    out: dict[str, Any],
    config: dict[str, Any],
    method_name: str = "embedding_mean_selected",
    selected_members: list[str] | None = None,
) -> dict[str, Any]:
    """Average same-dimensional member embeddings, then re-normalize."""
    query_embs, gallery_embs, _, _ = _build_member_dicts(out)

    member_names = selected_members or list(query_embs.keys())
    if len(member_names) < 2:
        raise ValueError("Need at least 2 members for same-dim mean fusion.")

    query_dims = [query_embs[name].shape[1] for name in member_names]
    gallery_dims = [gallery_embs[name].shape[1] for name in member_names]

    if len(set(query_dims)) != 1 or len(set(gallery_dims)) != 1:
        raise ValueError(
            f"{method_name} requires same-dimensional embeddings, got query dims {query_dims} and gallery dims {gallery_dims}"
        )

    query_stack = np.stack([l2_normalize(query_embs[name]) for name in member_names], axis=0)
    gallery_stack = np.stack([l2_normalize(gallery_embs[name]) for name in member_names], axis=0)

    fused_query = l2_normalize(query_stack.mean(axis=0))
    fused_gallery = l2_normalize(gallery_stack.mean(axis=0))
    fused_sim = fused_query @ fused_gallery.T

    return {
        "name": method_name,
        "sim_matrix": fused_sim,
        "meta": {
            "family": "embedding_mean",
            "members_used": "|".join(member_names),
            "n_members_used": len(member_names),
            "embedding_dim": int(query_dims[0]),
        },
        "artifacts": {},
    }


def build_fusion_suite_results(
    out: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build all fusion variants evaluated by the ensemble runner."""
    results = [
        run_score_fusion(out, config),
        run_embedding_concat_fusion(out, config),
    ]

    mean_members = config.get("fusion_suite", {}).get("mean_embedding_members")
    if mean_members:
        results.append(
            run_same_dim_mean_embedding_fusion(
                out=out,
                config=config,
                method_name="embedding_mean_selected",
                selected_members=mean_members,
            )
        )

    return results