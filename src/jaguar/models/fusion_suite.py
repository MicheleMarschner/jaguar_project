from typing import Any

import numpy as np
import pandas as pd


def compute_query_confidences(
    sim_matrix: np.ndarray,
    top_k: int = 2,
    agg: str = "mean",
) -> np.ndarray:
    """Compute one confidence value per query: top1 - aggregate(top2..topK)."""
    if top_k < 2:
        raise ValueError("top_k must be >= 2")
    if agg not in {"mean", "max"}:
        raise ValueError("agg must be 'mean' or 'max'")

    n_queries, n_gallery = sim_matrix.shape
    k = min(top_k, n_gallery)

    sorted_scores = np.sort(sim_matrix, axis=1)[:, ::-1]
    top1 = sorted_scores[:, 0]
    competitors = sorted_scores[:, 1:k]

    if competitors.shape[1] == 0:
        raise ValueError("Need at least one competitor score.")

    if agg == "mean":
        ref = competitors.mean(axis=1)
    else:
        ref = competitors.max(axis=1)

    conf = top1 - ref
    return np.maximum(conf, 0.0)


def compute_adaptive_weights(
    confidences: dict[str, np.ndarray],
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Convert per-model confidences into per-query fusion weights."""
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if not confidences:
        raise ValueError("confidences must not be empty")

    model_names = list(confidences.keys())
    conf_stack = np.stack(
        [np.maximum(confidences[name], 0.0) for name in model_names],
        axis=0,
    )  # [n_models, n_queries]

    conf_power = np.power(conf_stack, alpha)
    denom = conf_power.sum(axis=0, keepdims=True) + eps
    weight_stack = conf_power / denom

    return {name: weight_stack[i] for i, name in enumerate(model_names)}


def fuse_similarity_matrices_query_weighted(
    sim_mats: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    """Fuse similarity matrices with query-specific weights."""
    if not sim_mats:
        raise ValueError("sim_mats must not be empty")

    model_names = list(sim_mats.keys())
    ref_shape = sim_mats[model_names[0]].shape
    fused = np.zeros(ref_shape, dtype=np.float32)

    for name in model_names:
        sim = sim_mats[name]
        w = weights[name]

        if sim.shape != ref_shape:
            raise ValueError(f"Shape mismatch for {name}: {sim.shape} vs {ref_shape}")
        if w.shape[0] != ref_shape[0]:
            raise ValueError(f"Weight length mismatch for {name}")

        fused += sim * w[:, None]

    return fused


def run_score_fusion(out: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Return the existing fixed score-fusion result."""
    return {
        "name": "score_fusion",
        "sim_matrix": out["fused_sim_matrix"],
        "meta": {"family": "score"},
        "artifacts": {},
    }


def run_embedding_concat_fusion(
    out: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return the existing concatenated-embedding fusion result."""
    return {
        "name": "embedding_concat",
        "sim_matrix": out["fused_embedding_sim_matrix"],
        "meta": {"family": "embedding_concat"},
        "artifacts": {},
    }


def run_adaptive_margin_fusion(
    out: dict[str, Any],
    method_name: str,
    top_k: int = 2,
    agg: str = "mean",
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Build query-adaptive score fusion from member similarity matrices."""
    sim_mats = {
        name: member_out["sim_matrix"]
        for name, member_out in out["member_outputs"].items()
    }

    confidences = {
        name: compute_query_confidences(sim, top_k=top_k, agg=agg)
        for name, sim in sim_mats.items()
    }

    weights = compute_adaptive_weights(confidences, alpha=alpha)
    fused_sim = fuse_similarity_matrices_query_weighted(sim_mats, weights)

    weight_df = pd.DataFrame(weights)
    weight_df.insert(0, "query_idx", np.arange(len(weight_df)))

    conf_df = pd.DataFrame(confidences)
    conf_df.insert(0, "query_idx", np.arange(len(conf_df)))

    return {
        "name": method_name,
        "sim_matrix": fused_sim,
        "meta": {
            "family": "adaptive_score",
            "top_k": int(top_k),
            "agg": agg,
            "alpha": float(alpha),
        },
        "artifacts": {
            "query_weights": weight_df,
            "query_confidences": conf_df,
        },
    }


def run_rrf_fusion(
    out: dict[str, Any],
    method_name: str = "rrf",
    k: int = 60,
) -> dict[str, Any]:
    """Build Reciprocal Rank Fusion from member similarity matrices."""
    if k <= 0:
        raise ValueError("k must be > 0")

    sim_mats = {
        name: member_out["sim_matrix"]
        for name, member_out in out["member_outputs"].items()
    }

    model_names = list(sim_mats.keys())
    ref_shape = sim_mats[model_names[0]].shape
    n_queries, n_gallery = ref_shape

    rrf_scores = np.zeros(ref_shape, dtype=np.float32)

    for name in model_names:
        sim = sim_mats[name]
        if sim.shape != ref_shape:
            raise ValueError(f"Shape mismatch for {name}: {sim.shape} vs {ref_shape}")

        # descending order per query
        ranked_idx = np.argsort(-sim, axis=1)

        # ranks[row, col] = 1-based rank of gallery col for that query
        ranks = np.empty_like(ranked_idx, dtype=np.int32)
        row_ids = np.arange(n_queries)[:, None]
        ranks[row_ids, ranked_idx] = np.arange(1, n_gallery + 1, dtype=np.int32)[None, :]

        rrf_scores += 1.0 / (k + ranks)

    return {
        "name": method_name,
        "sim_matrix": rrf_scores,
        "meta": {
            "family": "rank_fusion",
            "rrf_k": int(k),
        },
        "artifacts": {},
    }


def run_winner_takes_most_fusion(
    out: dict[str, Any],
    method_name: str,
    top_k: int = 5,
    agg: str = "mean",
    alpha: float = 1.0,
    delta_threshold: float = 0.05,
    strong_weight: float = 0.85,
) -> dict[str, Any]:
    """Build query-adaptive fusion that strongly favors one model only in clear-confidence cases."""
    sim_mats = {
        name: member_out["sim_matrix"]
        for name, member_out in out["member_outputs"].items()
    }

    model_names = list(sim_mats.keys())
    if len(model_names) != 2:
        raise ValueError("run_winner_takes_most_fusion currently supports exactly 2 models.")

    a, b = model_names

    confidences = {
        name: compute_query_confidences(sim, top_k=top_k, agg=agg)
        for name, sim in sim_mats.items()
    }

    conf_a = np.power(np.maximum(confidences[a], 0.0), alpha)
    conf_b = np.power(np.maximum(confidences[b], 0.0), alpha)

    delta = conf_a - conf_b
    weak_weight = 1.0 - strong_weight

    w_a = np.full_like(conf_a, 0.5, dtype=np.float32)
    w_b = np.full_like(conf_b, 0.5, dtype=np.float32)

    mask_a = delta > delta_threshold
    mask_b = delta < -delta_threshold

    w_a[mask_a] = strong_weight
    w_b[mask_a] = weak_weight

    w_a[mask_b] = weak_weight
    w_b[mask_b] = strong_weight

    weights = {
        a: w_a,
        b: w_b,
    }

    fused_sim = fuse_similarity_matrices_query_weighted(sim_mats, weights)

    weight_df = pd.DataFrame(weights)
    weight_df.insert(0, "query_idx", np.arange(len(weight_df)))
    weight_df["delta"] = delta
    weight_df["selected_model"] = np.where(
        mask_a, a,
        np.where(mask_b, b, "tie")
    )

    conf_df = pd.DataFrame(confidences)
    conf_df.insert(0, "query_idx", np.arange(len(conf_df)))

    return {
        "name": method_name,
        "sim_matrix": fused_sim,
        "meta": {
            "family": "winner_takes_most",
            "top_k": int(top_k),
            "agg": agg,
            "alpha": float(alpha),
            "delta_threshold": float(delta_threshold),
            "strong_weight": float(strong_weight),
        },
        "artifacts": {
            "query_weights": weight_df,
            "query_confidences": conf_df,
        },
    }

def build_fusion_suite_results(
    out: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build all fusion variants to evaluate in one shared loop."""

    member_names = list(out["member_outputs"].keys())
    n_members = len(member_names)

    results = []

    results.append(run_score_fusion(out, config))
    results.append(run_embedding_concat_fusion(out, config))
    
    if n_members >= 2:
        results.append(
            run_adaptive_margin_fusion(
                out=out,
                method_name="adaptive_margin_top5",
                top_k=5,
                agg="mean",
                alpha=1.0,
            )
        )
        results.append(
            run_rrf_fusion(
                out=out,
                method_name="rrf_k60",
                k=60,
            )
        )

    if n_members == 2:
        results.append(
            run_winner_takes_most_fusion(
                out=out,
                method_name="winner_takes_most_top5",
                top_k=5,
                agg="mean",
                alpha=1.0,
                delta_threshold=0.05,
                strong_weight=0.85,
            )
        )

    return results