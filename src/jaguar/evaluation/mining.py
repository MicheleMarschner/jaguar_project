"""
Pair mining utilities for Jaguar re-ID analysis and XAI/CAM inspection.

Project role:
- selects informative reference pairs for a given query image (easy/hard positive/negative)
- supports duplicate-aware filtering via burst/cluster IDs
- supports quality-aware pair selection (sharpness + brightness heuristic)
- returns compact pair metadata used by downstream visualization / explanation analysis

This is analysis/mining logic (not training loss mining).

Pair taxonomy used throughout analysis:
 - easy positive: same ID, highest similarity
 - hard positive: same ID, lowest similarity
 - hard negative: different ID, high similarity (ranked imposter)
 - easy negative: different ID, lowest similarity
"""

import numpy as np
from typing import Any, Optional, Sequence, Tuple, Dict, List

from jaguar.utils.utils_mining import quality_score

def best_quality_from_top_m_imposters(
    i: int,
    sim: np.ndarray,
    ranked: np.ndarray,
    labels,
    filepaths: List[str],
    sharpness_fn,
    top_m: int = 20,
    cluster_id: Optional[np.ndarray] = None,
) -> Tuple[int, float, Dict[str, float]]:
    """
    Among *already retrieval-relevant* imposters (closest different-ID neighbors), prefer the visually cleaner image.
    This improves interpretability of CAM/XAI examples without changing the pair type.
    Returns: (j, sim_ij, {"quality":..., "sharpness":..., "brightness":...})
    """
    pool = []
    for j in ranked[i]:
        j = int(j)
        if _allow_pair(i, j, labels, cluster_id, same_id_required=False):
            q, sharp, bright = quality_score(filepaths[j], sharpness_fn)
            pool.append((q, j, float(sim[i, j]), sharp, bright))
            if len(pool) >= top_m:
                break

    if not pool:
        raise ValueError(f"No imposters found for i={i} within top_m={top_m}.")

    pool.sort(key=lambda t: t[0], reverse=True)
    q, j, s, sharp, bright = pool[0]
    return j, s, {"quality": q, "sharpness": sharp, "brightness": bright}


def best_quality_from_top_m_positives(
    i: int,
    sim: np.ndarray,
    ranked: np.ndarray,
    labels,
    filepaths: List[str],
    sharpness_fn,
    top_m: int = 20,
    cluster_id: Optional[np.ndarray] = None,
) -> Tuple[int, float, Dict[str, float]]:
    """
    Same idea for positives: keep retrieval relevance, then choose a cleaner reference image
    for more reliable qualitative inspection.
    Returns: (j, sim_ij, quality_dict)
    """
    pool = []
    for j in ranked[i]:
        j = int(j)
        if _allow_pair(i, j, labels, cluster_id, same_id_required=True):
            q, sharp, bright = quality_score(filepaths[j], sharpness_fn)
            pool.append((q, j, float(sim[i, j]), sharp, bright))
            if len(pool) >= top_m:
                break

    if not pool:
        raise ValueError(f"No positives found for i={i} within top_m={top_m}.")

    pool.sort(key=lambda t: t[0], reverse=True)
    q, j, s, sharp, bright = pool[0]
    return j, s, {"quality": q, "sharpness": sharp, "brightness": bright}


# Shared pair-validity gate used by all mining strategies so pair definitions stay consistent
# (identity relation + duplicate/burst filtering).
def _allow_pair(
    i: int,
    j: int,
    labels,
    cluster_id: Optional[np.ndarray],
    same_id_required: Optional[bool],
) -> bool:
    """Internal filter for candidate pairs."""
    if j == i:
        return False

    if same_id_required is True and labels[j] != labels[i]:
        return False
    if same_id_required is False and labels[j] == labels[i]:
        return False

    if cluster_id is not None:
        ci = int(cluster_id[i])
        # exclude within same burst/duplicate cluster if query is clustered
        if ci != -1 and int(cluster_id[j]) == ci:
            return False

    return True


# "Anchor-preserving" positive: visually/embedding-similar same-ID example.
# Useful as a stable reference pair in qualitative explanation comparisons.
def easy_positive(
    i: int,
    sim: np.ndarray,
    ranked: np.ndarray,
    labels,
    cluster_id: Optional[np.ndarray] = None,
) -> Tuple[int, float]:
    """
    Easiest positive = highest-sim same-ID neighbor (first same-ID in ranked list).
    Returns: (j, sim_ij)
    """
    for j in ranked[i]:
        j = int(j)
        if _allow_pair(i, j, labels, cluster_id, same_id_required=True):
            return j, float(sim[i, j])
    raise ValueError(f"No positive found for i={i} (identity has only one sample or all filtered).")


# Challenging same-ID pair: tests intra-identity variation (pose, lighting, occlusion, background).
# Often more informative than easy positives for XAI/CAM comparisons.
def hard_positive(
    i: int,
    sim: np.ndarray,
    ranked: np.ndarray,
    labels,
    cluster_id: Optional[np.ndarray] = None,
    within_top_k: Optional[int] = None,
) -> Tuple[int, float]:
    """
    Hard positive = same-ID neighbor with LOWEST similarity.

    within_top_k:
      - If None: consider all same-ID occurrences present in ranked[i]
      - If set (e.g., 200): only consider first K retrievals (faster)
    Returns: (j, sim_ij)
    """
    order = ranked[i] if within_top_k is None else ranked[i][:within_top_k]

    candidates = []
    for j in order:
        j = int(j)
        if _allow_pair(i, j, labels, cluster_id, same_id_required=True):
            candidates.append((j, float(sim[i, j])))

    if not candidates:
        raise ValueError(f"No hard positive for i={i} (all filtered or none exist).")
    return min(candidates, key=lambda t: t[1])  # lowest sim


# Retrieval-confusing imposter (different ID but high similarity).
# Core failure-analysis pair type for re-ID explanations.
def hard_negative(
    i: int,
    sim: np.ndarray,
    ranked: np.ndarray,
    labels,
    imposter_rank: int = 1,
    cluster_id: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, float]]:
    """
    Return the k-th valid hard negative (imposter) in ranked order. 
    Returns None if fewer than `imposter_rank` valid imposters exist.
    """
    count = 0
    for j in ranked[i]:
        j = int(j)
        if _allow_pair(i, j, labels, cluster_id, same_id_required=False):
            count += 1
            if count == imposter_rank:
                return j, float(sim[i, j])
    return None


# Trivially different imposter (very low similarity).
# Useful as a contrast case when hard negatives are too subtle.
def easy_negative(
    i: int,
    sim: np.ndarray,
    ranked: np.ndarray,
    labels,
    cluster_id: Optional[np.ndarray] = None,
) -> Tuple[int, float]:
    """
    Easy negative = farthest different-ID among ranked list (reverse scan).
    Returns: (j, sim_ij)
    """
    for j in ranked[i][::-1]:
        j = int(j)
        if _allow_pair(i, j, labels, cluster_id, same_id_required=False):
            return j, float(sim[i, j])
    raise ValueError(f"No easy negative found for i={i}.")


## TODO Kann man das generischer machen?
# Batch helper used when we need one deterministic reference pair per query
# (e.g., precomputing pair lists/artifacts for later analysis runs).
def build_easy_positive_pairs(
    query_indices: Sequence[int],
    sim: np.ndarray,
    ranked: np.ndarray,
    labels: np.ndarray,
    easy_positive_fn,   # pass your jaguar.evaluation.mining.easy_positive
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      ref_indices [N]
      pair_sims   [N]
    """
    q = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    ref_indices = []
    pair_sims = []

    for i in q:
        j, s = easy_positive_fn(
            int(i),
            sim=sim,
            ranked=ranked,
            labels=labels,
        )
        ref_indices.append(int(j))
        pair_sims.append(float(s))

    return np.asarray(ref_indices, dtype=np.int64), np.asarray(pair_sims, dtype=np.float32)


def mining_pack_for_query(
    i: int,
    sim: np.ndarray,
    ranked: np.ndarray,
    labels,
    cluster_id: Optional[np.ndarray] = None,
    neg_ranks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, Any]:
    """
    One-stop mining output for a query image: standardized pair set for visualization/CAM pipelines 
    (easy positive, hard positive, and selected hard negatives).
    """
    pos_e_i, pos_e_sim = easy_positive(i, sim, ranked, labels, cluster_id)
    pos_h_i, pos_h_sim = hard_positive(i, sim, ranked, labels, cluster_id)

    negs = []
    for r in neg_ranks:
        res = hard_negative(i, sim, ranked, labels, imposter_rank=r, cluster_id=cluster_id)
        if res is None:
            # not enough imposters for this rank -> skip it
            continue
        n_i, n_sim = res
        negs.append({"rank": r, "idx": n_i, "sim": n_sim})

    return {
        "q": i,
        "q_label": labels[i],
        "pos_easy": {"idx": pos_e_i, "sim": pos_e_sim},
        "pos_hard": {"idx": pos_h_i, "sim": pos_h_sim},
        "negs": negs,
    }