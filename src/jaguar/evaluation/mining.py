import numpy as np
from typing import Any, Optional, Tuple, Dict, List

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
    Among the first `top_m` valid imposters (closest different-ID neighbors),
    choose the one with best image quality.
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
    Among the first `top_m` valid positives (same-ID neighbors in ranked order),
    choose the one with best image quality.
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


def hard_negative(
    i: int,
    sim: np.ndarray,
    ranked: np.ndarray,
    labels,
    imposter_rank: int = 1,
    cluster_id: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, float]]:
    """
    Same as hard_negative, but returns None instead of raising if rank doesn't exist.
    """
    count = 0
    for j in ranked[i]:
        j = int(j)
        if _allow_pair(i, j, labels, cluster_id, same_id_required=False):
            count += 1
            if count == imposter_rank:
                return j, float(sim[i, j])
    return None


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


def mining_pack_for_query(
    i: int,
    sim: np.ndarray,
    ranked: np.ndarray,
    labels,
    cluster_id: Optional[np.ndarray] = None,
    neg_ranks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, Any]:
    """
    Returns a structured dict of recommended pairs for CAM analysis.
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