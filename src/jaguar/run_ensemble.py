import argparse

from jaguar.models.ensemble import create_simple_ensemble
from jaguar.utils.utils import load_toml_config

import numpy as np
from sklearn.metrics import average_precision_score


import numpy as np
import pandas as pd


def top1_is_correct(sim_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Returns a boolean array of shape [N]:
    True if the top-1 non-self retrieval has the same label as the query.
    """
    sim = np.asarray(sim_matrix).copy()
    labels = np.asarray(labels)

    np.fill_diagonal(sim, -np.inf)
    top1_idx = np.argmax(sim, axis=1)
    return labels[top1_idx] == labels


def pairwise_error_overlap(
    sim_a: np.ndarray,
    sim_b: np.ndarray,
    labels: np.ndarray,
    name_a: str = "model_a",
    name_b: str = "model_b",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare top-1 correctness overlap between two models.

    Returns:
    - summary_df: counts + fractions
    - per_query_df: one row per query
    """
    labels = np.asarray(labels)

    correct_a = top1_is_correct(sim_a, labels)
    correct_b = top1_is_correct(sim_b, labels)

    both_correct = correct_a & correct_b
    only_a = correct_a & ~correct_b
    only_b = ~correct_a & correct_b
    both_wrong = ~correct_a & ~correct_b

    per_query_df = pd.DataFrame({
        "query_idx": np.arange(len(labels)),
        "label": labels,
        f"{name_a}_correct": correct_a,
        f"{name_b}_correct": correct_b,
        "both_correct": both_correct,
        f"only_{name_a}": only_a,
        f"only_{name_b}": only_b,
        "both_wrong": both_wrong,
    })

    n = len(labels)
    summary_df = pd.DataFrame([
        {"case": "both_correct", "count": int(both_correct.sum()), "fraction": both_correct.mean()},
        {"case": f"only_{name_a}", "count": int(only_a.sum()), "fraction": only_a.mean()},
        {"case": f"only_{name_b}", "count": int(only_b.sum()), "fraction": only_b.mean()},
        {"case": "both_wrong", "count": int(both_wrong.sum()), "fraction": both_wrong.mean()},
    ])

    return summary_df, per_query_df

def compute_pairwise_ap_from_sim(labels, sim_matrix):
    """
    Pairwise AP directly from a similarity matrix.
    Treats each pair as same-ID vs different-ID classification.
    """
    labels = np.asarray(labels)
    sim_matrix = np.asarray(sim_matrix)

    y_true = []
    y_scores = []

    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            y_true.append(int(labels[i] == labels[j]))
            y_scores.append(float(sim_matrix[i, j]))

    return average_precision_score(y_true, y_scores)


def compute_ib_map_from_sim(labels, sim_matrix):
    """
    Identity-balanced mAP directly from a similarity matrix.
    """
    labels = np.asarray(labels)
    sim_matrix = np.asarray(sim_matrix).copy()

    n = len(labels)
    np.fill_diagonal(sim_matrix, -np.inf)

    identity_map = {}

    for i in range(n):
        q_label = labels[i]
        scores = sim_matrix[i]
        idx_rank = np.argsort(-scores)

        ranked_labels = labels[idx_rank]
        rels = (ranked_labels == q_label).astype(int)

        num_rel = rels.sum()
        if num_rel == 0:
            continue

        precision_at_k = np.cumsum(rels) / (np.arange(len(rels)) + 1)
        ap = (precision_at_k * rels).sum() / num_rel

        identity_map.setdefault(q_label, []).append(ap)

    identity_map = {k: float(np.mean(v)) for k, v in identity_map.items()}
    ib_map = float(np.mean(list(identity_map.values()))) if identity_map else 0.0
    return ib_map, identity_map


class ReIDSimEvalBundle:
    """
    Lightweight ReID evaluation bundle for precomputed similarity matrices.

    Use this when you already have:
    - sim_matrix: [N, N]
    - labels: identity label per image

    This is ideal for:
    - score fusion ensembles
    - reranked similarity matrices
    - any retrieval output that is no longer naturally represented as embeddings
    """

    def __init__(self, sim_matrix, labels):
        self.sim_matrix_input = np.asarray(sim_matrix, dtype=np.float64)
        self.labels = np.asarray(labels)

        self._sim_matrix = None
        self._ranked_indices = None

    def similarity_matrix(self):
        if self._sim_matrix is None:
            sim = self.sim_matrix_input.copy()
            np.fill_diagonal(sim, -np.inf)
            self._sim_matrix = sim
        return self._sim_matrix

    def ranked_indices(self):
        if self._ranked_indices is None:
            self._ranked_indices = np.argsort(-self.similarity_matrix(), axis=1)
        return self._ranked_indices

    def mAP(self):
        ib_map, _ = compute_ib_map_from_sim(self.labels, self.similarity_matrix())
        return ib_map

    def identity_balanced_map(self):
        return self.mAP()

    def pairwise_ap(self):
        return compute_pairwise_ap_from_sim(self.labels, self.similarity_matrix())

    def top_k_accuracy(self, k=1):
        ranked = self.ranked_indices()
        labels = self.labels

        correct = 0
        total = len(labels)

        for i in range(total):
            query_label = labels[i]
            top_k = ranked[i][:k]
            if query_label in labels[top_k]:
                correct += 1

        return correct / total if total > 0 else 0.0

    def rank1(self):
        return self.top_k_accuracy(k=1)

    def rank5(self):
        return self.top_k_accuracy(k=5)

    def ndcg(self, k=None):
        ranked = self.ranked_indices()
        labels = self.labels
        ndcgs = []

        for i in range(len(labels)):
            query_label = labels[i]
            order = ranked[i]
            if k is not None:
                order = order[:k]

            relevance = (labels[order] == query_label).astype(int)
            if relevance.sum() == 0:
                continue

            discounts = 1 / np.log2(np.arange(2, len(relevance) + 2))
            dcg = np.sum(relevance * discounts)

            ideal_rel = np.sort(relevance)[::-1]
            idcg = np.sum(ideal_rel * discounts)

            ndcgs.append(dcg / (idcg + 1e-8))

        return float(np.mean(ndcgs)) if ndcgs else 0.0

    def mean_positive_similarity(self):
        sim = self.similarity_matrix()
        labels = self.labels

        pos_sims = []
        for i in range(len(labels)):
            same_id = (labels == labels[i])
            same_id[i] = False
            if same_id.sum() > 0:
                pos_sims.extend(sim[i][same_id])

        return float(np.mean(pos_sims)) if pos_sims else 0.0

    def mean_negative_similarity(self):
        sim = self.similarity_matrix()
        labels = self.labels

        neg_sims = []
        for i in range(len(labels)):
            diff_id = (labels != labels[i])
            neg_sims.extend(sim[i][diff_id])

        return float(np.mean(neg_sims)) if neg_sims else 0.0

    def similarity_gap(self):
        return self.mean_positive_similarity() - self.mean_negative_similarity()

    def compute_all(self, k_ndcg=10):
        id_map = self.identity_balanced_map()
        map_score = self.mAP()
        pair_ap = self.pairwise_ap()
        rank1 = self.rank1()
        rank5 = self.rank5()
        ndcg_k = self.ndcg(k=k_ndcg)
        pos_sim = self.mean_positive_similarity()
        neg_sim = self.mean_negative_similarity()
        sim_gap = self.similarity_gap()

        return {
            "id_balanced_mAP": id_map,
            "mAP": map_score,
            "pairwise_AP": pair_ap,
            "rank1": rank1,
            "rank5": rank5,
            f"nDCG@{k_ndcg}": ndcg_k,
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "sim_gap": sim_gap,
        }


import numpy as np
from sklearn.metrics import average_precision_score


class ReIDRectEvalBundle:
    def __init__(self, sim_matrix, query_labels, gallery_labels, self_in_gallery_offset=None):
        self.sim_matrix = np.asarray(sim_matrix, dtype=np.float64).copy()
        self.query_labels = np.asarray(query_labels)
        self.gallery_labels = np.asarray(gallery_labels)

        # if gallery = train + val, the query itself sits in the val tail
        if self_in_gallery_offset is not None:
            for i in range(len(self.query_labels)):
                self.sim_matrix[i, self_in_gallery_offset + i] = -np.inf

    def ranked_indices(self):
        return np.argsort(-self.sim_matrix, axis=1)

    def mAP(self):
        ranked = self.ranked_indices()
        aps = []

        for i in range(len(self.query_labels)):
            rels = (self.gallery_labels[ranked[i]] == self.query_labels[i]).astype(int)
            num_rel = rels.sum()
            if num_rel == 0:
                continue

            precision_at_k = np.cumsum(rels) / (np.arange(len(rels)) + 1)
            ap = (precision_at_k * rels).sum() / num_rel
            aps.append(ap)

        return float(np.mean(aps)) if aps else 0.0

    def rank1(self):
        ranked = self.ranked_indices()
        top1 = ranked[:, 0]
        return float(np.mean(self.gallery_labels[top1] == self.query_labels))

    def pairwise_ap(self):
        y_true = []
        y_score = []

        for i in range(len(self.query_labels)):
            for j in range(len(self.gallery_labels)):
                y_true.append(int(self.query_labels[i] == self.gallery_labels[j]))
                y_score.append(float(self.sim_matrix[i, j]))

        return average_precision_score(y_true, y_score)

    def compute_all(self):
        return {
            "mAP": self.mAP(),
            "pairwise_AP": self.pairwise_ap(),
            "rank1": self.rank1(),
        }



def top1_predictions(sim_matrix: np.ndarray) -> np.ndarray:
    sim = np.asarray(sim_matrix).copy()
    np.fill_diagonal(sim, -np.inf)
    return np.argmax(sim, axis=1)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ensemble experiment")
    parser.add_argument(
        "--base_config",
        type=str,
        required=True,
        help="Path to the base config TOML file, relative to PATHS.configs and without .toml",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        required=False,
        help="Optional ensemble override TOML, relative to PATHS.configs and without .toml",
    )
    return parser.parse_args()


def deep_update(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def main():
    args = parse_args()

    base_config = load_toml_config(args.base_config)
    
    if args.experiment_config is not None:
        experiment_config = load_toml_config(args.experiment_config)
        config = deep_update(base_config, experiment_config)
    else:
        config = base_config

    out = create_simple_ensemble(config, save_dir=None)

    query_labels = out["query_labels"]
    gallery_labels = out["gallery_labels"]
    self_offset = out["self_in_gallery_offset"]

    for name, member_out in out["member_outputs"].items():
        bundle = ReIDRectEvalBundle(
            sim_matrix=member_out["sim_matrix"],
            query_labels=query_labels,
            gallery_labels=gallery_labels,
            self_in_gallery_offset=self_offset,
        )
        metrics = bundle.compute_all()
        print(
            f"{name:12s} "
            f"mAP={metrics['mAP']:.4f} "
            f"pAP={metrics['pairwise_AP']:.4f} "
            f"rank1={metrics['rank1']:.4f}"
        )

    score_bundle = ReIDRectEvalBundle(
        sim_matrix=out["fused_sim_matrix"],
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        self_in_gallery_offset=self_offset,
    )
    score_metrics = score_bundle.compute_all()
    print(
        f"{'score_fusion':12s} "
        f"mAP={score_metrics['mAP']:.4f} "
        f"pAP={score_metrics['pairwise_AP']:.4f} "
        f"rank1={score_metrics['rank1']:.4f}"
    )

    emb_bundle = ReIDRectEvalBundle(
        sim_matrix=out["fused_embedding_sim_matrix"],
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        self_in_gallery_offset=self_offset,
    )
    emb_metrics = emb_bundle.compute_all()
    print(
        f"{'emb_fusion':12s} "
        f"mAP={emb_metrics['mAP']:.4f} "
        f"pAP={emb_metrics['pairwise_AP']:.4f} "
        f"rank1={emb_metrics['rank1']:.4f}"
    )
    print(f"Built ensemble: {config['ensemble']['name']}")
    print(f"Fused sim shape: {out['fused_sim_matrix'].shape}")



    """
    labels = out["labels"]

    # --------------------------------------------------
    # Single models
    # --------------------------------------------------
    member_metrics = {}

    for name, member_out in out["member_outputs"].items():
        bundle = ReIDSimEvalBundle(
            sim_matrix=member_out["sim_matrix"],
            labels=labels,
        )
        metrics = bundle.compute_all()
        member_metrics[name] = metrics

        print(
            f"{name:12s} "
            f"mAP={metrics['mAP']:.4f} "
            f"pAP={metrics['pairwise_AP']:.4f} "
            f"rank1={metrics['rank1']:.4f} "
            f"sim_gap={metrics['sim_gap']:.4f}"
        )

    # --------------------------------------------------
    # Score fusion
    # --------------------------------------------------
    score_bundle = ReIDSimEvalBundle(
        sim_matrix=out["fused_sim_matrix"],
        labels=labels,
    )
    score_metrics = score_bundle.compute_all()

    print(
        f"{'score_fusion':12s} "
        f"mAP={score_metrics['mAP']:.4f} "
        f"pAP={score_metrics['pairwise_AP']:.4f} "
        f"rank1={score_metrics['rank1']:.4f} "
        f"sim_gap={score_metrics['sim_gap']:.4f}"
    )

    # --------------------------------------------------
    # Embedding fusion
    # --------------------------------------------------
    emb_bundle = ReIDSimEvalBundle(
        sim_matrix=out["fused_embedding_sim_matrix"],
        labels=labels,
    )
    emb_metrics = emb_bundle.compute_all()

    print(
        f"{'emb_fusion':12s} "
        f"mAP={emb_metrics['mAP']:.4f} "
        f"pAP={emb_metrics['pairwise_AP']:.4f} "
        f"rank1={emb_metrics['rank1']:.4f} "
        f"sim_gap={emb_metrics['sim_gap']:.4f}"
    )

    # --------------------------------------------------
    # Error overlap checks
    # --------------------------------------------------
    sim_eva = out["member_outputs"]["EVA-02"]["sim_matrix"]
    sim_miewid = out["member_outputs"]["MiewID"]["sim_matrix"]
    sim_ens = out["fused_sim_matrix"]

    summary_eva_ens, _ = pairwise_error_overlap(
        sim_eva, sim_ens, labels, name_a="eva", name_b="ensemble"
    )
    print("\nEVA vs ensemble")
    print(summary_eva_ens)

    summary_miewid_ens, _ = pairwise_error_overlap(
        sim_miewid, sim_ens, labels, name_a="miewid", name_b="ensemble"
    )
    print("\nMiewID vs ensemble")
    print(summary_miewid_ens)

    # --------------------------------------------------
    # Direct prediction overlap
    # --------------------------------------------------
    top1_eva = top1_predictions(sim_eva)
    top1_miewid = top1_predictions(sim_miewid)

    print("\nTop1 identical fraction EVA vs MiewID:",
        np.mean(top1_eva == top1_miewid))

    """

if __name__ == "__main__":
    main()