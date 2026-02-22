## map
## archloss

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

###############################################################
# Example usage (after validation loop):
#
# bundle = ReIDEvalBundle(
#     model=model,
#     embeddings=val_embeddings,
#     labels=val_labels,
#     device=device
# )
# 
# results = bundle.compute_all()
# print(results)
###############################################################

class ReIDEvalBundle:
    """
    Lightweight evaluation bundle for re-identification.

    Design goals:
    - Compute embeddings once
    - Compute similarity matrix once
    - Reuse for all metrics (mAP, AP, Rank-k, similarity diagnostics)
    - Fits small validation sets ~300 images)
    """

    def __init__(self, model, embeddings, labels, device="cuda"):
        self.model = model
        self.device = device

        # Convert inputs safely
        if isinstance(embeddings, torch.Tensor):
            self.input_embeddings = embeddings.detach().cpu().numpy()
        else:
            self.input_embeddings = np.asarray(embeddings)

        self.labels = np.asarray(labels)

        # Internal cache (simple, transparent)
        self._finetuned_embeddings = None
        self._sim_matrix = None
        self._ranked_indices = None

    # -------------------------------------------------
    # Core cached computations
    # -------------------------------------------------
    def finetuned_embeddings(self):
        """Forward pass through model head (e.g., MegaDescriptor fine-tuned layer)."""
        if self._finetuned_embeddings is None:
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(self.input_embeddings).to(self.device)
                emb = self.model.get_embeddings(x)
                self._finetuned_embeddings = emb.cpu().numpy()
        return self._finetuned_embeddings

    def similarity_matrix(self):
        """Cosine similarity matrix (NxN)."""
        if self._sim_matrix is None:
            emb = self.finetuned_embeddings()
            sim = cosine_similarity(emb)
            np.fill_diagonal(sim, -1.0)  # exclude self-matching
            self._sim_matrix = sim
        return self._sim_matrix

    def ranked_indices(self):
        """Indices sorted by similarity (descending) for each query."""
        if self._ranked_indices is None:
            sim = self.similarity_matrix()
            self._ranked_indices = np.argsort(-sim, axis=1)
        return self._ranked_indices

    # -------------------------------------------------
    # Core ReID Metrics
    # -------------------------------------------------
    def average_precision_per_query(self):
        """
        Standard AP per query.
        Returns list of APs (one per valid query).
        """
        ranked = self.ranked_indices()
        labels = self.labels

        ap_list = []

        for i in range(len(labels)):
            query_label = labels[i]
            order = ranked[i]

            # Remove self index if present at rank 0
            order = order[order != i]

            matches = (labels[order] == query_label).astype(int)
            n_pos = matches.sum()

            if n_pos == 0:
                continue  # skip identities with only one sample

            cumsum = np.cumsum(matches)
            precision_at_k = cumsum / (np.arange(1, len(matches) + 1))
            ap = np.sum(precision_at_k * matches) / n_pos
            ap_list.append(ap)

        return ap_list

    def mAP(self):
        """
        Standard mAP (mean AP over all queries).
        """
        aps = self.average_precision_per_query()
        return float(np.mean(aps)) if len(aps) > 0 else 0.0

    def identity_balanced_map(self):
        """
        Competition-style metric:
        Macro-average AP per identity (each jaguar equally weighted).
        """
        ranked = self.ranked_indices()
        labels = self.labels

        identity_aps = {}

        for i in range(len(labels)):
            query_label = labels[i]
            order = ranked[i]
            order = order[order != i]

            matches = (labels[order] == query_label).astype(int)
            n_pos = matches.sum()

            if n_pos == 0:
                continue

            cumsum = np.cumsum(matches)
            precision_at_k = cumsum / (np.arange(1, len(matches) + 1))
            ap = np.sum(precision_at_k * matches) / n_pos

            identity_aps.setdefault(query_label, []).append(ap)

        if not identity_aps:
            return 0.0

        identity_means = [np.mean(v) for v in identity_aps.values()]
        return float(np.mean(identity_means))

    # -------------------------------------------------
    # Retrieval Metrics 
    # -------------------------------------------------
    def top_k_accuracy(self, k=1):
        """
        Rank-k accuracy (CMC metric).
        Checks if correct identity appears in top-k retrievals.
        """
        ranked = self.ranked_indices()
        labels = self.labels

        correct = 0
        total = len(labels)

        for i in range(total):
            query_label = labels[i]
            top_k = ranked[i][:k]

            if query_label in labels[top_k]:
                correct += 1

        return correct / total

    def rank1(self):
        return self.top_k_accuracy(k=1)

    def rank5(self):
        return self.top_k_accuracy(k=5)
    
    def ndcg(self, k=None):
        """
        Normalized Discounted Cumulative Gain for ReID ranking quality.
        """
        ranked = self.ranked_indices()
        labels = self.labels
        ndcgs = []

        for i in range(len(labels)):
            query_label = labels[i]
            order = ranked[i]
            order = order[order != i]

            if k is not None:
                order = order[:k]

            relevance = (labels[order] == query_label).astype(int)
            if relevance.sum() == 0:
                continue

            # DCG
            discounts = 1 / np.log2(np.arange(2, len(relevance) + 2))
            dcg = np.sum(relevance * discounts)

            # Ideal DCG
            ideal_rel = np.sort(relevance)[::-1]
            idcg = np.sum(ideal_rel * discounts)

            ndcgs.append(dcg / (idcg + 1e-8))
        return float(np.mean(ndcgs)) if ndcgs else 0.0
    
    def recall_at_k(self, k=5):
        ranked = self.ranked_indices()
        labels = self.labels

        hits = 0
        for i in range(len(labels)):
            query_label = labels[i]
            top_k = ranked[i][:k]
            if query_label in labels[top_k]:
                hits += 1

        return hits / len(labels)
    
    def intra_inter_distance(self):
        emb = self.finetuned_embeddings()
        labels = self.labels

        intra, inter = [], []

        for i in range(len(labels)):
            dists = np.linalg.norm(emb[i] - emb, axis=1)

            same = (labels == labels[i])
            diff = (labels != labels[i])
            same[i] = False

            if same.sum() > 0:
                intra.extend(dists[same])
            inter.extend(dists[diff])

        return {
            "intra_dist": float(np.mean(intra)) if intra else 0.0,
            "inter_dist": float(np.mean(inter)) if inter else 0.0,
            "dist_gap": float(np.mean(inter) - np.mean(intra)) if intra else 0.0
        }

    # -------------------------------------------------
    # Embedding Diagnostics 
    # -------------------------------------------------
    def mean_positive_similarity(self):
        """Average similarity between same-identity images."""
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
        """Average similarity between different identities."""
        sim = self.similarity_matrix()
        labels = self.labels

        neg_sims = []
        for i in range(len(labels)):
            diff_id = (labels != labels[i])
            neg_sims.extend(sim[i][diff_id])

        return float(np.mean(neg_sims))

    def similarity_gap(self):
        """
        Separation quality of embeddings.
        Higher = better identity discrimination.
        """
        return self.mean_positive_similarity() - self.mean_negative_similarity()


    # -------------------------------------------------
    # Convenience: all metrics at once
    # -------------------------------------------------
    def compute_all(self, k_ndcg=10, k_recall=5):
        """
        One-call evaluation using shared cached computations.

        Includes:
        - Competition metric (identity-balanced mAP)
        - Standard ReID metrics (mAP, Rank-k)
        - Ranking quality (nDCG)
        - Retrieval framing (Recall@K)
        - Embedding diagnostics (similarity + distance structure)
        """

        # Core ranking metrics
        id_map = self.identity_balanced_map()
        map_score = self.mAP()
        rank1 = self.rank1()
        rank5 = self.rank5()

        # Retrieval / ranking quality
        ndcg_k = self.ndcg(k=k_ndcg)
        recall_k = self.recall_at_k(k=k_recall)

        # Embedding diagnostics (similarity space)
        pos_sim = self.mean_positive_similarity()
        neg_sim = self.mean_negative_similarity()
        sim_gap = self.similarity_gap()

        # Distance diagnostics (geometry of embedding space)
        dist_stats = self.intra_inter_distance()

        return {
            # --- Competition / primary ---
            "id_balanced_mAP": id_map,
            "mAP": map_score,

            # --- CMC / ReID standard ---
            "rank1": rank1,
            "rank5": rank5,

            # --- Ranking quality ---
            "nDCG@{}".format(k_ndcg): ndcg_k,

            # --- Retrieval framing ---
            "recall@{}".format(k_recall): recall_k,

            # --- Similarity diagnostics (cosine space) ---
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "sim_gap": sim_gap,

            # --- Distance diagnostics (L2 geometry) ---
            "intra_dist": dist_stats["intra_dist"],
            "inter_dist": dist_stats["inter_dist"],
            "dist_gap": dist_stats["dist_gap"],
        }

if __name__ == "__main__":
    import torch
    import numpy as np

    # ---- Dummy model (mimics your MegaDescriptor head) ----
    class DummyModel:
        def eval(self):
            pass

        def get_embeddings(self, x):
            # Simulate a learned projection
            return torch.nn.functional.normalize(x, dim=1)

    np.random.seed(42)
    torch.manual_seed(42)

    # ---- Simulate a small ReID validation set ----
    n_identities = 10
    images_per_id = 30  # 10 * 30 = 300 images (like your case)
    dim = 128

    labels = []
    embeddings = []

    for identity in range(n_identities):
        # Create identity cluster (same jaguar = similar embeddings)
        center = np.random.randn(dim)
        center = center / np.linalg.norm(center)

        for _ in range(images_per_id):
            emb = center + 0.1 * np.random.randn(dim)
            embeddings.append(emb)
            labels.append(identity)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    print("Embeddings shape:", embeddings.shape)
    print("Number of identities:", len(np.unique(labels)))

    # ---- Run evaluation bundle ----
    model = DummyModel()
    device = "cpu"

    bundle = ReIDEvalBundle(
        model=model,
        embeddings=embeddings,
        labels=labels,
        device=device
    )

    results = bundle.compute_all()

    # Optional advanced metrics
    ndcg_score = bundle.ndcg(k=10)
    recall5 = bundle.recall_at_k(k=5)
    dist_stats = bundle.intra_inter_distance()

    print("\n=== ReID Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print(f"nDCG@10: {ndcg_score:.4f}")
    print(f"Recall@5: {recall5:.4f}")
    print("\nDistance stats:", dist_stats)
    
if __name__ == "__main__":
    # Dummy model (mimics MegaDescriptor head) 
    class DummyModel:
        def eval(self):
            pass

        def get_embeddings(self, x):
            # Simulate a learned projection
            return torch.nn.functional.normalize(x, dim=1)

    np.random.seed(42)
    torch.manual_seed(42)

    # Simulate a small ReID validation set
    n_identities = 10
    images_per_id = 30  # 00 images in totale
    dim = 128

    labels = []
    embeddings = []

    for identity in range(n_identities):
        # Create identity cluster (same jaguar = similar embeddings)
        center = np.random.randn(dim)
        center = center / np.linalg.norm(center)

        for _ in range(images_per_id):
            emb = center + 0.1 * np.random.randn(dim)
            embeddings.append(emb)
            labels.append(identity)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    print("Embeddings shape:", embeddings.shape)
    print("Number of identities:", len(np.unique(labels)))

    # Run evaluation bundle
    model = DummyModel()
    device = "cpu"

    bundle = ReIDEvalBundle(
        model=model,
        embeddings=embeddings,
        labels=labels,
        device=device
    )

    results = bundle.compute_all()

    # Optional advanced metrics
    ndcg_score = bundle.ndcg(k=10)
    recall5 = bundle.recall_at_k(k=5)
    dist_stats = bundle.intra_inter_distance()

    print("\n=== ReID Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print(f"nDCG@10: {ndcg_score:.4f}")
    print(f"Recall@5: {recall5:.4f}")
    print("\nDistance stats:", dist_stats)