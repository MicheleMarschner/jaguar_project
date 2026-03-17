## map
## archloss

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
from sklearn.metrics import silhouette_score


def compute_pairwise_ap(labels, embeddings):
    """
    Compute pairwise average precision by treating each image pair as a same-identity or different-identity decision.
    """

    N = len(labels)

    sims = embeddings @ embeddings.T
    sims = sims / (
        np.linalg.norm(embeddings, axis=1, keepdims=True)
        * np.linalg.norm(embeddings, axis=1, keepdims=True).T
    )

    y_true = []
    y_scores = []

    for i in range(N):
        for j in range(i + 1, N):
            y_true.append(int(labels[i] == labels[j]))
            y_scores.append(sims[i, j])

    return average_precision_score(y_true, y_scores)

def compute_ib_map_from_embeddings(labels, embeddings):
    """
    Compute identity-balanced mean average precision directly from labels and embedding vectors.
    """
    N = len(labels)
    identity_map = {}

    # Compute cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T
    sim_matrix = sim_matrix / (np.linalg.norm(embeddings, axis=1, keepdims=True) *
                               np.linalg.norm(embeddings, axis=1, keepdims=True).T)
    np.fill_diagonal(sim_matrix, -1.0)  # exclude self-match
    sim_matrix = np.clip(sim_matrix, 0, 1)

    for i in range(N):
        q_label = labels[i]

        # Rank all others
        scores = sim_matrix[i]
        idx_rank = np.argsort(-scores)
        ranked_labels = [labels[j] for j in idx_rank]

        # Binary relevance
        rels = np.array([1 if l == q_label else 0 for l in ranked_labels])
        num_rel = rels.sum()
        if num_rel == 0:
            continue
        precision_at_k = np.cumsum(rels) / (np.arange(len(rels)) + 1)
        ap = (precision_at_k * rels).sum() / num_rel

        if q_label not in identity_map:
            identity_map[q_label] = []
        identity_map[q_label].append(ap)

    # Compute mean AP per identity
    identity_map = {k: np.mean(v) for k, v in identity_map.items()}
    ib_map = np.mean(list(identity_map.values()))
    return ib_map, identity_map

class ReIDEvalBundle:
    """
    Cached evaluation bundle for ReID embeddings, retrieval metrics, and similarity diagnostics.
    """
    def __init__(
        self,
        model=None,
        embeddings=None,
        labels=None,
        images=None,
        device="cuda",
        normalize=True,
    ):
        """
        Initialize the bundle from embeddings or images and optionally apply a model before evaluation.
        """
        self.model = model
        self.device = device
        self.normalize = normalize

        # Optional inputs
        self.images = images  # raw tensors (N, C, H, W)
        self.labels = np.asarray(labels) if labels is not None else None

        # Store initial embeddings (if provided)
        if embeddings is not None:
            if isinstance(embeddings, torch.Tensor):
                self.initial_embeddings = embeddings.detach().cpu().numpy()
            else:
                self.initial_embeddings = np.asarray(embeddings)
        else:
            self.initial_embeddings = None

        # Internal cache
        self._finetuned_embeddings = None
        self._sim_matrix = None
        self._ranked_indices = None
        
    # -------------------------------------------------------
    # Silhouette score for clustering quality of embeddings
    # -------------------------------------------------------
    def compute_silhouette(self):
        """Compute the silhouette score of the evaluated embeddings using cosine distance."""
        emb_np = self.finetuned_embeddings()
        lbl_np = self.labels
        # Note: If validation set grows, consider sampling 1000 points here
        return float(silhouette_score(emb_np, lbl_np, metric='cosine'))

    # -------------------------------------------------
    # Core cached computations
    # -------------------------------------------------
    def finetuned_embeddings(self):
        """
        Return the final embeddings used for evaluation, reusing cached results when available.
        """
        if self._finetuned_embeddings is not None:
            return self._finetuned_embeddings

        # Case 1: Precomputed embeddings, no model
        if self.initial_embeddings is not None and self.model is None:
            emb = self.initial_embeddings

        # Case 2: Precomputed embeddings + fine-tuned head
        elif self.initial_embeddings is not None and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                x = torch.from_numpy(self.initial_embeddings).float().to(self.device)
                # Supports both: model(x) or model.get_embeddings(x)
                if hasattr(self.model, "get_embeddings"):
                    emb = self.model.get_embeddings(x)
                else:
                    emb = self.model(x)
                emb = emb.detach().cpu().numpy()

        # Case 3: Images + model (full extraction)
        elif self.images is not None and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                x = self.images.to(self.device)
                if hasattr(self.model, "get_embeddings"):
                    emb = self.model.get_embeddings(x)
                else:
                    emb = self.model(x)
                emb = emb.detach().cpu().numpy()
        else:
            raise ValueError(
                "You must provide either:\n"
                "- embeddings\n"
                "- OR (images + model)"
            )
        # Optional normalization
        if self.normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms
        self._finetuned_embeddings = emb
        return self._finetuned_embeddings

    def similarity_matrix(self):
        """Return the cached cosine similarity matrix for all evaluated samples."""
        if self._sim_matrix is None:
            emb = self.finetuned_embeddings()
            sim = cosine_similarity(emb)
            np.fill_diagonal(sim, -1.0)  # exclude self-match
            self._sim_matrix = sim
        return self._sim_matrix

    def ranked_indices(self):
        """Return per-query retrieval rankings sorted by descending similarity."""
        if self._ranked_indices is None:
            sim = self.similarity_matrix()
            self._ranked_indices = np.argsort(-sim, axis=1)
        return self._ranked_indices

    # -------------------------------------------------
    # Core ReID Metrics
    # -------------------------------------------------
    def mAP(self):
        """Compute identity-balanced mean average precision from the evaluated embeddings."""
        self.finetuned_embeddings()
        ib_map, _ = compute_ib_map_from_embeddings(self.labels, self._finetuned_embeddings)
        return ib_map  # now mAP and identity-balanced mAP are the same

    def identity_balanced_map(self):
        """Return the identity-balanced mean average precision score."""
        return self.mAP()
    
    def pairwise_ap(self):
        """Compute pairwise average precision over all image pairs."""
        emb = self.finetuned_embeddings()
        return compute_pairwise_ap(self.labels, emb)

    # -------------------------------------------------
    # Retrieval Metrics 
    # -------------------------------------------------
    def top_k_accuracy(self, k=1):
        """
        Compute rank-k retrieval accuracy based on whether a correct identity appears in the top-k results.
        """
        ranked = self.ranked_indices()
        labels = self.labels

        correct = 0
        total = len(labels)

        for i in range(total):
            query_label = labels[i]
            # Exclude self index if it happens to be in top_k
            top_k = ranked[i][ranked[i] != i][:k]  # remove query itself
            if query_label in labels[top_k]:
                correct += 1

        return correct / total

    def rank1(self):
        """Return rank-1 retrieval accuracy."""
        return self.top_k_accuracy(k=1)

    def rank5(self):
        """Return rank-5 retrieval accuracy."""
        return self.top_k_accuracy(k=5)
    
    def ndcg(self, k=None):
        """
        Compute normalized discounted cumulative gain for the ranked retrieval results.
        """
        ranked = self.ranked_indices()
        labels = self.labels
        ndcgs = []

        for i in range(len(labels)):
            query_label = labels[i]
            order = ranked[i][ranked[i] != i]  # exclude self

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
        """Return recall at k, equivalent here to top-k retrieval accuracy."""
        # Recall@k is equivalent to top-k accuracy
        return self.top_k_accuracy(k)
    
    def intra_inter_distance(self):
        """
        Compute average intra-identity and inter-identity L2 distances in embedding space.
        """
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
        """Compute the mean cosine similarity between same-identity samples."""
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
        """Compute the mean cosine similarity between different-identity samples."""
        sim = self.similarity_matrix()
        labels = self.labels

        neg_sims = []
        for i in range(len(labels)):
            diff_id = (labels != labels[i])
            neg_sims.extend(sim[i][diff_id])

        return float(np.mean(neg_sims))

    def similarity_gap(self):
        """
        Compute the separation gap between positive and negative cosine similarities.
        """
        return self.mean_positive_similarity() - self.mean_negative_similarity()


    # -------------------------------------------------
    # Convenience: all metrics at once
    # -------------------------------------------------
    def compute_all(self, k_ndcg=10, k_recall=5, include_silhouette=False):
        """
        Compute the full set of retrieval, ranking, and embedding diagnostics in one cached pass.
        """

        # Core ranking metrics
        id_map = self.identity_balanced_map()
        map_score = self.mAP()
        rank1 = self.rank1()
        rank5 = self.rank5()
        pair_ap = self.pairwise_ap()

        # Retrieval / ranking quality
        ndcg_k = self.ndcg(k=k_ndcg)
        recall_k = self.recall_at_k(k=k_recall)

        # Embedding diagnostics (similarity space)
        pos_sim = self.mean_positive_similarity()
        neg_sim = self.mean_negative_similarity()
        sim_gap = self.similarity_gap()

        # Distance diagnostics (geometry of embedding space)
        dist_stats = self.intra_inter_distance()
        
        results = {
            # --- Competition / primary ---
            "id_balanced_mAP": id_map,
            "mAP": map_score,
            "pairwise_AP": pair_ap,

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
        
        if include_silhouette:
            # Silouhette score for embeddings 
            results["silhouette"] = self.compute_silhouette()
            
        return results 