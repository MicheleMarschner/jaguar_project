# tests/test_metrics.py

import pytest
import numpy as np
import torch
from jaguar.evaluation.metrics import ReIDEvalBundle

# Dummy model mimicking your MegaDescriptor head
class DummyModel:
    def eval(self):
        pass

    def get_embeddings(self, x):
        # Simulate a learned projection (normalized)
        return torch.nn.functional.normalize(x, dim=1)


@pytest.fixture
def small_reid_data():
    np.random.seed(42)
    torch.manual_seed(42)

    n_identities = 5
    images_per_id = 4
    dim = 16

    labels = []
    embeddings = []

    for identity in range(n_identities):
        center = np.random.randn(dim)
        center = center / np.linalg.norm(center)
        for _ in range(images_per_id):
            emb = center + 0.05 * np.random.randn(dim)
            embeddings.append(emb)
            labels.append(identity)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    return embeddings, labels


@pytest.fixture
def bundle(small_reid_data):
    embeddings, labels = small_reid_data
    model = DummyModel()
    device = "cpu"
    return ReIDEvalBundle(model=model, embeddings=embeddings, labels=labels, device=device)


def test_map(bundle):
    map_score = bundle.map()
    assert 0 <= map_score <= 1, "mAP should be between 0 and 1"


def test_identity_balanced_map(bundle):
    id_map = bundle.identity_balanced_map()
    assert 0 <= id_map <= 1, "Identity-balanced mAP should be between 0 and 1"


def test_rank1_rank5(bundle):
    r1 = bundle.rank1()
    r5 = bundle.rank5()
    assert 0 <= r1 <= 1, "Rank1 should be between 0 and 1"
    assert 0 <= r5 <= 1, "Rank5 should be between 0 and 1"
    assert r5 >= r1, "Rank5 should be >= Rank1"


def test_ndcg_recall(bundle):
    ndcg_score = bundle.ndcg(k=3)
    recall_score = bundle.recall_at_k(k=3)
    assert 0 <= ndcg_score <= 1, "nDCG should be between 0 and 1"
    assert 0 <= recall_score <= 1, "Recall@k should be between 0 and 1"


def test_intra_inter_distance(bundle):
    dist_stats = bundle.intra_inter_distance()
    assert dist_stats["intra_dist"] >= 0
    assert dist_stats["inter_dist"] >= 0
    assert dist_stats["dist_gap"] == pytest.approx(
        dist_stats["inter_dist"] - dist_stats["intra_dist"], rel=1e-6
    )


def test_similarity_gap(bundle):
    gap = bundle.similarity_gap()
    pos = bundle.mean_positive_similarity()
    neg = bundle.mean_negative_similarity()
    assert gap == pytest.approx(pos - neg, rel=1e-6)