import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import toml
import torch.nn.functional as F

from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.datasets.JaguarDataset import JaguarDataset
from jaguar.config import PATHS, DEVICE, DATA_ROOT

SUPPORTED_MULTISCALE_BACKBONES = [
    "convnext_v2",
    "efficientnet_b4",
]

def load_or_compute_embeddings(exp_dir, model, loader, tta_mode):
    exp_dir = Path(exp_dir)
    is_round2 = "round_2" in str(exp_dir).lower()

    if is_round2:
        emb_file = exp_dir / "embeddings_round_2_test.npy"
    else:
        emb_file = exp_dir / "embeddings_test.npy"

    # if emb_file.exists():
    #     print(f"Loading cached embeddings: {emb_file}")
    #     return np.load(emb_file)

    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, loader, tta_mode)

    print(f"Saving embeddings to {emb_file}")
    np.save(emb_file, embeddings)
    return embeddings

def k_reciprocal_rerank(prob, k1=20, k2=6, lambda_value=0.3):
    """
    Apply k-reciprocal re-ranking to refine similarity scores
    using neighborhood consistency (Jaccard distance).
    Pytorch implementation of https://github.com/zhunzhong07/person-re-ranking/blob/master/evaluation/utils/re_ranking.m 

    Parameters
        prob : np.ndarray; NxN similarity matrix (cosine similarity)
        k1 : int; size of the k-reciprocal neighborhood
        k2 : int; number of neighbors for local query expansion
        lambda_value : float; weighting between original distance and Jaccard distance

    Returns
        final_dist : np.ndarray; NxN re-ranked distance matrix
    """

    print("Applying Re-ranking...")

    # Step 0: convert similarity to distance
    original_dist = 1 - prob
    query_num = prob.shape[0]

    # Step 1: normalize distances per row
    original_dist = original_dist / np.max(original_dist, axis=1, keepdims=True)
    
    # Step 2: initial ranking
    initial_rank = np.argsort(original_dist, axis=1)
    V = np.zeros_like(original_dist, dtype=np.float32)

    # Step 3: compute k-reciprocal neighbors
    for i in range(query_num):
        forward_k = initial_rank[i, :k1 + 1]
        backward_k = initial_rank[forward_k, :k1 + 1]
        fi = np.where(backward_k == i)[1]  # reciprocal neighbors
        k_reciprocal_index = forward_k[fi]

        # Iterative k-reciprocal expansion
        k_reciprocal_exp = list(k_reciprocal_index)
        for candidate in k_reciprocal_index:
            candidate_forward = initial_rank[candidate, :int((k1 + 1) / 2)]
            candidate_backward = initial_rank[candidate_forward, :int((k1 + 1) / 2)]
            fi_candidate = np.where(candidate_backward == candidate)[1]
            candidate_k_reciprocal = candidate_forward[fi_candidate]

            if len(np.intersect1d(k_reciprocal_index, candidate_k_reciprocal)) > 2/3 * len(candidate_k_reciprocal):
                k_reciprocal_exp.extend(candidate_k_reciprocal)

        k_reciprocal_exp = np.unique(k_reciprocal_exp)
        # compute weights for this sample
        weight = np.exp(-original_dist[i, k_reciprocal_exp])
        V[i, k_reciprocal_exp] = weight / np.sum(weight)

    # Step 4: local query expansion (k2 > 1)
    if k2 > 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(query_num):
            neighbors = initial_rank[i, :k2]
            V_qe[i, :] = np.mean(V[neighbors, :], axis=0)
        V = V_qe

    # Step 5: compute Jaccard distance via inverted index
    gallery_num = prob.shape[0]
    invIndex = [np.where(V[:, i] != 0)[0] for i in range(gallery_num)]
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(gallery_num, dtype=np.float32)
        ind_nonzero = np.where(V[i, :] != 0)[0]
        for j in ind_nonzero:
            temp_min[invIndex[j]] += np.minimum(V[i, j], V[invIndex[j], j])
        jaccard_dist[i, :] = 1 - temp_min / (2 - temp_min)

    # Step 6: combine original distance and Jaccard distance
    return jaccard_dist * (1 - lambda_value) + original_dist * lambda_value


def query_expansion(emb, top_k=3):
    """
    Apply query expansion by averaging the embeddings of the
    top-k nearest neighbors.
    """

    print("Applying Query Expansion...")

    sims = emb @ emb.T
    indices = np.argsort(-sims, axis=1)[:, :top_k]

    new_emb = np.zeros_like(emb)

    for i in range(len(emb)):
        new_emb[i] = np.mean(emb[indices[i]], axis=0)

    return new_emb / np.linalg.norm(new_emb, axis=1, keepdims=True)


def extract_embeddings(model, loader, tta_mode):
    """
    Extract normalized embeddings with optional Test-Time Augmentation.

    Modes
    -----
    none        : original image only
    flip        : original + horizontal flip
    multiscale  : scales [0.75, 1.0, 1.25] + flipped versions
    """

    embeddings = []

    print("\nExtracting embeddings...")

    # scales used only for multiscale TTA
    scales = [1.0]
    if tta_mode == "multiscale":
        scales = [0.75, 1.0, 1.25]

    with torch.no_grad():
        for batch in tqdm(loader):

            imgs = batch["img"].to(DEVICE)
            feats_all = []

            for s in scales:

                # resize image if scale != 1
                if s != 1.0:
                    scaled_imgs = F.interpolate(
                        imgs,
                        scale_factor=s,
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    scaled_imgs = imgs

                # forward pass
                feats = model.get_embeddings(scaled_imgs)
                feats_all.append(feats)

                # optional horizontal flip
                if tta_mode in ["flip", "multiscale"]:
                    flipped = torch.flip(scaled_imgs, dims=[3])
                    feats_flip = model.get_embeddings(flipped)
                    feats_all.append(feats_flip)

            # average embeddings across augmentations
            feats = torch.stack(feats_all).mean(dim=0)

            feats = torch.nn.functional.normalize(feats, dim=1)
            embeddings.append(feats.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    print(f"Embedding shape: {embeddings.shape}")

    return embeddings

def build_dataset(test_csv, transform):
    """
    Construct the evaluation dataset and enforce consistent
    ordering with the submission CSV.
    """

    test_df = pd.read_csv(test_csv)

    unique_filenames = sorted(
        list(set(test_df["query_image"]) | set(test_df["gallery_image"]))
    )

    print(f"Found {len(unique_filenames)} unique test images.")

    dataset = JaguarDataset(
        base_root=PATHS.data_export / "init",
        data_root=DATA_ROOT,
        mode="test",
        is_test=True,
        transform=transform,
        include_duplicates=True,
    )

    dataset.samples = [
        {"filepath": fname, "ground_truth": {"label": ""}}
        for fname in unique_filenames
    ]

    dataset.labels = [""] * len(dataset.samples)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    dataset_filenames = [s["filepath"] for s in loader.dataset.samples]

    print("\nSANITY CHECK")
    print("First 10 dataset filenames:")
    print(dataset_filenames[:10])

    print("\nFirst 10 CSV-derived filenames:")
    print(unique_filenames[:10])

    assert dataset_filenames == unique_filenames, \
        "Dataset order does not match CSV filenames."

    print("Filename alignment confirmed.")

    return test_df, loader, unique_filenames


def compute_similarity(embeddings, use_qe, use_rerank):
    """
    Compute similarity matrix with optional QE and re-ranking.
    """

    print("\nComputing similarity matrix...")

    sim_matrix = embeddings @ embeddings.T

    if use_qe:
        qe_emb = query_expansion(embeddings)
        sim_matrix = qe_emb @ qe_emb.T

    if use_rerank:
        sim_matrix = k_reciprocal_rerank(sim_matrix)

    sim_matrix = np.clip(sim_matrix, 0, 1)

    print("\nSimilarity statistics:")
    print(f"Min: {sim_matrix.min():.4f}")
    print(f"Max: {sim_matrix.max():.4f}")
    print(f"Mean: {sim_matrix.mean():.4f}")
    print(f"Std: {sim_matrix.std():.4f}")

    return sim_matrix


def generate_submission(sim_matrix, test_df, filenames, output_path):
    """
    Convert similarity matrix into the submission format expected
    by the evaluation server.
    """

    filename_to_idx = {f: i for i, f in enumerate(filenames)}

    results = []

    print("\nGenerating submission...")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

        q_idx = filename_to_idx[row["query_image"]]
        g_idx = filename_to_idx[row["gallery_image"]]

        score = sim_matrix[q_idx, g_idx]

        results.append(
            {
                "row_id": row["row_id"],
                "similarity": score,
            }
        )

    submission = pd.DataFrame(results)

    assert len(submission) == len(test_df)
    assert submission["similarity"].min() >= 0
    assert submission["similarity"].max() <= 1

    print("\nSubmission statistics:")
    print(submission["similarity"].describe())

    save_path = output_path / "submission.csv"
    submission.to_csv(save_path, index=False)

    print(f"\nSubmission saved to {save_path}")


def load_model_from_toml(toml_path, checkpoint_path):
    """
    Load a trained model using parameters stored in a TOML config.
    """

    cfg = toml.load(toml_path)

    print(f"Loading model from {checkpoint_path}")

    model = JaguarIDModel(
        backbone_name=cfg["model"]["backbone_name"],
        num_classes=31,
        head_type=cfg["model"]["head_type"],
        device=DEVICE,
        emb_dim=cfg["model"]["emb_dim"],
        use_projection=cfg["model"]["use_projection"],
        use_forward_features=cfg["model"]["use_forward_features"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model