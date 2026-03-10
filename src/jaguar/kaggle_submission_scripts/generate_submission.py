import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from jaguar.config import DEVICE, DATA_ROOT, PATHS
from jaguar.datasets.JaguarDataset import JaguarDataset
from jaguar.models.jaguarid_models import JaguarIDModel


def k_reciprocal_rerank(sim_matrix, k1=20, lambda_value=0.3):
    """Optional lightweight reranking on similarity matrix."""
    print("Applying re-ranking...")

    dist = 1.0 - sim_matrix
    initial_rank = np.argsort(dist, axis=1)

    nn_k1 = []
    for i in range(sim_matrix.shape[0]):
        forward_k1 = initial_rank[i, : k1 + 1]
        backward_k1 = initial_rank[forward_k1, : k1 + 1]
        fi = np.where(backward_k1 == i)[0]
        nn_k1.append(forward_k1[fi])

    jaccard_dist = np.zeros_like(dist)

    for i in range(sim_matrix.shape[0]):
        ind_non_zero = np.where(dist[i, :] < 0.6)[0]
        ind_images = [
            j for j in ind_non_zero if len(np.intersect1d(nn_k1[i], nn_k1[j])) > 0
        ]

        for j in ind_images:
            intersection = len(np.intersect1d(nn_k1[i], nn_k1[j]))
            union = len(np.union1d(nn_k1[i], nn_k1[j]))
            if union > 0:
                jaccard_dist[i, j] = 1.0 - intersection / union

    reranked = 1.0 - (jaccard_dist * lambda_value + dist * (1.0 - lambda_value))
    return np.clip(reranked, 0.0, 1.0)


def query_expansion(embeddings, top_k=3):
    """Optional query expansion on normalized embeddings."""
    print("Applying query expansion...")
    sims = embeddings @ embeddings.T
    indices = np.argsort(-sims, axis=1)[:, :top_k]

    expanded = np.zeros_like(embeddings)
    for i in range(len(embeddings)):
        expanded[i] = np.mean(embeddings[indices[i]], axis=0)

    norms = np.linalg.norm(expanded, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return expanded / norms


def build_test_dataset(unique_filenames, transform):
    """
    Build a test dataset whose sample order exactly matches unique_filenames.
    """
    test_ds = JaguarDataset(
        base_root=".",          # not used for test loading in your current dataset
        data_root=DATA_ROOT,
        mode="test",
        is_test=True,
        transform=transform,
        include_duplicates=True,
    )

    test_ds.samples = [
        {
            "filepath": fname,
            "filename": fname,
            "ground_truth": {"label": ""},
        }
        for fname in unique_filenames
    ]
    test_ds.labels = [""] * len(test_ds.samples)

    return test_ds


def load_model(checkpoint_path, backbone_name, emb_dim, num_classes, head_type="arcface"):
    print(f"Loading model from {checkpoint_path}...")

    model = JaguarIDModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        head_type=head_type,
        device=DEVICE,
        emb_dim=emb_dim,
    )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()
    return model


def extract_embeddings(model, dataloader, use_tta=True):
    print("Extracting embeddings...")
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs = batch["img"].to(DEVICE)

            feats = model.get_embeddings(imgs)

            if use_tta:
                flipped = torch.flip(imgs, dims=[3])
                feats_flip = model.get_embeddings(flipped)
                feats = (feats + feats_flip) / 2.0

            feats = torch.nn.functional.normalize(feats, dim=1)
            all_embeddings.append(feats.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings


def validate_submission(submission_df, test_df):
    assert list(submission_df.columns) == ["row_id", "similarity"]
    assert len(submission_df) == len(test_df)
    assert submission_df["row_id"].tolist() == test_df["row_id"].tolist()
    assert np.isfinite(submission_df["similarity"].values).all()
    assert (submission_df["similarity"] >= 0).all()
    assert (submission_df["similarity"] <= 1).all()


def generate_submission():
    # --------------------------------------------------
    # CONFIG
    # --------------------------------------------------
    CHECKPOINT_PATH = PATHS.checkpoints / "kaggle_deduplication/closed_curated_traink_3_valk_3_p4/best_model.pth" 
    TEST_CSV_PATH = PATHS.data / "jaguar-re-id/test.csv"
    SAVE_PATH = PATHS.results / "submissions" / "submission.csv"

    BACKBONE_NAME = "EVA-02"   # adapt to your checkpoint
    EMB_DIM = 1024                       # adapt to your checkpoint
    NUM_CLASSES = 31                     # train identities
    HEAD_TYPE = "arcface"
    BATCH_SIZE = 32
    NUM_WORKERS = 1

    USE_TTA = True
    USE_QE = False
    USE_RERANK = False

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------
    model = load_model(
        checkpoint_path=CHECKPOINT_PATH,
        backbone_name=BACKBONE_NAME,
        emb_dim=EMB_DIM,
        num_classes=NUM_CLASSES,
        head_type=HEAD_TYPE,
    )

    # --------------------------------------------------
    # LOAD TEST CSV
    # --------------------------------------------------
    test_df = pd.read_csv(TEST_CSV_PATH)

    unique_filenames = sorted(
        set(test_df["query_image"]) | set(test_df["gallery_image"])
    )
    print(f"Found {len(unique_filenames)} unique test images.")

    # --------------------------------------------------
    # BUILD TEST DATASET
    # --------------------------------------------------
    test_ds = build_test_dataset(
        unique_filenames=unique_filenames,
        transform=model.backbone_wrapper.transform,
    )

    dataset_filenames = [s["filepath"] for s in test_ds.samples]
    assert dataset_filenames == unique_filenames, "Dataset order does not match unique_filenames"

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # --------------------------------------------------
    # EXTRACT EMBEDDINGS
    # --------------------------------------------------
    embeddings = extract_embeddings(
        model=model,
        dataloader=test_loader,
        use_tta=USE_TTA,
    )

    # --------------------------------------------------
    # SIMILARITY MATRIX
    # --------------------------------------------------
    if USE_QE:
        embeddings = query_expansion(embeddings, top_k=3)

    sim_matrix = embeddings @ embeddings.T

    if USE_RERANK:
        sim_matrix = k_reciprocal_rerank(sim_matrix)

    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)

    print("Similarity stats:")
    print(f"  min  = {sim_matrix.min():.6f}")
    print(f"  max  = {sim_matrix.max():.6f}")
    print(f"  mean = {sim_matrix.mean():.6f}")
    print(f"  std  = {sim_matrix.std():.6f}")

    # --------------------------------------------------
    # MAP FILENAMES TO INDICES
    # --------------------------------------------------
    filename_to_idx = {fname: i for i, fname in enumerate(unique_filenames)}

    # --------------------------------------------------
    # BUILD SUBMISSION
    # --------------------------------------------------
    similarities = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating submission"):
        q_idx = filename_to_idx[row["query_image"]]
        g_idx = filename_to_idx[row["gallery_image"]]
        similarities.append(sim_matrix[q_idx, g_idx])

    submission_df = pd.DataFrame(
        {
            "row_id": test_df["row_id"],
            "similarity": similarities,
        }
    )

    validate_submission(submission_df, test_df)

    print(submission_df["similarity"].describe())
    submission_df.to_csv(SAVE_PATH, index=False)
    print(f"Saved submission to {SAVE_PATH}")


if __name__ == "__main__":
    generate_submission()
