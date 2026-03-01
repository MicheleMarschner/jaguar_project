import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from jaguar.config import PATHS, DEVICE, DATA_ROOT
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.datasets.JaguarDataset import JaguarDataset


def generate_submission():

    # --------------------------------------------------
    # 1. CONFIG
    # --------------------------------------------------
    CHECKPOINT_PATH = "/home/vanessa/Documents/repos/jaguar_project/miewid/jaguar_reid_v1_epoch_20.pth"
    TEST_CSV_PATH = "/home/vanessa/Documents/repos/jaguar_project/data/round_1/raw/jaguar-re-id/test.csv"

    BACKBONE_NAME = "MiewID"
    EMB_DIM = 512
    NUM_CLASSES = 31
    BATCH_SIZE = 32

    print(f"Loading model from {CHECKPOINT_PATH}...")

    # --------------------------------------------------
    # 2. LOAD MODEL
    # --------------------------------------------------
    model = JaguarIDModel(
        backbone_name=BACKBONE_NAME,
        num_classes=NUM_CLASSES,
        head_type="arcface",
        device=DEVICE,
        emb_dim=EMB_DIM
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------
    # 3. LOAD TEST CSV
    # --------------------------------------------------
    test_df = pd.read_csv(TEST_CSV_PATH)

    unique_filenames = sorted(
        list(set(test_df['query_image']) | set(test_df['gallery_image']))
    )

    print(f"Found {len(unique_filenames)} unique test images.")

    # --------------------------------------------------
    # 4. BUILD DATASET USING EXACT SAME FILENAMES
    # --------------------------------------------------

    test_ds = JaguarDataset(
        base_root=PATHS.data_export / "init",
        data_root=DATA_ROOT,
        mode="test",
        is_test=True,
        transform=model.backbone_wrapper.transform
    )

    # 🔥 Force dataset to use EXACT same ordered filenames
    test_ds.samples = [
        {"filepath": fname, "ground_truth": {"label": ""}}
        for fname in unique_filenames
    ]

    test_ds.labels = [""] * len(test_ds.samples)

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # --------------------------------------------------
    # 5. SANITY CHECK ORDERING
    # --------------------------------------------------

    dataset_filenames = [s["filepath"] for s in test_loader.dataset.samples]

    print("\nSANITY CHECK")
    print("First 10 dataset filenames:")
    print(dataset_filenames[:10])

    print("\nFirst 10 CSV-derived filenames:")
    print(unique_filenames[:10])

    assert dataset_filenames == unique_filenames, \
        "❌ Dataset order does NOT match unique_filenames order!"

    print("✅ Filename alignment confirmed.")

    # --------------------------------------------------
    # 6. EXTRACT EMBEDDINGS
    # --------------------------------------------------

    embeddings = []

    print("\nExtracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            imgs = batch["img"].to(DEVICE)

            features = model.get_embeddings(imgs)

            # Ensure normalization
            features = torch.nn.functional.normalize(features, dim=1)

            embeddings.append(features.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    print(f"Embedding shape: {embeddings.shape}")

    # --------------------------------------------------
    # 7. COMPUTE SIMILARITY MATRIX
    # --------------------------------------------------

    print("\nComputing cosine similarity...")
    sim_matrix = cosine_similarity(embeddings)

    # Proper rescaling to [0, 1]
    sim_matrix = np.clip(sim_matrix,0,1)

    print("\nSimilarity statistics:")
    print(f"  Min: {sim_matrix.min():.4f}")
    print(f"  Max: {sim_matrix.max():.4f}")
    print(f"  Mean: {sim_matrix.mean():.4f}")
    print(f"  Std: {sim_matrix.std():.4f}")

    # --------------------------------------------------
    # 8. BUILD MAPPING FROM DATASET ORDER
    # --------------------------------------------------

    filename_to_idx = {
        fname: i for i, fname in enumerate(dataset_filenames)
    }

    # --------------------------------------------------
    # 9. GENERATE SUBMISSION
    # --------------------------------------------------

    print("\nGenerating submission...")

    results = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        q_idx = filename_to_idx[row['query_image']]
        g_idx = filename_to_idx[row['gallery_image']]

        score = sim_matrix[q_idx, g_idx]

        results.append({
            "row_id": row["row_id"],
            "similarity": score
        })

    submission_df = pd.DataFrame(results)

    # --------------------------------------------------
    # 10. FINAL VALIDATION
    # --------------------------------------------------

    assert len(submission_df) == len(test_df)
    assert submission_df['similarity'].min() >= 0
    assert submission_df['similarity'].max() <= 1

    print("Submission stats:")
    print(submission_df["similarity"].describe())

    save_path = "submission.csv"
    submission_df.to_csv(save_path, index=False)

    print(f"\n✅ Successfully saved submission to {save_path}")


if __name__ == "__main__":
    generate_submission()