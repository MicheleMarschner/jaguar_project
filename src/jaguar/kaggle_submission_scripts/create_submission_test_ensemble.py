import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from jaguar.config import PATHS, DEVICE, DATA_ROOT
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.datasets.JaguarDataset import JaguarDataset


# =========================================================
# Helpers
# =========================================================

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def global_minmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mn = x.min()
    mx = x.max()
    if mx - mn < eps:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def query_expansion(emb: np.ndarray, top_k: int = 3) -> np.ndarray:
    sims = emb @ emb.T
    indices = np.argsort(-sims, axis=1)[:, :top_k]

    out = np.zeros_like(emb)
    for i in range(len(emb)):
        out[i] = emb[indices[i]].mean(axis=0)

    return l2_normalize(out)


def k_reciprocal_rerank(prob: np.ndarray, k1: int = 20, k2: int = 6, lambda_value: float = 0.3) -> np.ndarray:
    print("Applying Re-ranking...")

    q_g_dist = 1 - prob
    original_dist = q_g_dist.copy()
    initial_rank = np.argsort(original_dist, axis=1)

    nn_k1 = []
    for i in range(prob.shape[0]):
        forward_k1 = initial_rank[i, :k1 + 1]
        backward_k1 = initial_rank[forward_k1, :k1 + 1]
        fi = np.where(backward_k1 == i)[0]
        nn_k1.append(forward_k1[fi])

    jaccard_dist = np.zeros_like(original_dist)

    for i in range(prob.shape[0]):
        ind_non_zero = np.where(original_dist[i, :] < 0.6)[0]
        ind_images = [
            inv for inv in ind_non_zero
            if len(np.intersect1d(nn_k1[i], nn_k1[inv])) > 0
        ]

        for j in ind_images:
            intersection = len(np.intersect1d(nn_k1[i], nn_k1[j]))
            union = len(np.union1d(nn_k1[i], nn_k1[j]))
            jaccard_dist[i, j] = 1 - intersection / union

    return 1 - (jaccard_dist * lambda_value + original_dist * (1 - lambda_value))


# =========================================================
# Data / model loading
# =========================================================

def build_test_loader(test_csv_path: Path, transform, batch_size: int = 32):
    test_df = pd.read_csv(test_csv_path)

    unique_filenames = sorted(
        list(set(test_df["query_image"]) | set(test_df["gallery_image"]))
    )

    test_ds = JaguarDataset(
        base_root=PATHS.data_export / "init",
        data_root=DATA_ROOT,
        mode="test",
        is_test=True,
        transform=transform,
        include_duplicates=True,
    )

    test_ds.samples = [
        {"filepath": fname, "ground_truth": {"label": ""}}
        for fname in unique_filenames
    ]
    test_ds.labels = [""] * len(test_ds.samples)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    dataset_filenames = [s["filepath"] for s in test_loader.dataset.samples]
    assert dataset_filenames == unique_filenames, "Dataset order mismatch."

    return test_df, test_loader, dataset_filenames


def load_model(cfg: dict):
    model = JaguarIDModel(
        backbone_name=cfg["backbone_name"],
        num_classes=cfg.get("num_classes", 31),
        head_type=cfg["head_type"],
        device=DEVICE,
        emb_dim=cfg["emb_dim"],
        use_projection=cfg.get("use_projection", True),
        use_forward_features=cfg.get("use_forward_features", False),
    )

    checkpoint = torch.load(cfg["checkpoint_path"], map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def extract_embeddings(model, test_loader, use_tta: bool = False) -> np.ndarray:
    all_emb = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extract embeddings"):
            imgs = batch["img"].to(DEVICE)

            emb = model.get_embeddings(imgs)

            if use_tta:
                flipped = torch.flip(imgs, dims=[3])
                emb_flip = model.get_embeddings(flipped)
                emb = (emb + emb_flip) / 2.0

            emb = torch.nn.functional.normalize(emb, dim=1)
            all_emb.append(emb.cpu().numpy())

    return np.concatenate(all_emb, axis=0)


# =========================================================
# Ensemble methods
# =========================================================

def compute_similarity_from_embeddings(
    emb: np.ndarray,
    use_qe: bool = False,
    use_rerank: bool = False,
) -> np.ndarray:
    if use_qe:
        emb = query_expansion(emb)

    sim = emb @ emb.T
    sim = np.clip(sim, 0, 1)

    if use_rerank:
        sim = k_reciprocal_rerank(sim)

    return np.clip(sim, 0, 1)


def build_embedding_concat_ensemble(embeddings_list: list[np.ndarray]) -> np.ndarray:
    fused = np.concatenate(embeddings_list, axis=1)
    return l2_normalize(fused)


def build_score_fusion_ensemble(
    sim_mats: list[np.ndarray],
    weights: list[float],
    use_globalminmax: bool = True,
    square_scores: bool = True,
) -> np.ndarray:
    assert len(sim_mats) == len(weights), "sim_mats and weights must have same length."

    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()

    fused = np.zeros_like(sim_mats[0], dtype=np.float32)

    for sim, w in zip(sim_mats, weights):
        s = sim.copy()
        if use_globalminmax:
            s = global_minmax(s)
        if square_scores:
            s = s ** 2
        fused += w * s

    return np.clip(fused, 0, 1)


# =========================================================
# Saving
# =========================================================

def save_embeddings(save_dir: Path, embeddings: np.ndarray, filenames: list[str], stem: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"{stem}_embeddings.npy", embeddings)
    pd.DataFrame({"filepath": filenames}).to_csv(save_dir / f"{stem}_filenames.csv", index=False)


def save_similarity(save_dir: Path, sim_matrix: np.ndarray, filenames: list[str], stem: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"{stem}_similarity.npy", sim_matrix)
    pd.DataFrame({"filepath": filenames}).to_csv(save_dir / f"{stem}_filenames.csv", index=False)


def build_submission(
    test_df: pd.DataFrame,
    dataset_filenames: list[str],
    sim_matrix: np.ndarray,
    save_path: Path,
):
    filename_to_idx = {fname: i for i, fname in enumerate(dataset_filenames)}

    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Build submission"):
        q_idx = filename_to_idx[row["query_image"]]
        g_idx = filename_to_idx[row["gallery_image"]]

        rows.append({
            "row_id": row["row_id"],
            "similarity": float(sim_matrix[q_idx, g_idx]),
        })

    submission_df = pd.DataFrame(rows)

    assert len(submission_df) == len(test_df)
    assert submission_df["similarity"].min() >= 0
    assert submission_df["similarity"].max() <= 1

    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(save_path, index=False)
    print(f"Saved submission to {save_path}")


# =========================================================
# Main
# =========================================================

def generate_ensemble_submission():
    TEST_CSV_PATH = PATHS.data / "jaguar-re-id/test.csv"
    OUTPUT_DIR = PATHS.results / "submissions"
    ARTIFACT_DIR = PATHS.results / "ensemble_test_artifacts"

    BATCH_SIZE = 32
    USE_TTA = False
    USE_QE = False
    USE_RERANK = False

    # "embedding_concat" or "score_fusion"
    ENSEMBLE_METHOD = "embedding_concat"

    MODEL_CONFIGS = {
        "EVA-02": {
            "checkpoint_path": PATHS.checkpoints / "backbone_eva02/best_model.pth",
            "backbone_name": "EVA-02",
            "head_type": "triplet",
            "emb_dim": 1024,
            "use_projection": True,
            "use_forward_features": False,
        },
        "MiewID": {
            "checkpoint_path": PATHS.checkpoints / "backbone_miewid/best_model.pth",
            "backbone_name": "MiewID",
            "head_type": "triplet",
            "emb_dim": 1536,   # ggf. anpassen
            "use_projection": True,
            "use_forward_features": False,
        },
        "ConvNeXt-V2": {
            "checkpoint_path": PATHS.checkpoints / "backbone_convnextv2/best_model.pth",
            "backbone_name": "ConvNeXt-V2",
            "head_type": "triplet",
            "emb_dim": 1024,   # ggf. anpassen
            "use_projection": True,
            "use_forward_features": False,
        },
    }

    # Für score_fusion:
    # equal weights: [1, 1, 1]
    # eva+miew favored example: [0.4, 0.4, 0.2]
    SCORE_FUSION_WEIGHTS = {
        "EVA-02": 0.4,
        "MiewID": 0.4,
        "ConvNeXt-V2": 0.2,
    }
    USE_GLOBALMINMAX = True
    SQUARE_SCORES = True

    model_names = list(MODEL_CONFIGS.keys())

    # erster Transform für Loader
    first_model = load_model(MODEL_CONFIGS[model_names[0]])
    test_df, test_loader, dataset_filenames = build_test_loader(
        test_csv_path=TEST_CSV_PATH,
        transform=first_model.backbone_wrapper.transform,
        batch_size=BATCH_SIZE,
    )
    del first_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_embeddings = {}
    all_similarities = {}

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        model = load_model(MODEL_CONFIGS[model_name])
        emb = extract_embeddings(model, test_loader, use_tta=USE_TTA)

        all_embeddings[model_name] = emb
        all_similarities[model_name] = compute_similarity_from_embeddings(
            emb,
            use_qe=USE_QE,
            use_rerank=USE_RERANK,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if ENSEMBLE_METHOD == "embedding_concat":
        ensemble_emb = build_embedding_concat_ensemble(
            [all_embeddings[name] for name in model_names]
        )
        ensemble_sim = compute_similarity_from_embeddings(
            ensemble_emb,
            use_qe=USE_QE,
            use_rerank=USE_RERANK,
        )

        save_embeddings(
            save_dir=ARTIFACT_DIR,
            embeddings=ensemble_emb,
            filenames=dataset_filenames,
            stem="ensemble_embedding_concat",
        )

        build_submission(
            test_df=test_df,
            dataset_filenames=dataset_filenames,
            sim_matrix=ensemble_sim,
            save_path=OUTPUT_DIR / "submission_ensemble_embedding_concat.csv",
        )

    elif ENSEMBLE_METHOD == "score_fusion":
        weights = [SCORE_FUSION_WEIGHTS[name] for name in model_names]

        ensemble_sim = build_score_fusion_ensemble(
            sim_mats=[all_similarities[name] for name in model_names],
            weights=weights,
            use_globalminmax=USE_GLOBALMINMAX,
            square_scores=SQUARE_SCORES,
        )

        save_similarity(
            save_dir=ARTIFACT_DIR,
            sim_matrix=ensemble_sim,
            filenames=dataset_filenames,
            stem="ensemble_score_fusion",
        )

        build_submission(
            test_df=test_df,
            dataset_filenames=dataset_filenames,
            sim_matrix=ensemble_sim,
            save_path=OUTPUT_DIR / "submission_ensemble_score_fusion.csv",
        )

    else:
        raise ValueError(f"Unknown ENSEMBLE_METHOD: {ENSEMBLE_METHOD}")


if __name__ == "__main__":
    generate_ensemble_submission()