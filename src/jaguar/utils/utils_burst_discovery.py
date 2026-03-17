"""
Provide image-feature and deduplication helpers for Jaguar dataset curation.

Includes:
- row-aligned metadata builders
- cached image features such as pHash and sharpness
- optional embedding-based kNN helpers
- cache save/load utilities for reproducible dedup runs
"""

from pathlib import Path
from typing import Optional
from tqdm import tqdm
import pandas as pd
from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import cv2

from jaguar.utils.utils import ensure_dir, json_default, save_parquet, to_abs, to_rel_path


def build_meta_from_jaguar_dataset(torch_ds) -> pd.DataFrame:
    """Build a row-aligned metadata table from a JaguarDataset instance."""
    rows = []

    for i, s in enumerate(torch_ds.samples):
        filepath = s[torch_ds.filepath_key]
        filename = s.get(torch_ds.filename_key) or Path(filepath).name
        packed = to_rel_path(torch_ds._resolve_path(filepath))

        row = {
            "emb_row": i,
            "ds_idx": i,
            "filepath_root": packed["root"],
            "filepath_rel": packed["rel"],
            "filename": filename,
        }

        if not torch_ds.is_test:
            row["identity_id"] = str(s["ground_truth"]["label"])
            row["identity_idx"] = int(torch_ds.labels_idx[i])
        else:
            row["identity_id"] = None
            row["identity_idx"] = None

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# pHash: Find bursts through image similarity
# ============================================================

def _compute_phash(filepath: str | Path, hash_size: int = 8):
    """Compute the perceptual hash of an image file."""
    img = Image.open(filepath).convert("RGB")
    return imagehash.phash(img, hash_size=hash_size)


def _compute_phash_for_dataset(meta_df: pd.DataFrame, hash_size: int = 8) -> list:
    """Compute perceptual hashes for all images listed in the metadata table."""
    phashes = []
    for root, rel in tqdm(
        meta_df[["filepath_root", "filepath_rel"]].itertuples(index=False, name=None),
        desc="pHash",
        total=len(meta_df),
    ):
        fp = None
        try:
            fp = to_abs(root, rel)
            phashes.append(_compute_phash(fp, hash_size=hash_size))
        except Exception as e:
            print(f"pHash failed for {fp}: {e}")
            phashes.append(None)
    return phashes


def add_phash_columns(meta_df: pd.DataFrame, hash_size: int = 8) -> pd.DataFrame:
    """Add perceptual-hash object and hex-string columns to the metadata table."""
    out = meta_df.copy()
    out["phash"] = _compute_phash_for_dataset(out, hash_size=hash_size)
    out["phash_hex"] = [str(h) if h is not None else None for h in out["phash"]]
    return out


def phash_distance(h1, h2) -> Optional[int]:
    """Return the Hamming distance between two perceptual hashes."""
    if h1 is None or h2 is None:
        return None
    return int(h1 - h2)


def _compute_sharpness(filepath: str | Path) -> float:
    """Compute image sharpness using the variance of the Laplacian."""
    image = cv2.imread(str(filepath))
    if image is None:
        return -1.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _compute_sharpness_for_dataset(meta_df: pd.DataFrame) -> list[float]:
    """Compute sharpness scores for all images listed in the metadata table."""
    vals = []
    for root, rel in tqdm(
        meta_df[["filepath_root", "filepath_rel"]].itertuples(index=False, name=None),
        desc="Sharpness",
        total=len(meta_df),
    ):
        fp = None
        try:
            fp = to_abs(root, rel)
            vals.append(_compute_sharpness(fp))
        except Exception as e:
            print(f"Sharpness failed for {fp}: {e}")
            vals.append(-1.0)
    return vals


def add_sharpness_column(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Add a sharpness column to the metadata table."""
    out = meta_df.copy()
    out["sharpness"] = _compute_sharpness_for_dataset(out)
    return out


def save_image_feature_cache(out_dir, file_path, meta_features: pd.DataFrame, config: dict):
    """Save image-feature metadata together with a summary and configuration file."""
    ensure_dir(out_dir)

    meta_save = meta_features.copy()
    if "phash" in meta_save.columns:
        meta_save["phash_hex"] = [str(h) if h is not None else None for h in meta_save["phash"]]
        meta_save = meta_save.drop(columns=["phash"])

    save_parquet(file_path, meta_save)

    summary = {
        "num_rows": int(len(meta_save)),
        "num_identities": int(meta_save["identity_id"].dropna().nunique()) if "identity_id" in meta_save.columns else None,
        "columns": list(meta_save.columns),
    }

    with open(out_dir / "image_features_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    with open(out_dir / "image_features_config.json", "w") as f:
        json.dump(config, f, indent=2, default=json_default)

    print(f"Saved image feature cache to: {out_dir}")


def load_or_create_meta_img_file(out_dir, meta_img_file, jag_meta, phash_size, dataset_name):
    """Load cached image features or create them if the cache does not exist."""
    if not meta_img_file.exists():
        meta_img_features = add_phash_columns(jag_meta, hash_size=phash_size)
        meta_img_features = add_sharpness_column(meta_img_features)
        save_image_feature_cache(
            out_dir=out_dir,
            file_path=meta_img_file,
            meta_features=meta_img_features,
            config={
                "dataset_name": dataset_name,
                "hash_type": "phash",
                "hash_size": phash_size,
                "sharpness": "laplacian_var_cv2",
            },
        )
    else:
        meta_img_features = pd.read_parquet(meta_img_file)

    return meta_img_features


# ------------------------------------------------------------
# Optional helpers retained for alternative candidate generation
# ------------------------------------------------------------
def find_nearest_neighbors_cosine(embeddings: np.ndarray, k: int = 50):
    """Find the top-k cosine neighbors for each embedding vector."""
    embs = embeddings.astype("float32").copy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= (norms + 1e-12)

    n, d = embs.shape
    k_eff = min(k, n)

    S = cosine_similarity(embs)
    idxs = np.argsort(-S, axis=1)[:, :k_eff]
    sims = np.take_along_axis(S, idxs, axis=1)
    return sims.astype(np.float32), idxs.astype(np.int64)


def save_model_knn_edge_candidates(out_dir, candidate_edges_df, precompute_config, file_name="candidate_edges.parquet"):
    """Save embedding-based candidate edges together with summary and configuration files."""
    ensure_dir(out_dir)
    save_parquet(out_dir / file_name, candidate_edges_df)

    summary = {
        "num_candidate_edges": int(len(candidate_edges_df)),
        "columns": list(candidate_edges_df.columns),
    }

    with open(out_dir / "precompute_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    with open(out_dir / "precompute_config.json", "w") as f:
        json.dump(precompute_config, f, indent=2, default=json_default)

    print(f"Saved model candidate-edge cache to: {out_dir}")