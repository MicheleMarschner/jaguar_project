"""
Image-feature and dedup helper utilities for Jaguar dataset curation.

Contains:
- row-aligned metadata builders (aligned to embedding rows)
- cached image features (pHash, sharpness)
- optional embedding kNN helper (retained for alternative candidate generation)
- cache save/load utilities for reproducible dedup runs
"""


from pathlib import Path
from typing import Optional
from tqdm import tqdm
import pandas as pd
from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity
from jaguar.utils.utils import ensure_dir, json_default, save_parquet
import numpy as np
import json
import cv2



def build_meta_from_jaguar_dataset(torch_ds) -> pd.DataFrame:
    """
    Build row-aligned metadata table from JaguarDataset.
    One row per dataset sample. Assumes embedding rows align to dataset order.
    """
    rows = []

    for i, s in enumerate(torch_ds.samples):
        filepath = s[torch_ds.filepath_key]
        filename = s.get(torch_ds.filename_key) or Path(filepath).name

        row = {
            "emb_row": i,      
            "ds_idx": i,       
            "filepath": str(torch_ds._resolve_path(filepath)),
            "filepath_rel": str(filepath),
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
# pHash: Find bursts through image similarity (pixel-level)
# ============================================================

def _compute_phash(filepath: str | Path, hash_size: int = 8):
    img = Image.open(filepath).convert("RGB")
    return imagehash.phash(img, hash_size=hash_size)


def _compute_phash_for_dataset(meta_df: pd.DataFrame, filepath_col: str = "filepath", hash_size: int = 8) -> list:
    phashes = []
    for fp in tqdm(meta_df[filepath_col].tolist(), desc="pHash"):
        try:
            phashes.append(_compute_phash(fp, hash_size=hash_size))
        except Exception as e:
            print(f"pHash failed for {fp}: {e}")
            phashes.append(None)
    return phashes


def add_phash_columns(meta_df: pd.DataFrame, filepath_col: str = "filepath", hash_size: int = 8) -> pd.DataFrame:
    """
    Adds pHash object + hex columns. Keep object column for in-memory sweeps.
    Save hex column to disk.
    """
    out = meta_df.copy()
    out["phash"] = _compute_phash_for_dataset(out, filepath_col=filepath_col, hash_size=hash_size)
    out["phash_hex"] = [str(h) if h is not None else None for h in out["phash"]]
    return out


def phash_distance(h1, h2) -> Optional[int]:
    if h1 is None or h2 is None:
        return None
    return int(h1 - h2)


def _compute_sharpness(filepath: str | Path) -> float:
    image = cv2.imread(str(filepath))
    if image is None:
        return -1.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _compute_sharpness_for_dataset(meta_df: pd.DataFrame, filepath_col: str = "filepath") -> list[float]:
    vals = []
    for fp in tqdm(meta_df[filepath_col].tolist(), desc="Sharpness"):
        try:
            vals.append(_compute_sharpness(fp))
        except Exception as e:
            print(f"Sharpness failed for {fp}: {e}")
            vals.append(-1.0)
    return vals

def add_sharpness_column(meta_df: pd.DataFrame, filepath_col: str = "filepath") -> pd.DataFrame:
    out = meta_df.copy()
    out["sharpness"] = _compute_sharpness_for_dataset(out, filepath_col=filepath_col)
    return out


def save_image_feature_cache(out_dir, file_path, meta_features: pd.DataFrame, config: dict):
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

    print(f"✅ Saved image feature cache to: {out_dir}")


def load_or_create_meta_img_file(out_dir, meta_img_file, jag_meta, phash_size, dataset_name):
    if not meta_img_file.exists():
        meta_img_features = add_phash_columns(jag_meta, filepath_col="filepath", hash_size=phash_size)
        meta_img_features = add_sharpness_column(meta_img_features, filepath_col="filepath")
        save_image_feature_cache(
            out_dir=out_dir,
            file_path=meta_img_file, 
            meta_features=meta_img_features, 
            config = {
                "dataset_name": dataset_name,
                "hash_type": "phash",
                "hash_size": phash_size,
                "sharpness": "laplacian_var_cv2",
            })
    else: 
        meta_img_features = pd.read_parquet(meta_img_file) 

    return  meta_img_features


# ------------------------------------------------------------
# Optional / retained helpers (not used in current "safe all-pairs pHash" pipeline)
# Kept for alternative candidate-generation experiments and future sweeps.
# ------------------------------------------------------------
def find_nearest_neighbors_cosine(embeddings: np.ndarray, k: int = 50):
    """
    embeddings: (N, D)
    Returns cosine similarities + neighbor indices via L2-normalized inner product.
    """
    embs = embeddings.astype("float32").copy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= (norms + 1e-12)

    n, d = embs.shape
    k_eff = min(k, n)

    S = cosine_similarity(embs)
    idxs = np.argsort(-S, axis=1)[:, :k_eff]     # descending
    sims = np.take_along_axis(S, idxs, axis=1)
    return sims.astype(np.float32), idxs.astype(np.int64)


def save_model_knn_edge_candidates(out_dir, candidate_edges_df, precompute_config, file_name="candidate_edges.parquet"):
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

    print(f"✅ Saved model candidate-edge cache to: {out_dir}")

