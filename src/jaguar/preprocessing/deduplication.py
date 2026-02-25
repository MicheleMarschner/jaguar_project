"""
Burst/duplicate assignment pipeline for Jaguar re-ID dataset curation.

Project role:
- detects near-duplicate "burst" images (primarily within the same identity)
- assigns burst_group_id / burst_role (singleton, representative, duplicate)
- selects one representative image per burst for downstream training splits
- saves reproducible dedup artifacts and writes assignments back to FiftyOne

Notes:
- Some helper functions are intentionally kept for alternative/older workflows
  (e.g., embedding kNN candidate generation) and may not be used in the current run path.
"""

from collections import deque
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import uuid
import imagehash
import fiftyone as fo

from typing import Optional

from jaguar.config import DEVICE, PATHS, SEED
from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.utils.utils import json_default
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export, load_or_extract_embeddings
from jaguar.utils.utils_deduplication import (
    phash_distance, 
    build_meta_from_jaguar_dataset, 
    load_or_create_meta_img_file, 
)

# Current candidate-generation strategy:
# compare pHash pairs *within each identity only* (closed-world dedup assumption for burst detection).
# This is cheaper and safer than global all-pairs and avoids cross-identity linking at this stage.
def precompute_vectorized_phash_within_identity(
    meta_df: pd.DataFrame,
    identity_col: str = "identity_id",
    phash_col: str = "phash",
    emb_row_col: str = "emb_row",
) -> pd.DataFrame:
    """
    Optimized 'All-Pairs' generator using Vectorization.
    100x faster than nested python loops.
    """
    # 1. Prepare Data
    meta = meta_df.reset_index(drop=True)
    meta[identity_col] = meta[identity_col].astype(str)
    
    # Expand pHash objects to boolean arrays (Fastest for Hamming dist)
    # Each hash is 64 bits -> 8 bytes -> we can use np.unpackbits or just list conversion
    # Simple way: Convert hex string to int, then to binary array
    
    # Helper: Convert hash list to numpy binary matrix (N x 64)
    def hashes_to_matrix(hash_list):
        # Convert ImageHash objects to boolean arrays
        return np.array([h.hash.flatten() for h in hash_list], dtype=np.int8)

    all_rows = []
    
    # Group by Identity
    grouped = meta.groupby(identity_col)
    
    print(f"Processing {len(grouped)} identities (Vectorized)...")
    
    for ident, group in tqdm(grouped):
        if len(group) < 2:
            continue
            
        # Extract local data
        local_phashes = group[phash_col].tolist()
        local_emb_rows = group[emb_row_col].values
        local_filepaths = group["filepath"].values
        
        # 2. Create Binary Matrix (M images x 64 bits)
        # Filter out Nones first
        valid_mask = [h is not None for h in local_phashes]
        if sum(valid_mask) < 2:
            continue
            
        bin_matrix = hashes_to_matrix([h for h, v in zip(local_phashes, valid_mask) if v])
        valid_indices = np.where(valid_mask)[0]
        
        # 3. Vectorized Hamming Distance (Broadcasting)
        # Shape: (M, 1, 64) != (1, M, 64) -> (M, M, 64) -> Sum over last axis
        # Result: (M, M) distance matrix
        # Note: This is fast for M < 1000. 
        diffs = (bin_matrix[:, None, :] != bin_matrix[None, :, :])
        dist_matrix = diffs.sum(axis=2)
        
        # 4. Extract Pairs (Upper Triangle)
        # We want pairs where dist <= 10 (Soft pre-filter)
        # (We use 10 to be safe, then apply exact Safe Threshold later)
        i_idx, j_idx = np.triu_indices(len(bin_matrix), k=1)
        
        # Filter by distance cap (Optimization)
        # Soft pre-filter cap for candidate cache size.
        # Final dedup decision uses a stricter threshold later (safe threshold diagnostics).
        relevant_mask = dist_matrix[i_idx, j_idx] <= 10 
        
        final_i = i_idx[relevant_mask]
        final_j = j_idx[relevant_mask]
        final_dists = dist_matrix[final_i, final_j]
        
        # Map back to original indices
        orig_i = valid_indices[final_i]
        orig_j = valid_indices[final_j]
        
        # 5. Build Edge List
        # This part is practically instantaneous compared to calculating hashes
        for k in range(len(final_dists)):
            all_rows.append({
                "identity_id": str(ident),
                "src_emb_row": int(local_emb_rows[orig_i[k]]),
                "dst_emb_row": int(local_emb_rows[orig_j[k]]),
                "src_filepath": str(local_filepaths[orig_i[k]]),
                "dst_filepath": str(local_filepaths[orig_j[k]]),
                "phash_dist": int(final_dists[k]),
            })

    return pd.DataFrame(all_rows)


# ============================================================
# Stage B1: Apply thresholds (Strict pHash Only)
# ============================================================

def filter_candidate_edges(
    candidate_edges_df: pd.DataFrame,
    phash_threshold: int = 4,
) -> pd.DataFrame:
    """
    edge decision rule: Apply pHash threshold rule to candidate edges.
    In current mode we use pHash-only thresholding (no embedding similarity rule).
    """
    if candidate_edges_df is None or len(candidate_edges_df) == 0:
        return candidate_edges_df.iloc[0:0].copy()

    df = candidate_edges_df.copy()

    # 1. Check pHash validity and threshold
    # (pHash must exist AND be <= threshold)
    keep = df["phash_dist"].notna() & (df["phash_dist"] <= int(phash_threshold))

    out = df.loc[keep].copy()
    
    # Metadata for debugging
    out["phash_ok"] = True
    out["sim_ok"] = False # We didn't check it, so effectively False
    out["edge_rule"] = "PHASH_ONLY" 
    out["phash_threshold"] = phash_threshold

    return out


# ============================================================
# Stage B2: Cluster from filtered edges
# ============================================================

def connected_components_from_edges(
    nodes_emb_rows: np.ndarray,
    edge_df_identity: pd.DataFrame,
) -> list[list[int]]:
    """
    Build connected components over a set of nodes.
    Each connected component is treated as one burst cluster (duplicate chain closure). If A~B and B~C pass the edge rule, 
    all three are grouped together.
    """
    nodes = [int(x) for x in nodes_emb_rows.tolist()]
    node_set = set(nodes)

    adj = {n: [] for n in nodes}

    if edge_df_identity is not None and len(edge_df_identity) > 0:
        for r in edge_df_identity.itertuples(index=False):
            a = int(r.src_emb_row)
            b = int(r.dst_emb_row)
            if a in node_set and b in node_set and a != b:
                adj[a].append(b)
                adj[b].append(a)

    visited = set()
    comps = []

    for n in nodes:
        if n in visited:
            continue
        q = deque([n])
        visited.add(n)
        comp = [n]

        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
                    comp.append(v)

        if len(comp) >= 2:
            comps.append(sorted(comp))

    return comps


# ============================================================
# Representative selection
# ============================================================

def choose_cluster_representative_by_embrow(
    cluster_emb_rows: list[int],
    meta_df: pd.DataFrame,
    embeddings_aligned: np.ndarray,
    emb_row_col: str = "emb_row",
    filepath_col: str = "filepath",
) -> int:
    """
    Representative policy:
    1) prefer images that are central in embedding space within the burst (avoid outliers)
    2) among sufficiently central images, prefer the sharpest image
    This balances identity typicality and visual quality.
    """
    cluster_meta = meta_df.set_index(emb_row_col).loc[cluster_emb_rows].reset_index()
    cluster_embs = embeddings_aligned[cluster_meta[emb_row_col].to_numpy()]

    if "sharpness" not in cluster_meta.columns:
        raise KeyError("Expected cached 'sharpness' column in meta_df.")

    sharpness_scores = (
        pd.to_numeric(cluster_meta["sharpness"], errors="coerce")
        .fillna(-1.0)
        .astype(float)
        .tolist()
    )

    centroid = np.mean(cluster_embs, axis=0, keepdims=True)
    centrality = cosine_similarity(cluster_embs, centroid).flatten()

    median_cent = np.median(centrality)
    candidate_pos = [t for t, c in enumerate(centrality) if c >= median_cent]
    if len(candidate_pos) == 0:
        candidate_pos = list(range(len(cluster_emb_rows)))

    best_sub_idx = candidate_pos[int(np.argmax([sharpness_scores[t] for t in candidate_pos]))]
    winner_emb_row = int(cluster_meta.iloc[best_sub_idx][emb_row_col])
    return winner_emb_row


# ============================================================
# Stage B3: Assign burst groups/roles
# ============================================================

def assign_bursts_from_filtered_edges(
    meta_df: pd.DataFrame,
    embeddings_aligned: np.ndarray,
    filtered_edges_df: pd.DataFrame,
    min_cluster_size: int = 2,
    identity_col: str = "identity_id",
    emb_row_col: str = "emb_row",
    filepath_col: str = "filepath",
    group_id_prefix: str = "burst",
) -> dict:
    """
    Converts filtered duplicate edges into image-level annotations: 
    burst_group_id, burst_cluster_size, and burst_role (singleton/representative/duplicate).
    """
    assert len(meta_df) == len(embeddings_aligned), "meta_df and embeddings must align"

    meta = meta_df.reset_index(drop=True).copy()
    if emb_row_col not in meta.columns:
        meta[emb_row_col] = np.arange(len(meta), dtype=int)

    # Default state: every image starts as a singleton.
    # Only images that appear in a duplicate-connected component are reassigned to burst roles.
    meta["burst_group_id"] = None
    meta["burst_cluster_size"] = None
    meta["burst_role"] = "singleton"

    all_clusters: list[list[int]] = []
    rep_filepaths = []
    dup_filepaths = []
    burst_filepaths = []

    identities = meta[identity_col].dropna().unique().tolist()

    # Process identities independently to avoid accidental cross-identity burst clusters.
    for ident in identities:
        nodes = meta.loc[meta[identity_col] == ident, emb_row_col].to_numpy(dtype=int)

        if len(nodes) < 2:
            continue

        if filtered_edges_df is None or len(filtered_edges_df) == 0:
            edges_ident = filtered_edges_df.iloc[0:0].copy() if filtered_edges_df is not None else pd.DataFrame()
        else:
            edges_ident = filtered_edges_df.loc[filtered_edges_df["identity_id"] == ident]

        comps = connected_components_from_edges(nodes_emb_rows=nodes, edge_df_identity=edges_ident)
        comps = [c for c in comps if len(c) >= min_cluster_size]

        # Assign one burst group per connected component and mark roles:
        # all members start as duplicates, then one representative is selected. 
        for comp in comps:
            all_clusters.append(comp)

            group_id = f"{group_id_prefix}_{ident}_{uuid.uuid4().hex[:8]}"
            winner_emb_row = choose_cluster_representative_by_embrow(
                cluster_emb_rows=comp,
                meta_df=meta,
                embeddings_aligned=embeddings_aligned,
                emb_row_col=emb_row_col,
                filepath_col=filepath_col,
            )

            comp_mask = meta[emb_row_col].isin(comp)
            meta.loc[comp_mask, "burst_group_id"] = group_id
            meta.loc[comp_mask, "burst_cluster_size"] = len(comp)
            meta.loc[comp_mask, "burst_role"] = "duplicate"

            winner_mask = meta[emb_row_col] == winner_emb_row
            meta.loc[winner_mask, "burst_role"] = "representative"

            burst_fps = meta.loc[comp_mask, filepath_col].tolist()
            rep_fp = meta.loc[winner_mask, filepath_col].iloc[0]
            dup_fps = meta.loc[comp_mask & (~winner_mask), filepath_col].tolist()

            burst_filepaths.extend(burst_fps)
            rep_filepaths.append(rep_fp)
            dup_filepaths.extend(dup_fps)

    summary = {
        "num_images": int(len(meta)),
        "num_identities": int(meta[identity_col].dropna().nunique()),
        "num_burst_clusters": int(len(all_clusters)),
        "num_burst_images": int(len(set(burst_filepaths))),
        "num_duplicates_tagged": int(len(set(dup_filepaths))),
        "num_representatives": int(len(set(rep_filepaths))),
        "num_filtered_edges": int(0 if filtered_edges_df is None else len(filtered_edges_df)),
        "min_cluster_size": int(min_cluster_size),
    }

    return {
        "meta": meta,
        "clusters_global_row_indices": all_clusters,
        "summary": summary,
        "rep_filepaths": sorted(set(rep_filepaths)),
        "dup_filepaths": sorted(set(dup_filepaths)),
        "burst_filepaths": sorted(set(burst_filepaths)),
    }


# ============================================================
# Save / Load artifacts
# ============================================================

def save_dedup_artifacts(
    out_dir,
    meta_df: pd.DataFrame,
    summary_dict: dict,
    config_dict: dict,
    candidate_edges_df: Optional[pd.DataFrame] = None,
    filtered_edges_df: Optional[pd.DataFrame] = None,
):
    """
    Save both image-level assignments and provenance (config/summary/edge caches) 
    so split generation and later experiments can be traced to the exact dedup run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_save = meta_df.copy()
    if "phash" in meta_save.columns:
        meta_save["phash_hex"] = [str(h) if h is not None else None for h in meta_save["phash"]]
        meta_save = meta_save.drop(columns=["phash"])

    meta_parquet = out_dir / "burst_assignments.parquet"
    meta_save.to_parquet(meta_parquet, index=False)

    if "burst_group_id" in meta_save.columns:
        clustered = meta_save[meta_save["burst_group_id"].notna()].copy()
        if len(clustered) > 0:
            rep_map = (
                clustered[clustered["burst_role"] == "representative"]
                .groupby("burst_group_id")["filepath"]
                .first()
                .to_dict()
            )
            cluster_summary = (
                clustered.groupby(["burst_group_id", "identity_id"], dropna=False)
                .agg(
                    cluster_size=("filepath", "count"),
                    num_duplicates=("burst_role", lambda s: int((s == "duplicate").sum())),
                )
                .reset_index()
            )
            cluster_summary["representative_filepath"] = cluster_summary["burst_group_id"].map(rep_map)
            cluster_summary.to_parquet(out_dir / "cluster_summary.parquet", index=False)

    if candidate_edges_df is not None and len(candidate_edges_df) > 0:
        candidate_edges_df.to_parquet(out_dir / "candidate_edges_raw.parquet", index=False)

    if filtered_edges_df is not None and len(filtered_edges_df) > 0:
        filtered_edges_df.to_parquet(out_dir / "candidate_edges_filtered.parquet", index=False)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2, default=json_default)

    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=json_default)

    print(f"✅ Saved dedup artifacts to: {out_dir}")


# ============================================================
# Wrapper
# ============================================================

def run_burst_assignment_from_precomputed_candidates(
    meta_df: pd.DataFrame,
    embeddings_aligned: np.ndarray,
    candidate_edges_raw_df: pd.DataFrame,
    phash_threshold: int = 4,
    min_cluster_size: int = 2,
):

    filtered_edges_df = filter_candidate_edges(
        candidate_edges_df=candidate_edges_raw_df,
        phash_threshold=phash_threshold,
    )

    assigned = assign_bursts_from_filtered_edges(
        meta_df=meta_df,
        embeddings_aligned=embeddings_aligned,
        filtered_edges_df=filtered_edges_df,
        min_cluster_size=min_cluster_size,
        identity_col="identity_id",
        emb_row_col="emb_row",
        filepath_col="filepath",
    )

    out = assigned["summary"].copy()
    out["meta"] = assigned["meta"]
    out["filtered_edges_df"] = filtered_edges_df
    out["clusters_global_row_indices"] = assigned["clusters_global_row_indices"]
    out["rep_filepaths"] = assigned["rep_filepaths"]
    out["dup_filepaths"] = assigned["dup_filepaths"]
    out["burst_filepaths"] = assigned["burst_filepaths"]
    return out


# ============================================================
# FiftyOne and helpers
# ============================================================

def _ensure_sample_field_type(dataset, field_name: str, field_cls):
    schema = dataset.get_field_schema()
    if field_name in schema and not isinstance(schema[field_name], field_cls):
        dataset.delete_sample_field(field_name)
        schema = dataset.get_field_schema()
    if field_name not in schema:
        dataset.add_sample_field(field_name, field_cls)

def _series_nan_to_none(s: pd.Series) -> pd.Series:
    return s.where(s.notna(), None)

def set_values_typed(dataset, view, df: pd.DataFrame, field_name: str, field_cls) -> None:
    s = df[field_name].astype("object").copy()
    def _is_missing(x): return pd.isna(x)

    if field_cls is fo.StringField:
        values = [None if _is_missing(x) else str(x) for x in s.tolist()]
    elif field_cls is fo.IntField:
        values = [None if _is_missing(x) else int(x) for x in s.tolist()]
    elif field_cls is fo.FloatField:
        values = [None if _is_missing(x) else float(x) for x in s.tolist()]
    else:
        values = [None if _is_missing(x) else x for x in s.tolist()]

    _ensure_sample_field_type(dataset, field_name, field_cls)
    view.set_values(field_name, values)


def apply_dedup_assignments_to_fiftyone(
    dataset,  
    meta_assignments: pd.DataFrame,
    filepath_col: str = "filepath",
    group_col: str = "burst_group_id",
    size_col: str = "burst_cluster_size",
    role_col: str = "burst_role",
    duplicate_tag: str = "duplicate",
    burst_tag: str = "burst",
    rep_tag: str = "burst_representative",
):
    """
    Mirrors dedup assignments into FiftyOne fields/tags for visual QA and manual inspection.
    """
    df = meta_assignments.copy()
    df[filepath_col] = df[filepath_col].astype(str)
    view_all = dataset.select_by(filepath_col, df[filepath_col].tolist())

    if group_col in df.columns:
        set_values_typed(dataset, view_all, df, group_col, fo.StringField)
    if size_col in df.columns:
        set_values_typed(dataset, view_all, df, size_col, fo.IntField)
    if role_col in df.columns:
        set_values_typed(dataset, view_all, df, role_col, fo.StringField)

    if role_col in df.columns:
        role_series = _series_nan_to_none(df[role_col])
        burst_fps = df.loc[role_series.isin(["duplicate", "representative"]), filepath_col].astype(str).tolist()
        dup_fps = df.loc[role_series == "duplicate", filepath_col].astype(str).tolist()
        rep_fps = df.loc[role_series == "representative", filepath_col].astype(str).tolist()

        if burst_fps: dataset.select_by(filepath_col, burst_fps).tag_samples(burst_tag)
        if dup_fps: dataset.select_by(filepath_col, dup_fps).tag_samples(duplicate_tag)
        if rep_fps: dataset.select_by(filepath_col, rep_fps).tag_samples(rep_tag)

    dataset.save()
    print(f"Applied dedup assignments to {len(df)} samples.")


# ============================================================
# Diagnostics
# ============================================================

def compute_phash_threshold_diagnostics(
    meta_df: pd.DataFrame,
    thresholds: list[int],
    identity_col: str = "identity_id",
    phash_col: str = "phash",
    phash_hex_col: str = "phash_hex",
    max_within_pairs_per_identity: int = 200,
    max_cross_pairs_total: int = 5000,
    seed: int = 51,
) -> pd.DataFrame:
    df = meta_df.copy().reset_index(drop=True)
    if phash_col not in df.columns:
        df[phash_col] = [
            imagehash.hex_to_hash(h) if pd.notna(h) else None
            for h in df[phash_hex_col]
        ]
    df = df[df[identity_col].notna() & df[phash_col].notna()].copy()
    rng = np.random.default_rng(seed)

    # 1. Within-Identity
    within_dists: list[int] = []
    for ident, g in df.groupby(identity_col, dropna=True):
        idx = g.index.to_numpy()
        n = len(idx)
        if n < 2: continue
        
        all_pairs_count = (n * (n - 1)) // 2
        if all_pairs_count <= max_within_pairs_per_identity:
            for a_pos in range(n):
                for b_pos in range(a_pos + 1, n):
                    d = phash_distance(df.at[idx[a_pos], phash_col], df.at[idx[b_pos], phash_col])
                    if d is not None: within_dists.append(int(d))
        else:
            seen = set()
            target = int(max_within_pairs_per_identity)
            tries = 0
            while len(seen) < target and tries < target * 10:
                i, j = rng.integers(0, n, size=2)
                tries += 1
                if i == j: continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen: continue
                seen.add((a, b))
                d = phash_distance(df.at[idx[a], phash_col], df.at[idx[b], phash_col])
                if d is not None: within_dists.append(int(d))

    # 2. Cross-Identity
    cross_dists: list[int] = []
    indices = df.index.to_numpy()
    labels = df[identity_col].astype(str).to_numpy()
    n_total = len(indices)
    
    if n_total >= 2 and max_cross_pairs_total > 0:
        seen = set()
        tries = 0
        while len(cross_dists) < int(max_cross_pairs_total) and tries < max_cross_pairs_total * 20:
            i, j = rng.integers(0, n_total, size=2)
            tries += 1
            if i == j or labels[i] == labels[j]: continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen: continue
            seen.add((a, b))
            d = phash_distance(df.at[indices[a], phash_col], df.at[indices[b], phash_col])
            if d is not None: cross_dists.append(int(d))

    within_arr = np.array(within_dists, dtype=np.int32)
    cross_arr = np.array(cross_dists, dtype=np.int32)
    rows = []
    n_within = len(within_arr)
    n_cross = len(cross_arr)

    for t in sorted(set(int(x) for x in thresholds)):
        w_links = int((within_arr <= t).sum()) if n_within else 0
        c_links = int((cross_arr <= t).sum()) if n_cross else 0
        w_rate = (w_links / n_within) if n_within else 0.0
        c_rate = (c_links / n_cross) if n_cross else 0.0
        rows.append({
            "threshold": int(t),
            "n_within_pairs": n_within,
            "within_links": w_links,
            "n_cross_pairs": n_cross,
            "cross_links": c_links,
            "cross_link_rate": float(c_rate),
        })

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def load_or_create_phash_diagnostics(
    meta_df: pd.DataFrame,
    out_file: Path,
    thresholds: list[int],
    identity_col: str,
    phash_col: str,
    phash_hex_col: str,
    max_within_pairs_per_identity: int,
    max_cross_pairs_total: int,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        diag_df = pd.read_parquet(out_file)
    else:
        diag_df = compute_phash_threshold_diagnostics(
            meta_df=meta_df,
            thresholds=thresholds,
            identity_col=identity_col,
            phash_col=phash_col,
            phash_hex_col=phash_hex_col,
            max_within_pairs_per_identity=max_within_pairs_per_identity,
            max_cross_pairs_total=max_cross_pairs_total,
            seed=seed,
        )
        diag_df.to_parquet(out_file, index=False)

    # Determine Safe Threshold
    safe_rows = diag_df[diag_df['cross_links'] == 0]
    
    if len(safe_rows) > 0:
        COLLISION_FREE_THRESHOLD = int(safe_rows['threshold'].max())
        print(f"✅ Found Safe Threshold: {COLLISION_FREE_THRESHOLD} (0 Collisions)")
    else:
        print("⚠️ Warning: No zero-collision threshold found. Defaulting to strict 2.")
        COLLISION_FREE_THRESHOLD = 2
    
    return diag_df, COLLISION_FREE_THRESHOLD


if __name__ == "__main__":
    # --- CONFIG ---
    MIN_CLUSTER_SIZE = 2
    model_name = "MegaDescriptor-L"
    dataset_name = "jaguar_init"
    
    # 1. LOAD DATA & MODEL
    # Embeddings are only used for representative selection (centrality), not for candidate search in this pipeline.
    fo_wrapper, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export / "init", dataset_name=dataset_name, processing_fn=None
    )
    print(f"[Info] Dataset: {len(torch_ds)}")

    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    embeddings = load_or_extract_embeddings(model_wrapper, torch_ds, split="training")
    
    jag_meta = build_meta_from_jaguar_dataset(torch_ds)
    
    # 2. SETUP
    out_dir = PATHS.runs / "deduplication"
    run_folder = out_dir / f"dedup__{model_name.lower()}"
    precompute_run_folder = run_folder / "precompute_candidates_all_pairs"
    meta_img_file = out_dir / "meta_img_features.parquet"
    phash_size = 8
    
    # 3. LOAD/COMPUTE META FEATURES
    meta_img_features = load_or_create_meta_img_file(out_dir, meta_img_file, jag_meta, phash_size, dataset_name)
    meta_img_features["identity_id"] = meta_img_features["identity_id"].astype(str)

    # 4. PRECOMPUTE CANDIDATES (ALL-PAIRS WITHIN IDENTITY)
    cand_file = precompute_run_folder / "candidate_edges_all_pairs.parquet"
    if not cand_file.exists():
        precompute_run_folder.mkdir(parents=True, exist_ok=True)
        cand_raw = precompute_vectorized_phash_within_identity(
            meta_df=meta_img_features,
            identity_col="identity_id",
        )
        cand_raw.to_parquet(cand_file, index=False)
        print(f"[Info] Saved candidates: {cand_file}")
    else:
        print(f"[Info] Loading candidates: {cand_file}")
        cand_raw = pd.read_parquet(cand_file)

    # 5. FIND SAFE THRESHOLD (ZERO COLLISIONS)
    # Choose a conservative pHash threshold using sampled within-ID vs cross-ID diagnostics.
    # Goal: collision-free threshold on sampled cross-identity pairs ("safe mode").
    diag_file = run_folder / "phash_diagnostics.parquet"
    diag_df, SAFE_THRESHOLD = load_or_create_phash_diagnostics(
        meta_df=meta_img_features,
        out_file=diag_file,
        thresholds=list(range(0, 21)),
        identity_col="identity_id",
        phash_col="phash",
        phash_hex_col="phash_hex",
        max_within_pairs_per_identity=500,
        max_cross_pairs_total=10000,
        seed=SEED
    )

    # 6. RUN DEDUPLICATION
    # Final dedup run = strict pHash-only clustering + representative selection.
    print(f"\n[Run] Executing Safe Deduplication (pHash <= {SAFE_THRESHOLD})...")
    result = run_burst_assignment_from_precomputed_candidates(
        meta_df=meta_img_features,
        embeddings_aligned=embeddings,
        candidate_edges_raw_df=cand_raw,
        phash_threshold=SAFE_THRESHOLD,  # Strict Safety
        min_cluster_size=MIN_CLUSTER_SIZE,
    )

    # 7. SAVE FINAL ARTIFACTS
    final_dir = run_folder / f"dedup_final_SAFE_ph{SAFE_THRESHOLD}"
    save_dedup_artifacts(
        out_dir=final_dir,
        meta_df=result["meta"],
        summary_dict={k: v for k, v in result.items() if k not in ("meta", "filtered_edges_df", "clusters_global_row_indices", "rep_filepaths", "dup_filepaths", "burst_filepaths")},
        config_dict={"method": "safe_all_pairs", "safe_threshold": SAFE_THRESHOLD, "sim_threshold": "None"},
        candidate_edges_df=cand_raw,
        filtered_edges_df=result["filtered_edges_df"],
    )

    # 8. EXPORT TO FIFTYONE
    # Write assignments back to FiftyOne and export a dedup-annotated manifest for downstream split generation.
    apply_dedup_assignments_to_fiftyone(fo_wrapper.get_dataset(), pd.read_parquet(final_dir / "burst_assignments.parquet"))
    fo_wrapper.export_manifest(PATHS.data_export / "dedup")
    print("\n[Done] Deduplication Complete.")