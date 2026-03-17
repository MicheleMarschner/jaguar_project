"""
Split and Duplicate-Aware Curation for Jaguar Re-ID.

Project role:
- Produces reproducible train/val split artifacts from the *dedup-annotated manifest*.
- Optionally performs *post-split curation* inside burst groups to reduce near-duplicate bias
  while preserving the split protocol (no moving samples across splits).

Pipeline Steps:
- Construct train/val splits from the burst-annotated manifest.
- Support two protocols: open-set (identity-disjoint) and closed-set (burst-disjoint, identity-overlapping).
- Treat burst groups as atomic units during splitting to avoid leakage of near-duplicates across splits.
- Optionally apply post-split duplicate-aware curation within each split.
- Sub-cluster burst members with a stricter pHash threshold and retain up to K images per sub-cluster.
- Rank retained images by embedding centrality and sharpness.

Purpose:
- Enforce a controlled evaluation protocol while limiting redundancy within splits.
- Provide reproducible split artifacts for experiments on generalization under duplicate-aware data curation.

Notes / assumptions:
- Embeddings must be aligned with dataset order via emb_row indexing:
  embeddings[emb_row] corresponds to the same sample as split_df row with emb_row.
- Some helper functions may remain in the module for alternative workflows; the active path
  is the split + (optional) intra-burst curation described above.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from typing import Optional, List
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import imagehash
try:
    import fiftyone as fo
    HAS_FIFTYONE = True
except ImportError:
    fo = None
    HAS_FIFTYONE = False

from jaguar.config import DATA_ROOT, DEVICE
from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.datasets.FiftyOneDataset import rewrite_samples_json_to_data_relative
from jaguar.utils.utils import ensure_dir, save_parquet, to_rel_path
from jaguar.utils.utils_models import load_or_extract_embeddings
from jaguar.utils.utils_setup import get_split_paths
from jaguar.utils.utils_datasets import get_group_aware_stratified_train_val_split, load_full_jaguar_from_FO_export
from jaguar.utils.utils_split_and_curate import (
    build_split_table_from_torch_dataset,
    print_keep_drop_summary,
    save_split_bundle,
    summarize_splits,
)


def _require_fiftyone() -> None:
    if not HAS_FIFTYONE:
        raise ImportError("FiftyOne is not installed or not usable in this environment.")

# ============================================================
# Splitting Strategies
# ============================================================

def make_open_set_splits(
    df: pd.DataFrame,
    val_split_size: float = 0.2,
    seed: int = 51,
    identity_col: str = "identity_id",
) -> pd.DataFrame:
    """
    Open-Set: Identities are mutually exclusive (Identity-Disjoint).
    Validation measures generalization to entirely new jaguars.

    in detail (identity-disjoint):
      - all images of an identity go to the same split
      - val_split_size targets the fraction of IMAGES in val (approximately)
      - exact ratio may not be achievable because identities are indivisible units
    """
    out = df.copy().reset_index(drop=True)
    out["split_tmp"] = None

    # Calculate counts per identity (identity = indivisible unit in open-set)
    id_counts = out[identity_col].astype(str).value_counts().reset_index()
    id_counts.columns = [identity_col, "n_images"]
    
    # Shuffle for reproducibility
    # (shuffle first, then sort by size desc with stable sort to break ties reproducibly)
    id_counts = id_counts.sample(frac=1.0, random_state=seed).sort_values("n_images", ascending=False)

    # We approximate the requested val image ratio by selecting whole identities (greedy selection). Exact val_split_size is usually 
    # impossible because identities are indivisible.
    target_val = int(len(out) * val_split_size)
    current_val = 0
    val_ids = []

    # Greedy packing
    for _, row in id_counts.iterrows():
        n = row["n_images"]
        if abs(target_val - (current_val + n)) < abs(target_val - current_val):
            val_ids.append(row[identity_col])
            current_val += n

    # Fallback if empty
    if len(val_ids) == 0:
        val_ids = [id_counts.iloc[0][identity_col]]

    val_set = set(val_ids)
    
    out.loc[out[identity_col].astype(str).isin(val_set), "split_tmp"] = "val"
    out["split_tmp"] = out["split_tmp"].fillna("train")
    
    return out


def make_closed_set_splits(
    df: pd.DataFrame,
    val_split_size: float = 0.2,
    seed: int = 51,
    identity_col: str = "identity_id",
) -> pd.DataFrame:
    """
    Closed-Set: Same identity may appear in both train and val, but exact images/bursts stay separated. Useful for measuring generalization 
    to new images of known identities.
    Identities overlap, but Bursts/Images are disjoint.
    
    Uses group-aware stratification:
    1. Collapses data into "Burst Groups" (Atomic Units).
    2. Stratifies these groups based on Identity.
    3. Expands back to image rows.
    """
    train_idx, val_idx, out, _ = get_group_aware_stratified_train_val_split(
        df=df,
        val_split_size=val_split_size,
        seed=seed,
        identity_col=identity_col,
        burst_group_col="burst_group_id",
        filepath_col="filename",
    )

    out["split_tmp"] = None
    out.loc[train_idx, "split_tmp"] = "train"
    out.loc[val_idx, "split_tmp"] = "val"

    # Cleanup internal col
    out.drop(columns=["_split_group_id"], inplace=True, errors="ignore")
    return out


# ============================================================
# Intra-Burst Logic (Clustering & Ranking)
# ============================================================

def get_intra_burst_subclusters(
    phashes: List[Optional[imagehash.ImageHash]], 
    indices: List[int],
    threshold: int
) -> List[List[int]]:
    """
    Given a list of pHashes for a single burst, find connected components 
    where hamming_distance <= threshold.
    Returns lists of global indices (emb_rows).
    """
    if len(indices) < 2:
        return [indices]

    G = nx.Graph()
    G.add_nodes_from(indices)

    # Pairwise comparison within the burst (efficient for small N < 50)
    # We only link if distance is valid and <= threshold
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            h1, h2 = phashes[i], phashes[j]
            if h1 is not None and h2 is not None:
                if (h1 - h2) <= threshold:
                    G.add_edge(indices[i], indices[j])

    # Each component is a tight duplicate group
    return [list(c) for c in nx.connected_components(G)]


def rank_subcluster_by_quality(
    emb_rows: List[int],
    meta_df: pd.DataFrame,
    embeddings: np.ndarray,
    emb_row_col: str = "emb_row"
) -> List[int]:
    """
    Ranks images in a sub-cluster by:
    1. Cosine Centrality (filter to >= median)
    2. Sharpness (sort filtered by desc)
    Returns: Ordered list of emb_rows (best first).
    """
    # Get Embeddings & Centroid
    cluster_embs = embeddings[emb_rows]
    centroid = np.mean(cluster_embs, axis=0, keepdims=True)
    
    # Calculate Centrality
    sims = cosine_similarity(cluster_embs, centroid).flatten()
    
    # Filter by Centrality (Keep top 50%)
    median_sim = np.median(sims)
    
    # Create list of (emb_row, similarity, sharpness)
    candidates = []
    for i, r_idx in enumerate(emb_rows):
        sharp = meta_df.loc[meta_df[emb_row_col] == r_idx, "sharpness"].fillna(-1.0).values[0]
        candidates.append({
            "emb_row": r_idx,
            "sim": sims[i],
            "sharpness": float(sharp),
            "pass_median": sims[i] >= median_sim
        })

    # 4. Sort
    # Primary key: Passed Median? (True > False)
    # Secondary key: Sharpness (High > Low)
    # Tertiary key: Similarity (High > Low) - tiebreaker
    candidates.sort(key=lambda x: (x["pass_median"], x["sharpness"], x["sim"]), reverse=True)

    return [c["emb_row"] for c in candidates]


# ============================================================
# Main Curation Orchestrator
# ============================================================

def apply_post_split_curation(
    split_df: pd.DataFrame,
    meta_img_df: pd.DataFrame,
    embeddings: np.ndarray,
    train_k: int = 1,
    val_k: int = 100,  # effectively "all"
    phash_threshold: int = 2,
) -> pd.DataFrame:
    """
    Iterates through assignments, processes bursts, and flags 'keep_curated'.
    """
    print(f"\n[Curation] Processing splits (Train K={train_k}, Val K={val_k}, pHash Thresh={phash_threshold})...")
    
    out = split_df.copy()
    
    # Attach metadata (sharpness/phash hex)
    # Note: We need the actual pHash objects for calculation, 
    meta_img_df["phash"] = meta_img_df["phash_hex"].apply(
        lambda x: imagehash.hex_to_hash(x) if pd.notna(x) else None
    )

    # Merge meta data onto split DF for easy access
    # After this merge, `out` carries per-image sharpness needed for ranking within subclusters.
    cols_to_merge = ["emb_row", "sharpness", "phash_hex"]
    out = out.merge(
        meta_img_df[cols_to_merge],
        on="emb_row",
        how="left",
    )
    out["phash"] = out["phash_hex"].apply(lambda x: imagehash.hex_to_hash(x) if pd.notna(x) else None)

    out["keep_curated"] = False
    out["curation_reason"] = "dropped"
    
    # -------------------------------------------------
    # A. Handle Singletons (Always Keep)
    # -------------------------------------------------
    singleton_mask = (out["burst_role"] == "singleton") | (out["burst_group_id"].isna())
    out.loc[singleton_mask, "keep_curated"] = True
    out.loc[singleton_mask, "curation_reason"] = "singleton"

    # -------------------------------------------------
    # B. Handle Bursts (Group by Split + BurstID)
    # -------------------------------------------------
    # Filter only burst members
    burst_df = out[~singleton_mask].copy()
    
    # Group by Split -> Burst ID
    grouped = burst_df.groupby(["split_tmp", "burst_group_id"])
    
    # Bursts are curated *within each split* to avoid train/val leakage:
    # we never move images across splits, we only drop/keep inside each split's burst groups.
    for (split_name, burst_id), group in tqdm(grouped, desc="Curating Bursts"):
        
        # Determine K for this split
        k = train_k if split_name == "train" else val_k
        
        # Get data for this burst
        # phashes and indices share the same row order (both derived from the grouped DataFrame).
        indices = group["emb_row"].values
        phashes = group["phash"].tolist()
        
        # 1. Sub-cluster within this burst
        # Subclusters = tighter duplicate sets inside one (looser) burst cluster.
        # We keep top-K per subcluster so a large burst doesn't collapse to a single image.
        subclusters = get_intra_burst_subclusters(phashes, indices, phash_threshold)
        
        # 2. Process each sub-cluster
        for i, cluster_indices in enumerate(subclusters, start=1):

            # assign a unique subcluster ID
            sub_id = f"{burst_id}__sub{i}"
            out.loc[out["emb_row"].isin(cluster_indices), "duplicate_cluster_id"] = sub_id

            # Rank indices by (Centrality + Sharpness)
            ranked_indices = rank_subcluster_by_quality(
                cluster_indices, out, embeddings, emb_row_col="emb_row"
            )
            
            # Select Top K
            kept_indices = ranked_indices[:k]
            
            # Update DataFrame
            out.loc[out["emb_row"].isin(kept_indices), "keep_curated"] = True
            out.loc[out["emb_row"].isin(kept_indices), "curation_reason"] = f"burst_top{k}"

    # Clean up temp cols
    out.drop(columns=["phash"], inplace=True, errors="ignore")
    
    return out


# ============================================================
# FiftyOne and helpers
# ============================================================

def _ensure_sample_field_type(dataset, field_name: str, field_cls):
    _require_fiftyone()
    schema = dataset.get_field_schema()
    if field_name in schema and not isinstance(schema[field_name], field_cls):
        dataset.delete_sample_field(field_name)
        schema = dataset.get_field_schema()
    if field_name not in schema:
        dataset.add_sample_field(field_name, field_cls)

def set_values_typed(dataset, view, df: pd.DataFrame, field_name: str, field_cls) -> None:
    _require_fiftyone()
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

def apply_curation_assignments_to_fiftyone(
    dataset,
    final_df: pd.DataFrame,
    filepath_col: str = "filename",
    split_col: str = "split_final",   # or split_tmp
    keep_col: str = "keep_curated",
    reason_col: str = "curation_reason",
    # tags
    dup_tag: str = "curation_duplicate",
    rep_train_tag: str = "curation_rep_train",
):
    _require_fiftyone()
    df = final_df.copy()
    df[filepath_col] = df[filepath_col].astype(str)

    view_all = dataset.select_by(filepath_col, df[filepath_col].tolist())

    # store fields (so you can filter/sort in FO without relying on tags)
    if split_col in df.columns:
        set_values_typed(dataset, view_all, df, split_col, fo.StringField)
    if keep_col in df.columns:
        _ensure_sample_field_type(dataset, keep_col, fo.BooleanField)
        view_all.set_values(keep_col, df[keep_col].fillna(False).astype(bool).tolist())
    if reason_col in df.columns:
        set_values_typed(dataset, view_all, df, reason_col, fo.StringField)

    # tags
    dup_fps = df.loc[df[reason_col] == "dropped", filepath_col].tolist()

    rep_train_fps = df.loc[
        (df[split_col] == "train") & (df[reason_col] == "burst_top1"),
        filepath_col
    ].tolist()

    if dup_fps:
        dataset.select_by(filepath_col, dup_fps).tag_samples(dup_tag)
    if rep_train_fps:
        dataset.select_by(filepath_col, rep_train_fps).tag_samples(rep_train_tag)

    dataset.save()
    print(f"Applied curation tags/fields to {len(df)} samples.")



def run_phash_sweep(
    split_df,
    meta_img_df,
    embeddings,
    thresholds,
    train_k,
    val_k,
    save_dir=None,
):
    """
    Runs the post-split curation for multiple pHash thresholds and 
    returns a summary of kept/dropped counts per split.

    thresholds: list of int, e.g. [2, 3, 4, 5]
    """

    results = []

    for th in thresholds:
        print(f"\n===== SWEEP: pHash threshold = {th} =====")
        curated = apply_post_split_curation(
            split_df=split_df,
            meta_img_df=meta_img_df,
            embeddings=embeddings,
            train_k=train_k,
            val_k=val_k,
            phash_threshold=th,
        )

        # Compute stats
        df = curated.copy()
        df["kept"] = df["keep_curated"]
        df["dropped"] = ~df["keep_curated"]

        def summarize_for(split_name):
            sub = df[df["split_tmp"] == split_name]
            kept = int(sub["kept"].sum())
            drop = int(sub["dropped"].sum())
            total = len(sub)
            pct_drop = drop / total if total > 0 else 0.0
            return kept, drop, total, pct_drop

        train_stats = summarize_for("train")
        val_stats = summarize_for("val")

        results.append({
            "phash_threshold": th,
            "train_kept": train_stats[0],
            "train_dropped": train_stats[1],
            "train_total": train_stats[2],
            "train_drop_pct": round(train_stats[3], 4),

            "val_kept": val_stats[0],
            "val_dropped": val_stats[1],
            "val_total": val_stats[2],
            "val_drop_pct": round(val_stats[3], 4),
        })

    results_df = pd.DataFrame(results)

    file_path = save_dir / "pHash_threshold_sweep.parquet"
    save_parquet(file_path, results_df)
    print(f"Sweep summary saved → {file_path}")

    return results_df



def create_splits_and_curate(
        split_strategy, 
        include_duplicates, 
        train_k_per_dedup, 
        val_k_per_dedup, 
        phash_thresh_dedup, 
        val_split_size, 
        seed,
        use_fiftyone,
    ):

    fo_dataset_name = "jaguar_burst"
    paths = get_split_paths(
        split_strategy=split_strategy,
        include_duplicates=include_duplicates,
        train_k=train_k_per_dedup,
        val_k=val_k_per_dedup,
        phash_threshold=phash_thresh_dedup,
    )

    manifest_dir = paths["manifest_dir"]
    out_root = paths["write_root"]
    meta_img_file = paths["meta_img_file"]
    export_dir = paths["export_dir"]

    ensure_dir(out_root)
    
    # Load Data
    print(f"Loading {fo_dataset_name}...")
    fo_wrapper, torch_ds = load_full_jaguar_from_FO_export(
        manifest_dir, 
        dataset_name=fo_dataset_name,
        use_fiftyone=use_fiftyone
    ) 
    # Load Embeddings
    model_name = "MegaDescriptor-L"
    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    embeddings = load_or_extract_embeddings(model_wrapper, torch_ds, split="training", num_workers=0)
    
    # Load Metadata (Sharpness/pHash)
    if not meta_img_file.exists():
        raise FileNotFoundError(f"Run burst discovery first to generate meta features: {meta_img_file}")
    meta_img_df = pd.read_parquet(meta_img_file)
    
    # Build Split Table
    # convert dataset into a split-ready metadata table (one row = one image, including 
    # identity and dedup annotations)
    df_full = build_split_table_from_torch_dataset(torch_ds)

    # Generate Raw Splits
    # Assign train/val according to the evaluation protocol (closed-set = known identities; open-set = unseen identities).
    print(f"Generating {split_strategy} splits...")
    if split_strategy == "open_set":
        split_df = make_open_set_splits(df_full, val_split_size=val_split_size, seed=seed)
    elif split_strategy == "closed_set":
        split_df = make_closed_set_splits(df_full, val_split_size=val_split_size, seed=seed)
    else:
        raise ValueError(f"Unknown strategy: {split_strategy}")

    # Apply Post-Split Curation
    # choose which images are eligible for splitting based on dedup policy. This controls whether 
    # training/validation sees all burst members or only deduplicated data.
    if not include_duplicates: 
        # Instead of running only one threshold → run sweep
        PHASH_SWEEP = [2, 3, 4, 5, 6]   # your choice

        sweep_out = run_phash_sweep(
            split_df=split_df,
            meta_img_df=meta_img_df,
            embeddings=embeddings,
            thresholds=PHASH_SWEEP,
            train_k=train_k_per_dedup,
            val_k=val_k_per_dedup,
            save_dir=out_root
        )

        print("\n===== Sweep Results =====")
        print(sweep_out)

        final_df = apply_post_split_curation(
            split_df=split_df,
            meta_img_df=meta_img_df,
            embeddings=embeddings,
            train_k=train_k_per_dedup,
            val_k=val_k_per_dedup,
            phash_threshold=phash_thresh_dedup
        )
    else:
        final_df = split_df.copy()
        final_df["keep_curated"] = True
        final_df["curation_reason"] = "keep_all"
    
    print(len(final_df))
    # Finalize columns
    # Contract: downstream training should filter on keep_curated == True.
    # The full table is kept for audit/debugging.
    final_df["split_final"] = final_df["split_tmp"]
    
    print_keep_drop_summary(final_df)

    config = {
        "strategy": split_strategy,
        "dedup_policy": include_duplicates,
        "train_k": train_k_per_dedup,
        "val_k": val_k_per_dedup,
        "intra_burst_phash_threshold": phash_thresh_dedup,
        "fo_dataset_name": fo_dataset_name,
        "base_dedup_manifest": to_rel_path(manifest_dir),
    }

    summary = summarize_splits(final_df)
    config["data_summary"] = summary
        
    # persist split assignments as reusable experiment artifacts.
    save_split_bundle(
        out_root=out_root,
        final_df=final_df,
        config=config,
    )
    print(f"\nDone. Artifacts saved to {out_root}")
    
    # Change in FiftyOne and export
    if use_fiftyone:
        apply_curation_assignments_to_fiftyone(
            dataset=fo_wrapper.get_dataset(),
            final_df=final_df,
        )
        fo_wrapper.export_manifest(export_dir)
        rewrite_samples_json_to_data_relative(export_dir, DATA_ROOT)
    else:
        print("[Split] use_fiftyone=False -> skipping FiftyOne sync/export")