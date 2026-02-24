from __future__ import annotations

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
    find_nearest_neighbors_cosine, 
    phash_distance, 
    build_meta_from_jaguar_dataset, 
    load_or_create_meta_img_file, 
    save_model_knn_edge_candidates
)

# ============================================================
# kNN (model - dependent): Find bursts through embeddings similarity
# ============================================================

def build_knn_candidate_edges_for_identity(
    sims: np.ndarray,
    idxs: np.ndarray,
    phashes_local: list,
    idx_global: np.ndarray,
    fps_local: list[str],
    identity_id: str,
) -> list[dict]:
    """
    Build unique undirected candidate edge rows for one identity from kNN output.
    Threshold-free: stores raw signals only (cos_sim, phash_dist).
    """
    rows = []
    seen_pairs = set()  # (min_local, max_local)

    n = sims.shape[0]
    for i in range(n):
        # skip self neighbor at column 0 if present
        for j_local, sim in zip(idxs[i, 1:], sims[i, 1:]):
            j_local = int(j_local)
            if i == j_local:
                continue

            a, b = (i, j_local) if i < j_local else (j_local, i)
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))

            d_ph = phash_distance(phashes_local[a], phashes_local[b])

            rows.append(
                {
                    "identity_id": str(identity_id),
                    "src_emb_row": int(idx_global[a]),  # global row index in meta/embeddings
                    "dst_emb_row": int(idx_global[b]),
                    "src_filepath": str(fps_local[a]),
                    "dst_filepath": str(fps_local[b]),
                    "cos_sim": float(sim),
                    "phash_dist": (int(d_ph) if d_ph is not None else None),
                }
            )

    return rows


def precompute_knn_candidate_edges_grouped_by_identity(
    meta_df: pd.DataFrame,
    embeddings_aligned: np.ndarray,
    k_candidates: int = 50,
    identity_col: str = "identity_id",
    filepath_col: str = "filepath",
    phash_col: str = "phash",
) -> pd.DataFrame:
    """
    threshold-independent:
      - per identity cosine kNN
      - build raw candidate edge table with cos_sim + phash_dist

    Returns one row per undirected candidate pair.
    """
    assert len(meta_df) == len(embeddings_aligned), "meta_df and embeddings must align"

    meta = meta_df.reset_index(drop=True).copy()
    all_rows: list[dict] = []

    if phash_col not in meta.columns and "phash_hex" in meta.columns:
        meta = meta.copy()
        meta[phash_col] = [
            imagehash.hex_to_hash(h) if pd.notna(h) and h is not None else None
            for h in meta["phash_hex"]
        ]

    identities = meta[identity_col].dropna().unique().tolist()
    print(f"Precomputing candidates for {len(identities)} identities...")

    for ident in tqdm(identities, desc="Identities"):
        idx_global = meta.index[meta[identity_col] == ident].to_numpy()
        if len(idx_global) < 2:
            continue

        embs_id = embeddings_aligned[idx_global]
        fps_id = meta.loc[idx_global, filepath_col].tolist()
        ph_id = meta.loc[idx_global, phash_col].tolist() if phash_col in meta.columns else [None] * len(idx_global)

        sims, idxs = find_nearest_neighbors_cosine(embs_id, k=min(k_candidates, len(idx_global)))

        rows_id = build_knn_candidate_edges_for_identity(
            sims=sims,
            idxs=idxs,
            phashes_local=ph_id,
            idx_global=idx_global,
            fps_local=fps_id,
            identity_id=str(ident),
        )
        all_rows.extend(rows_id)

    cand_df = pd.DataFrame(all_rows)
    if len(cand_df) == 0:
        cand_df = pd.DataFrame(
            columns=[
                "identity_id",
                "src_emb_row",
                "dst_emb_row",
                "src_filepath",
                "dst_filepath",
                "cos_sim",
                "phash_dist",
            ]
        )

    return cand_df



# ============================================================
# Stage B1: Apply thresholds (cheap sweep)
# ============================================================

def filter_candidate_edges(
    candidate_edges_df: pd.DataFrame,
    sim_threshold: Optional[float] = None,
    phash_threshold: Optional[int] = 4,
    use_or_rule: bool = True,
    require_same_identity: bool = True,
) -> pd.DataFrame:
    """
    Apply threshold rule to raw candidate edges.
    Returns filtered edge table (still one row per undirected pair).
    """
    if candidate_edges_df is None or len(candidate_edges_df) == 0:
        return candidate_edges_df.iloc[0:0].copy()

    df = candidate_edges_df.copy()

    # Boolean masks (allow disabling one branch by passing None)
    sim_ok = pd.Series(False, index=df.index)
    ph_ok = pd.Series(False, index=df.index)

    if sim_threshold is not None:
        sim_ok = df["cos_sim"] >= float(sim_threshold)

    if phash_threshold is not None:
        # phash_dist can be NaN / None
        ph_ok = df["phash_dist"].notna() & (df["phash_dist"] <= int(phash_threshold))

    if use_or_rule:
        keep = sim_ok | ph_ok
    else:
        keep = sim_ok & ph_ok

    out = df.loc[keep].copy()
    out["sim_ok"] = sim_ok.loc[keep].values
    out["phash_ok"] = ph_ok.loc[keep].values
    out["edge_rule"] = "OR" if use_or_rule else "AND"
    out["sim_threshold"] = sim_threshold
    out["phash_threshold"] = phash_threshold

    if require_same_identity and "identity_id" in out.columns:
        # Already per-identity from generation, but keep the guard
        pass

    return out


# ============================================================
# Stage B2: Cluster from filtered edges
# ============================================================

def connected_components_from_edges(
    nodes_emb_rows: np.ndarray,
    edge_df_identity: pd.DataFrame,
) -> list[list[int]]:
    """
    Build connected components over a set of nodes (emb_row IDs) using filtered edges.
    Returns list of components as emb_row lists.
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

        # only non-singletons are burst candidates
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
    Returns winning emb_row from a cluster using:
      centrality (to cluster centroid) + cached sharpness tie-break.
    Requires meta_df['sharpness'].
    """
    cluster_meta = meta_df.set_index(emb_row_col).loc[cluster_emb_rows].reset_index()
    cluster_embs = embeddings_aligned[cluster_meta[emb_row_col].to_numpy()]

    if "sharpness" not in cluster_meta.columns:
        raise KeyError("Expected cached 'sharpness' column in meta_df (load meta_img_features cache).")

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
# Stage B3: Assign burst groups/roles from filtered edges
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
    Cheap re-run step for threshold sweeps:
      - connected components from filtered edges
      - representative selection
      - row-level assignments

    Returns dict with:
      - meta (updated dataframe)
      - clusters_global_row_indices (emb_row lists)
      - summary stats
    """
    assert len(meta_df) == len(embeddings_aligned), "meta_df and embeddings must align"

    meta = meta_df.reset_index(drop=True).copy()
    if emb_row_col not in meta.columns:
        meta[emb_row_col] = np.arange(len(meta), dtype=int)

    # init assignment columns
    meta["burst_group_id"] = None
    meta["burst_cluster_size"] = None
    meta["burst_role"] = "singleton"

    all_clusters: list[list[int]] = []
    rep_filepaths = []
    dup_filepaths = []
    burst_filepaths = []

    identities = meta[identity_col].dropna().unique().tolist()
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

            # assign all as duplicate first
            comp_mask = meta[emb_row_col].isin(comp)
            meta.loc[comp_mask, "burst_group_id"] = group_id
            meta.loc[comp_mask, "burst_cluster_size"] = len(comp)
            meta.loc[comp_mask, "burst_role"] = "duplicate"

            # upgrade winner
            winner_mask = meta[emb_row_col] == winner_emb_row
            meta.loc[winner_mask, "burst_role"] = "representative"

            # bookkeeping
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
    Parquet-oriented artifact saving.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # row-level assignments
    meta_save = meta_df.copy()
    if "phash" in meta_save.columns:
        meta_save["phash_hex"] = [str(h) if h is not None else None for h in meta_save["phash"]]
        meta_save = meta_save.drop(columns=["phash"])

    meta_parquet = out_dir / "burst_assignments.parquet"
    meta_save.to_parquet(meta_parquet, index=False)

    # cluster summary
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

    # raw candidates (threshold-free)
    if candidate_edges_df is not None and len(candidate_edges_df) > 0:
        candidate_edges_df.to_parquet(out_dir / "candidate_edges_raw.parquet", index=False)

    # filtered edges (for this run)
    if filtered_edges_df is not None and len(filtered_edges_df) > 0:
        filtered_edges_df.to_parquet(out_dir / "candidate_edges_filtered.parquet", index=False)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2, default=json_default)

    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=json_default)

    print(f"✅ Saved dedup artifacts to: {out_dir}")


# ============================================================
# Optional convenience wrapper for one sweep run
# ============================================================

def run_burst_assignment_from_precomputed_candidates(
    meta_df: pd.DataFrame,
    embeddings_aligned: np.ndarray,
    candidate_edges_raw_df: pd.DataFrame,
    sim_threshold: float = None,
    phash_threshold: int = 4,
    use_or_rule: bool = True,
    min_cluster_size: int = 2,
):
    filtered_edges_df = filter_candidate_edges(
        candidate_edges_df=candidate_edges_raw_df,
        sim_threshold=sim_threshold,
        phash_threshold=phash_threshold,
        use_or_rule=use_or_rule,
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

    # flatten return
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
    """
    Ensure a FiftyOne sample field exists with the desired type.
    If it exists with the wrong type, delete and recreate it.
    """
    schema = dataset.get_field_schema()

    if field_name in schema and not isinstance(schema[field_name], field_cls):
        dataset.delete_sample_field(field_name)
        schema = dataset.get_field_schema()

    if field_name not in schema:
        dataset.add_sample_field(field_name, field_cls)


def _series_nan_to_none(s: pd.Series) -> pd.Series:
    """Convert pandas NaN/NA to Python None for safe FiftyOne writes."""
    return s.where(s.notna(), None)


def set_values_typed(dataset, view, df: pd.DataFrame, field_name: str, field_cls) -> None:
    """
    Normalize values + ensure FO field schema type + write values.
    Handles pandas NaN/NA safely for String/Int/Float fields.
    """
    # IMPORTANT: force object dtype first, otherwise NaN may stay float NaN
    s = df[field_name].astype("object").copy()

    def _is_missing(x):
        return pd.isna(x)

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
    df = meta_assignments.copy()
    df[filepath_col] = df[filepath_col].astype(str)

    view_all = dataset.select_by(filepath_col, df[filepath_col].tolist())

    # typed field writes (clean + safe)
    if group_col in df.columns:
        set_values_typed(dataset, view_all, df, group_col, fo.StringField)
    if size_col in df.columns:
        set_values_typed(dataset, view_all, df, size_col, fo.IntField)
    if role_col in df.columns:
        set_values_typed(dataset, view_all, df, role_col, fo.StringField)

    # tags from role column
    if role_col in df.columns:
        role_series = _series_nan_to_none(df[role_col])

        burst_fps = df.loc[role_series.isin(["duplicate", "representative"]), filepath_col].astype(str).tolist()
        dup_fps = df.loc[role_series == "duplicate", filepath_col].astype(str).tolist()
        rep_fps = df.loc[role_series == "representative", filepath_col].astype(str).tolist()

        if burst_fps:
            dataset.select_by(filepath_col, burst_fps).tag_samples(burst_tag)
        if dup_fps:
            dataset.select_by(filepath_col, dup_fps).tag_samples(duplicate_tag)
        if rep_fps:
            dataset.select_by(filepath_col, rep_fps).tag_samples(rep_tag)

    dataset.save()

    print(f"Applied dedup assignments to {len(df)} samples.")
    print(f"Tagged burst members: {len(burst_fps)}")
    print(f"Tagged duplicates: {len(dup_fps)}")
    print(f"Tagged representatives: {len(rep_fps)}")


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
    """
    Goal:
      Estimate how pHash threshold behaves on:
      - within-identity pairs (expected to contain more near-duplicates / bursts)
      - cross-identity pairs (safety / collision signal)

    This helps choose a sensible phash_threshold grid/range before the real
    graph+embedding dedup sweep.
    """
    df = meta_df.copy().reset_index(drop=True)

    # Ensure pHash object column exists (reuse cached phash_hex if needed)
    if phash_col not in df.columns:
        df[phash_col] = [
            imagehash.hex_to_hash(h) if pd.notna(h) and h is not None else None
            for h in df[phash_hex_col]
        ]

    # Keep only rows with identity + phash
    df = df[df[identity_col].notna()].copy()
    df = df[df[phash_col].notna()].copy()

    rng = np.random.default_rng(seed)

    # ----------------------------
    # Sample within-identity pairs
    # ----------------------------
    within_dists: list[int] = []

    for ident, g in df.groupby(identity_col, dropna=True):
        idx = g.index.to_numpy()
        n = len(idx)
        if n < 2:
            continue

        # all possible pairs if small, otherwise sample
        # num possible = n*(n-1)/2
        all_pairs_count = (n * (n - 1)) // 2

        if all_pairs_count <= max_within_pairs_per_identity:
            # enumerate all pairs
            for a_pos in range(n):
                for b_pos in range(a_pos + 1, n):
                    h1 = df.at[idx[a_pos], phash_col]
                    h2 = df.at[idx[b_pos], phash_col]
                    d = phash_distance(h1, h2)
                    if d is not None:
                        within_dists.append(int(d))
        else:
            # random sample pairs for this identity
            seen = set()
            target = int(max_within_pairs_per_identity)
            tries = 0
            max_tries = target * 10

            while len(seen) < target and tries < max_tries:
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n))
                tries += 1
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                seen.add((a, b))

                h1 = df.at[idx[a], phash_col]
                h2 = df.at[idx[b], phash_col]
                d = phash_distance(h1, h2)
                if d is not None:
                    within_dists.append(int(d))

    # ----------------------------
    # 2) Sample cross-identity pairs (global)
    # ----------------------------
    cross_dists: list[int] = []

    indices = df.index.to_numpy()
    labels = df[identity_col].astype(str).to_numpy()
    n_total = len(indices)

    if n_total >= 2 and max_cross_pairs_total > 0:
        seen = set()
        tries = 0
        max_tries = max_cross_pairs_total * 20

        while len(cross_dists) < int(max_cross_pairs_total) and tries < max_tries:
            i = int(rng.integers(0, n_total))
            j = int(rng.integers(0, n_total))
            tries += 1
            if i == j:
                continue
            if labels[i] == labels[j]:
                continue

            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))

            h1 = df.at[indices[a], phash_col]
            h2 = df.at[indices[b], phash_col]
            d = phash_distance(h1, h2)
            if d is not None:
                cross_dists.append(int(d))

    within_arr = np.array(within_dists, dtype=np.int32) if within_dists else np.array([], dtype=np.int32)
    cross_arr = np.array(cross_dists, dtype=np.int32) if cross_dists else np.array([], dtype=np.int32)

    rows = []
    n_within = int(len(within_arr))
    n_cross = int(len(cross_arr))

    for t in sorted(set(int(x) for x in thresholds)):
        within_links = int((within_arr <= t).sum()) if n_within else 0
        cross_links = int((cross_arr <= t).sum()) if n_cross else 0

        within_rate = (within_links / n_within) if n_within else 0.0
        cross_rate = (cross_links / n_cross) if n_cross else 0.0

        ratio = (within_rate / cross_rate) if cross_rate > 0 else (np.inf if within_rate > 0 else 0.0)

        rows.append({
            "threshold": int(t),
            "n_within_pairs": n_within,
            "within_links": within_links,
            "within_link_rate": float(within_rate),
            "n_cross_pairs": n_cross,
            "cross_links": cross_links,
            "cross_link_rate": float(cross_rate),
            "link_ratio_within_to_cross": float(ratio) if np.isfinite(ratio) else np.inf,
        })

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def load_or_create_phash_diagnostics(
    meta_df: pd.DataFrame,
    out_file: Path,
    thresholds: list[int],
    identity_col: str = "identity_id",
    phash_col: str = "phash",
    phash_hex_col: str = "phash_hex",
    max_within_pairs_per_identity: int = 200,
    max_cross_pairs_total: int = 5000,
    seed: int = 51,
    force_recompute: bool = False,
) -> pd.DataFrame:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not force_recompute:
        return pd.read_parquet(out_file)

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

    safe_rows = diag_df[diag_df['cross_links'] == 0]
    
    if len(safe_rows) > 0:
        # Pick the highest threshold that is still safe
        COLLISION_FREE_THRESHOLD = int(safe_rows['threshold'].max())
        print(f"✅ Found Safe Threshold: {COLLISION_FREE_THRESHOLD} (0 Collisions)")
        print(f"   -> At this level, you catch {safe_rows.loc[safe_rows['threshold']==COLLISION_FREE_THRESHOLD, 'within_links'].values[0]} duplicates.")
    else:
        # Fallback if even threshold 0 has collisions (rare)
        print("⚠️ Warning: No zero-collision threshold found. Defaulting to strict 2.")
        COLLISION_FREE_THRESHOLD = 2

    # OPTIONAL: Print the 'danger zone' to see what we are avoiding
    next_t = COLLISION_FREE_THRESHOLD + 1
    if next_t in diag_df['threshold'].values:
        bad_row = diag_df[diag_df['threshold'] == next_t].iloc[0]
        print(f"   -> Warning: At threshold {next_t}, you would have {bad_row['cross_links']} collisions!")
    
    return diag_df, COLLISION_FREE_THRESHOLD


if __name__ == "__main__":
    K_CANDIDATES = 50
    MIN_CLUSTER_SIZE = 2
    
    model_name = "MegaDescriptor-L"
    dataset_name = "jaguar_init"

    # 1. LOAD DATA
    fo_wrapper, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export / "init",
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=False,
    )
    print("[Info] Dataset size:", len(torch_ds))

    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    embeddings = load_or_extract_embeddings(model_wrapper, torch_ds, split="training")
    
    jag_meta = build_meta_from_jaguar_dataset(torch_ds)
    assert len(jag_meta) == len(embeddings), "Mismatch in meta vs embeddings"

    # 2. SETUP PATHS
    out_dir = PATHS.runs / "deduplication"
    run_folder = out_dir / f"dedup__{model_name.lower()}"
    precompute_run_folder = run_folder / f"dedup_precompute__{model_name.lower()}__k{K_CANDIDATES}"
    meta_img_file = out_dir / "meta_img_features.parquet"
    phash_size = 8
    candidate_edges_file_name = "candidate_edges.parquet"

    # 3. PRECOMPUTE META & CANDIDATES (WITHIN IDENTITY)
    # This is fine! We only want to find duplicates within the same jaguar folder.
    meta_img_features = load_or_create_meta_img_file(out_dir, meta_img_file, jag_meta, phash_size, dataset_name)

    if not precompute_run_folder.exists():
        print("[Info] Precomputing kNN candidates...")
        cand_raw = precompute_knn_candidate_edges_grouped_by_identity(
            meta_df=meta_img_features,
            embeddings_aligned=embeddings,
            k_candidates=K_CANDIDATES,
            identity_col="identity_id",
            filepath_col="filepath",
            phash_col="phash",
        )
        save_model_knn_edge_candidates(
            out_dir=precompute_run_folder, 
            candidate_edges_df=cand_raw,
            precompute_config={"k": K_CANDIDATES, "model": model_name},
            file_name=candidate_edges_file_name
        )
    else:
        print("[Info] Loading precomputed candidates...")
        cand_raw = pd.read_parquet(precompute_run_folder / candidate_edges_file_name) 

    # 4. CALCULATE SAFE THRESHOLD
    # This checks CROSS-IDENTITY collisions to find the mathematical limit.
    # It ensures our pHash threshold is strict enough.
    phash_diag_file = run_folder / "phash_threshold_diagnostics.parquet"
    
    phash_diag_df, COLLISION_FREE_THRESHOLD = load_or_create_phash_diagnostics(
        meta_df=meta_img_features,
        out_file=phash_diag_file,
        thresholds=list(range(0, 21)),
        max_within_pairs_per_identity=500,
        max_cross_pairs_total=10000,
        seed=SEED,
    )

    # 5. EXECUTE DEDUPLICATION
    # We use the DINO candidates (speed) but filter with SAFE_THRESHOLD (safety)
    print(f"\n[Run] Executing Safe Deduplication with pHash <= {COLLISION_FREE_THRESHOLD}...")
    
    result = run_burst_assignment_from_precomputed_candidates(
        meta_df=meta_img_features,
        embeddings_aligned=embeddings,
        candidate_edges_raw_df=cand_raw,
        sim_threshold=None,              # DISABLE Semantic Merging (Aggressive)
        phash_threshold=COLLISION_FREE_THRESHOLD, # ENABLE Safe Pixel Merging
        use_or_rule=True,
        min_cluster_size=MIN_CLUSTER_SIZE,
    )

    # 6. SAVE ARTIFACTS
    run_dir = run_folder / (
        f"dedup_final__{model_name.lower()}__ph{COLLISION_FREE_THRESHOLD}__SAFE"
    )
    
    # FIXED: Replaced 'chosen_sim', 'chosen_ph' with actual values
    save_dedup_artifacts(
        out_dir=run_dir,
        meta_df=result["meta"],
        summary_dict={
            k: v for k, v in result.items()
            if k not in (
                "meta", "filtered_edges_df", "clusters_global_row_indices",
                "rep_filepaths", "dup_filepaths", "burst_filepaths", 
            )
        },
        config_dict={
            "dataset_name": dataset_name,
            "model_name": model_name,
            "sim_threshold": "None (Safe Mode)",
            "phash_threshold": COLLISION_FREE_THRESHOLD,
            "use_or_rule": True,
            "min_cluster_size": MIN_CLUSTER_SIZE,
        },
        candidate_edges_df=cand_raw,
        filtered_edges_df=result["filtered_edges_df"],
    )
    
    # 7. APPLY TO FIFTYONE
    meta_assignments = pd.read_parquet(run_dir / "burst_assignments.parquet")
    apply_dedup_assignments_to_fiftyone(
        fo_wrapper.get_dataset(),         
        meta_assignments,
    )
    fo_wrapper.export_manifest(PATHS.data_export / "dedup")