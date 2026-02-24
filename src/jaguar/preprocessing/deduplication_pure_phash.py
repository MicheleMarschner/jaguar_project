## ask ai stuid tomorrow
from __future__ import annotations

from collections import deque
from pathlib import Path

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import uuid
import imagehash
import fiftyone as fo

from typing import Optional

from jaguar.config import DEVICE, PATHS, SEED
from jaguar.utils.utils import json_default
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.utils.utils_deduplication import (
    phash_distance, 
    build_meta_from_jaguar_dataset, 
    load_or_create_meta_img_file, 
    save_model_knn_edge_candidates 
)

# ============================================================
# Candidate Generation (Simplified: All-Pairs Within Identity)
# ============================================================

def precompute_all_pairs_phash_within_identity(
    meta_df: pd.DataFrame,
    identity_col: str = "identity_id",
    filepath_col: str = "filepath",
    phash_col: str = "phash",
) -> pd.DataFrame:
    """
    Generates candidate edges by comparing ALL images within the same identity folder.
    Replaces the complex DINO kNN search.
    """
    meta = meta_df.reset_index(drop=True).copy()
    all_rows: list[dict] = []
    
    # Ensure pHash objects exist
    if phash_col not in meta.columns and "phash_hex" in meta.columns:
        meta[phash_col] = [
            imagehash.hex_to_hash(h) if pd.notna(h) and h is not None else None
            for h in meta["phash_hex"]
        ]
    
    # Ensure we have an integer ID for graph nodes (reuse index)
    if "emb_row" not in meta.columns:
        meta["emb_row"] = meta.index.astype(int)

    identities = meta[identity_col].dropna().unique().tolist()
    print(f"Generating all-pairs candidates for {len(identities)} identities...")

    for ident in tqdm(identities, desc="Identities"):
        # Get all images for this jaguar
        group = meta[meta[identity_col] == ident]
        if len(group) < 2:
            continue
            
        indices = group.index.to_numpy()
        phashes = group[phash_col].tolist()
        filepaths = group[filepath_col].tolist()
        emb_rows = group["emb_row"].tolist()
        
        n = len(indices)
        
        # BRUTE FORCE ALL PAIRS (Fast because N is usually small, <100 per id)
        for i in range(n):
            for j in range(i + 1, n): # Only check upper triangle
                
                h1 = phashes[i]
                h2 = phashes[j]
                
                # If either hash is missing, skip
                if h1 is None or h2 is None:
                    continue

                d_ph = phash_distance(h1, h2)
                
                # OPTIONAL SPEEDUP: 
                # If distance is huge (different poses), don't even save the row.
                # This keeps the dataframe small.
                if d_ph is not None and d_ph > 10: 
                    continue

                all_rows.append({
                    "identity_id": str(ident),
                    "src_emb_row": int(emb_rows[i]),
                    "dst_emb_row": int(emb_rows[j]),
                    "src_filepath": str(filepaths[i]),
                    "dst_filepath": str(filepaths[j]),
                    "cos_sim": None, # Unused
                    "phash_dist": int(d_ph) if d_ph is not None else None,
                })

    cand_df = pd.DataFrame(all_rows)
    if len(cand_df) == 0:
        cand_df = pd.DataFrame(columns=[
            "identity_id", "src_emb_row", "dst_emb_row", 
            "src_filepath", "dst_filepath", "cos_sim", "phash_dist"
        ])
    return cand_df


# ============================================================
# Stage B1: Filtering
# ============================================================

def filter_candidate_edges(
    candidate_edges_df: pd.DataFrame,
    phash_threshold: int = 4,
) -> pd.DataFrame:
    """
    Simple filtering: Keep edge only if pHash distance is small enough.
    """
    if candidate_edges_df is None or len(candidate_edges_df) == 0:
        return candidate_edges_df.iloc[0:0].copy()

    df = candidate_edges_df.copy()

    # Filter logic
    ph_ok = df["phash_dist"].notna() & (df["phash_dist"] <= int(phash_threshold))
    
    out = df.loc[ph_ok].copy()
    out["phash_ok"] = True
    out["phash_threshold"] = phash_threshold

    return out


# ============================================================
# Stage B2: Clustering
# ============================================================

def connected_components_from_edges(
    nodes_emb_rows: np.ndarray,
    edge_df_identity: pd.DataFrame,
) -> list[list[int]]:
    """
    Build connected components over a set of nodes (emb_row IDs).
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
# Representative Selection
# ============================================================

def choose_cluster_representative_by_sharpness(
    cluster_emb_rows: list[int],
    meta_df: pd.DataFrame,
    emb_row_col: str = "emb_row",
) -> int:
    """
    Returns winning emb_row from a cluster using purely sharpness.
    """
    cluster_meta = meta_df.set_index(emb_row_col).loc[cluster_emb_rows].reset_index()

    if "sharpness" not in cluster_meta.columns:
        # Fallback if sharpness missing: just take the first one
        return cluster_emb_rows[0]

    sharpness_scores = (
        pd.to_numeric(cluster_meta["sharpness"], errors="coerce")
        .fillna(-1.0)
        .astype(float)
        .tolist()
    )

    # Pick the index with max sharpness
    best_sub_idx = int(np.argmax(sharpness_scores))
    winner_emb_row = int(cluster_meta.iloc[best_sub_idx][emb_row_col])
    return winner_emb_row


# ============================================================
# Stage B3: Assignment
# ============================================================

def assign_bursts_from_filtered_edges(
    meta_df: pd.DataFrame,
    filtered_edges_df: pd.DataFrame,
    min_cluster_size: int =