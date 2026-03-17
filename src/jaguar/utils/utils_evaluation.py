from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from jaguar.config import EXPERIMENTS_STORE, PATHS, DEVICE
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.utils.utils import ensure_dir, resolve_path, write_json
from jaguar.utils.utils_datasets import (
    build_eval_processing_fn,
    build_processing_fn,
    get_transforms,
    load_split_jaguar_from_FO_export,
)
from jaguar.utils.utils_models import load_or_extract_jaguarid_embeddings


@dataclass
class EvalContext:
    """
    Shared evaluation bundle for one trained JaguarID run, including model, splits, and index mappings.
    """
    model: JaguarIDModel
    train_ds: object
    val_ds: object
    split_df: pd.DataFrame
    parquet_root: Path
    checkpoint_dir: Path
    train_local_to_emb_row: np.ndarray
    val_local_to_emb_row: np.ndarray


def build_local_to_emb_row(
    torch_ds,
    split_df: pd.DataFrame,
    split: str,
    dataset_filename_col: str = "filename",
    split_filename_col: str = "filename",
) -> np.ndarray:
    """
    Map dataset-local sample order to the global emb_row ids stored in the split table.
    """
    split_sub = split_df[split_df["split_final"] == split].copy()
    split_sub[split_filename_col] = split_sub[split_filename_col].astype(str)

    # Safer early failure if filenames are not unique within the split
    dup_mask = split_sub[split_filename_col].duplicated(keep=False)
    if dup_mask.any():
        dup_names = split_sub.loc[dup_mask, split_filename_col].unique()[:10]
        raise ValueError(
            f"Filename -> emb_row mapping is not unique for split='{split}'. "
            f"Example duplicates: {list(dup_names)}"
        )

    filename_to_emb_row = dict(
        zip(split_sub[split_filename_col], split_sub["emb_row"].astype(np.int64))
    )

    local_to_emb_row = np.empty(len(torch_ds), dtype=np.int64)
    missing = []

    for i, sample in enumerate(torch_ds.samples):
        filename = str(sample.get(dataset_filename_col, ""))
        emb_row = filename_to_emb_row.get(filename)

        if emb_row is None:
            missing.append(filename)
        else:
            local_to_emb_row[i] = emb_row

    if missing:
        raise ValueError(
            f"Could not map {len(missing)} dataset samples to emb_row for split='{split}'. "
            f"First missing filenames: {missing[:10]}"
        )

    return local_to_emb_row


def map_emb_rows_to_local_indices(
    emb_rows: np.ndarray,
    local_to_emb_row: np.ndarray,
) -> np.ndarray:
    """
    Convert global emb_row ids back into dataset-local indices for the current split dataset.
    """
    emb_rows = np.asarray(emb_rows, dtype=np.int64)

    local_to_emb_row = np.asarray(local_to_emb_row, dtype=np.int64)
    emb_row_to_local = {int(emb_row): int(local_idx) for local_idx, emb_row in enumerate(local_to_emb_row)}

    missing = [int(e) for e in emb_rows if int(e) not in emb_row_to_local]
    if missing:
        raise ValueError(
            f"Could not map {len(missing)} emb_row ids to local indices. "
            f"First missing: {missing[:10]}"
        )

    return np.asarray([emb_row_to_local[int(e)] for e in emb_rows], dtype=np.int64)


def build_eval_context(
    config: dict,
    train_config: dict, 
    checkpoint_dir: Path | str,
    eval_val_setting: str | None = None,
) -> EvalContext:
    """
    Build the shared evaluation context for one trained run, including datasets, model, and split metadata.
    """

    parquet_root = resolve_path(config["data"]["split_data_path"], EXPERIMENTS_STORE)
    split_df = pd.read_parquet(parquet_root)

    train_processing_fn = build_processing_fn(train_config, split="train")
    if eval_val_setting is None:
        val_processing_fn = build_processing_fn(train_config, split="val")
    else:
        val_processing_fn = build_eval_processing_fn(eval_val_setting, train_config)

    _, train_ds, val_ds = load_split_jaguar_from_FO_export(
        PATHS.data_export / "splits_curated",
        overwrite_db=False,
        parquet_path=parquet_root,
        dataset_name=config["xai"]["dataset_name"],
        train_processing_fn=train_processing_fn,
        val_processing_fn=val_processing_fn,
        include_duplicates=config["split"]["include_duplicates"],
        use_fiftyone=config["data"]["use_fiftyone"],
    )

    num_classes = len(train_ds.label_to_idx)

    model = JaguarIDModel(
        backbone_name=train_config["model"]["backbone_name"],
        num_classes=num_classes,
        head_type=train_config["model"]["head_type"],
        device=DEVICE,
        emb_dim=train_config["model"]["emb_dim"],
        freeze_backbone=train_config["model"]["freeze_backbone"],
        loss_s=train_config["model"].get("s", 30.0),
        loss_m=train_config["model"].get("m", 0.5),
    )

    load_checkpoint_into_model(model, checkpoint_dir / "best_model.pth")

    train_ds.transform = get_transforms(train_config, model.backbone_wrapper, is_training=True)
    val_ds.transform = get_transforms(train_config, model.backbone_wrapper, is_training=False)

    train_local_to_emb_row = build_local_to_emb_row(train_ds, split_df, split="train")
    val_local_to_emb_row = build_local_to_emb_row(val_ds, split_df, split="val")

    return EvalContext(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        split_df=split_df,
        parquet_root=Path(parquet_root),
        checkpoint_dir=checkpoint_dir,
        train_local_to_emb_row=train_local_to_emb_row,
        val_local_to_emb_row=val_local_to_emb_row,
    )



# ============================================================
# Retrieval state from already extracted embeddings
# ============================================================

@dataclass
class RetrievalState:
    """
    Prepared retrieval state containing embeddings, labels, similarities, and burst metadata for query-gallery evaluation.
    """
    q_global: np.ndarray
    g_global: np.ndarray
    sim_matrix: np.ndarray
    labels_q: np.ndarray
    labels_g: np.ndarray
    burst_q: np.ndarray
    burst_g: np.ndarray


def l2_normalize(emb: np.ndarray) -> np.ndarray:
    """
    L2-normalize embedding vectors so cosine-style similarity is computed on a common scale.
    """
    emb = np.asarray(emb, dtype=np.float32)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


def build_query_gallery_retrieval_state(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_global_indices: np.ndarray,
    gallery_global_indices: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    split_df: pd.DataFrame,
) -> RetrievalState:
    """
    Build the full query-gallery retrieval state from embeddings, labels, indices, and burst metadata.
    """
    q_global = np.asarray(query_global_indices, dtype=np.int64)
    g_global = np.asarray(gallery_global_indices, dtype=np.int64)

    emb_q = l2_normalize(query_embeddings)
    emb_g = l2_normalize(gallery_embeddings)

    # Rectangular Similarity [N, M]: Row i = query i, Col j = gallery item j
    sim_matrix = emb_q @ emb_g.T

    # Filter labels
    labels_q = np.asarray(query_labels)
    labels_g = np.asarray(gallery_labels)

    # burst filtering: Skip same-burst matches so “easy positives” aren’t just near-duplicate frames.
    #bg = split_df.set_index("emb_row")["burst_group_id"]
    #burst_q = bg.reindex(q_global).fillna(-1).to_numpy()
    #burst_g = bg.reindex(g_global).fillna(-1).to_numpy()
    bg = split_df.set_index("emb_row")["burst_group_id"]
    burst_q = bg.reindex(q_global).to_numpy()
    burst_g = bg.reindex(g_global).to_numpy()

    return RetrievalState(
        q_global=q_global,
        g_global=g_global,
        sim_matrix=sim_matrix,
        labels_q=labels_q,
        labels_g=labels_g,
        burst_q=burst_q,
        burst_g=burst_g,
    )


def build_query_gallery_retrieval_state_from_sim(
    sim_matrix: np.ndarray,
    query_global_indices: np.ndarray,
    gallery_global_indices: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    split_df: pd.DataFrame,
) -> RetrievalState:
    """
    Build the same retrieval state structure when a similarity matrix is already available.
    """
    q_global = np.asarray(query_global_indices, dtype=np.int64)
    g_global = np.asarray(gallery_global_indices, dtype=np.int64)

    bg = split_df.set_index("emb_row")["burst_group_id"]
    burst_q = bg.reindex(q_global).fillna(-1).to_numpy()
    burst_g = bg.reindex(g_global).fillna(-1).to_numpy()

    return RetrievalState(
        q_global=q_global,
        g_global=g_global,
        sim_matrix=np.asarray(sim_matrix, dtype=np.float64),
        labels_q=np.asarray(query_labels),
        labels_g=np.asarray(gallery_labels),
        burst_q=burst_q,
        burst_g=burst_g,
    )


# ============================================================
# Ranking / evaluation
# ============================================================

def get_ranked_candidates_for_query(retrieval: RetrievalState, i: int):
    """
    Build the valid ranked gallery list for one query after removing self-matches and same-burst items.
    """
    # Get ranks for specific query (descending similarity); processes row i of sim_matrix
    sims_i = retrieval.sim_matrix[i]
    ranked_g_indices = np.argsort(-sims_i)

    q_idx_global = int(retrieval.q_global[i])
    q_label = retrieval.labels_q[i]

    rows = []
    valid_rank = 0

    # iterate through the ranked gallery items
    for g_idx in ranked_g_indices:
        g_idx_global = int(retrieval.g_global[g_idx])

        # Skip self-match
        if q_idx_global == g_idx_global:
            continue

        # Skip same burst group (avoid trivial near-duplicates)
        if retrieval.burst_q[i] != -1 and retrieval.burst_g[g_idx] == retrieval.burst_q[i]:
            continue

        valid_rank += 1

        rows.append({
            "gallery_local_idx": int(g_idx),
            "gallery_global_idx": g_idx_global,
            "gallery_label": retrieval.labels_g[g_idx],
            "sim": float(sims_i[g_idx]),
            "rank_in_gallery": valid_rank,
            "is_same_id": bool(q_label == retrieval.labels_g[g_idx]),
        })

    return q_idx_global, q_label, rows


def evaluate_query_gallery_retrieval(
    retrieval: RetrievalState,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute per-query retrieval metrics and aggregate summary statistics
    from a prepared retrieval state.

    Strict filtering:
    - excludes same-global-index matches (original counterpart / self-match)
    - excludes same-burst matches (near-duplicate easy positives)
    """
    query_rows = []
    ap_list = []
    rank1_hits = []
    skipped_rows = []

    n_queries = len(retrieval.q_global)
    n_queries_no_valid_gallery = 0
    n_queries_no_valid_positive = 0
    n_queries_with_query_burst = 0
    n_queries_without_query_burst = 0
    total_same_burst_candidates_found = 0

    for i in tqdm(range(n_queries), desc="Evaluating Retrieval"):
        q_idx_global = int(retrieval.q_global[i])
        q_label = retrieval.labels_q[i]
        q_burst = retrieval.burst_q[i]

        sims = retrieval.sim_matrix[i]
        g_idx = retrieval.g_global
        g_labels = retrieval.labels_g
        g_bursts = retrieval.burst_g

        # Strict validity mask
        #valid_mask = np.ones(len(g_idx), dtype=bool)
        valid_mask = np.ones(len(g_idx), dtype=bool)

        # remove exact same sample / original counterpart
        #valid_mask &= (g_idx != q_idx_global)
        same_index_mask = (g_idx == q_idx_global)
        valid_mask &= ~same_index_mask

        # remove same-burst samples
        #valid_mask &= (g_bursts != q_burst)
        q_burst_present = pd.notna(q_burst)
        g_burst_present = pd.notna(g_bursts)
        same_burst_mask = q_burst_present & g_burst_present & (g_bursts == q_burst)
        valid_mask &= ~same_burst_mask
        n_same_burst_candidates = int(np.sum(same_burst_mask & ~same_index_mask))
        total_same_burst_candidates_found += n_same_burst_candidates


        #valid_positions = np.where(valid_mask)[0]
        #if len(valid_positions) == 0:
        #    continue

        valid_positions = np.where(valid_mask)[0]
        if len(valid_positions) == 0:
            n_queries_no_valid_gallery += 1
            skipped_rows.append({
                "query_idx": q_idx_global,
                "query_label": q_label,
                "query_burst": q_burst,
                "skip_reason": "no_valid_gallery",
                "n_gallery_total": int(len(g_idx)),
                "n_excluded_same_index": int(np.sum(same_index_mask)),
                "n_excluded_same_burst": n_same_burst_candidates,
                "query_burst_present": bool(pd.notna(q_burst)),
            })
            continue

        sims_valid = sims[valid_positions]
        order = np.argsort(-sims_valid)
        ranked_positions = valid_positions[order]

        ranked_labels = g_labels[ranked_positions]
        rels = (ranked_labels == q_label).astype(np.int64)

        #num_rel = int(rels.sum())
        #if num_rel == 0:
        #    continue

        num_rel = int(rels.sum())
        if num_rel == 0:
            n_queries_no_valid_positive += 1
            skipped_rows.append({
                "query_idx": q_idx_global,
                "query_label": q_label,
                "query_burst": q_burst,
                "skip_reason": "no_valid_positive",
                "n_gallery_total": int(len(g_idx)),
                "n_gallery_valid": int(len(valid_positions)),
                "n_excluded_same_index": int(np.sum(same_index_mask)),
                "n_excluded_same_burst": n_same_burst_candidates,
                "query_burst_present": bool(pd.notna(q_burst)),
            })
            continue

        precision_at_k = np.cumsum(rels) / (np.arange(len(rels)) + 1)
        ap = float((precision_at_k * rels).sum() / num_rel)

        rank1_correct = bool(rels[0] == 1)
        first_pos_rank = int(np.where(rels == 1)[0][0] + 1)

        top1_pos = ranked_positions[0]

        ap_list.append(ap)
        rank1_hits.append(rank1_correct)

        query_rows.append({
            "query_idx": q_idx_global,
            "query_label": q_label,
            "query_burst": q_burst,
            "query_burst_present": bool(pd.notna(q_burst)),
            "n_gallery_total": int(len(g_idx)),
            "n_gallery_valid": int(len(ranked_positions)),
            "n_excluded_same_index": int(np.sum(same_index_mask)),
            "n_excluded_same_burst": n_same_burst_candidates,
            "n_relevant": num_rel,
            "rank1_correct": rank1_correct,
            "ap": ap,
            "first_pos_rank": first_pos_rank,
            "top1_idx": int(retrieval.g_global[top1_pos]),
            "top1_label": retrieval.labels_g[top1_pos],
            "top1_sim": float(retrieval.sim_matrix[i, top1_pos]),
        })

    query_df = pd.DataFrame(query_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    summary = {
        "mAP": float(np.mean(ap_list)) if ap_list else 0.0,
        "rank1": float(np.mean(rank1_hits)) if rank1_hits else 0.0,
        "n_queries_total": int(n_queries),
        "n_queries_eval": int(len(ap_list)),
        "n_queries_no_valid_gallery": int(n_queries_no_valid_gallery),
        "n_queries_no_valid_positive": int(n_queries_no_valid_positive),
        "n_queries_with_query_burst": int(n_queries_with_query_burst),
        "n_queries_without_query_burst": int(n_queries_without_query_burst),
        "total_same_burst_candidates_found": int(total_same_burst_candidates_found),
    }

    return query_df, summary, skipped_df

    """
        query_rows.append({
            "query_idx": q_idx_global,
            "query_label": q_label,
            "n_gallery_valid": int(len(ranked_positions)),
            "n_relevant": num_rel,
            "rank1_correct": rank1_correct,
            "ap": ap,
            "first_pos_rank": first_pos_rank,
            "top1_idx": int(retrieval.g_global[top1_pos]),
            "top1_label": retrieval.labels_g[top1_pos],
            "top1_sim": float(retrieval.sim_matrix[i, top1_pos]),
        })

        
        
    query_df = pd.DataFrame(query_rows)

    summary = {
        "mAP": float(np.mean(ap_list)) if ap_list else 0.0,
        "rank1": float(np.mean(rank1_hits)) if rank1_hits else 0.0,
        "n_queries_eval": len(ap_list),
    }
    

    return query_df, summary
    """


def build_original_gallery_base(config: dict, train_config: dict, checkpoint_dir: Path):
    """
    Build the fixed original gallery state used as the shared reference across query-setting comparisons.
    """

    ctx_orig = build_eval_context(
        config=config,
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        eval_val_setting="original",
    )

    train_embeddings_orig = load_or_extract_jaguarid_embeddings(
        model=ctx_orig.model,
        torch_ds=ctx_orig.train_ds,
        split="train",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix="original_train",
    )

    val_embeddings_orig = load_or_extract_jaguarid_embeddings(
        model=ctx_orig.model,
        torch_ds=ctx_orig.val_ds,
        split="val",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix="original_val",
    )

    gallery_embeddings_orig = np.concatenate(
        [train_embeddings_orig, val_embeddings_orig],
        axis=0,
    )

    gallery_labels_orig = np.concatenate(
        [np.asarray(ctx_orig.train_ds.labels), np.asarray(ctx_orig.val_ds.labels)],
        axis=0,
    )

    gallery_global_indices_orig = np.concatenate(
        [ctx_orig.train_local_to_emb_row, ctx_orig.val_local_to_emb_row],
        axis=0,
    )

    train_files = [str(s["filename"]) for s in ctx_orig.train_ds.samples]
    val_files = [str(s["filename"]) for s in ctx_orig.val_ds.samples]
    gallery_files_orig = train_files + val_files

    return {
        "ctx_orig": ctx_orig,
        "train_embeddings_orig": train_embeddings_orig,
        "val_embeddings_orig": val_embeddings_orig,
        "gallery_embeddings_orig": gallery_embeddings_orig,
        "gallery_labels_orig": gallery_labels_orig,
        "gallery_global_indices_orig": gallery_global_indices_orig,
        "gallery_files_orig": gallery_files_orig,
    }


def build_query_for_setting(
    config: dict,
    ctx_orig,
    setting: str,
) -> dict:
    """
    Build the query-side embeddings and metadata for one evaluation setting against the same fixed gallery.
    """
    val_processing_fn = build_eval_processing_fn(setting, config)
    
    _, _, val_ds = load_split_jaguar_from_FO_export(
        PATHS.data_export / "splits_curated",
        overwrite_db=False,
        parquet_path=ctx_orig.parquet_root,
        dataset_name="jaguar_splits_curated",
        train_processing_fn=None,
        val_processing_fn=val_processing_fn,
        include_duplicates=config["split"]["include_duplicates"],
        use_fiftyone=config["data"]["use_fiftyone"],
    )

    val_ds.transform = get_transforms(
        config,
        ctx_orig.model.backbone_wrapper,
        is_training=False,
    )

    val_local_to_emb_row = build_local_to_emb_row(
        val_ds,
        ctx_orig.split_df,
        split="val",
    )

    query_embeddings = load_or_extract_jaguarid_embeddings(
        model=ctx_orig.model,
        torch_ds=val_ds,
        split="val",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix=f"{setting}_val",
    )

    query_labels = np.asarray(val_ds.labels)
    query_global_indices = val_local_to_emb_row


    #query_local_idx_setting = map_emb_rows_to_local_indices(
    #    emb_rows=query_emb_rows,
    #    local_to_emb_row=ctx_setting.val_local_to_emb_row,
    #)

    #query_embeddings = val_embeddings_setting[query_local_idx_setting]
    #query_labels = np.asarray(ctx_setting.val_ds.labels)[query_local_idx_setting]
    #query_global_indices = query_emb_rows

    return {
        "query_embeddings": query_embeddings,
        "query_labels": query_labels,
        "query_global_indices": query_global_indices,
        "val_ds": val_ds,
    }


def build_val_query_for_setting(
    config: dict,
    ctx_val,
    setting: str,
) -> dict:
    """
    Build val-only query embeddings and metadata for one evaluation setting.

    This is the val-only counterpart to build_query_for_setting(...):
    queries come from the validation split under the requested setting,
    and are intended to be evaluated against a val-only gallery.
    """
    val_processing_fn = build_eval_processing_fn(setting, config)

    _, _, val_ds = load_split_jaguar_from_FO_export(
        PATHS.data_export / "splits_curated",
        overwrite_db=False,
        parquet_path=ctx_val.parquet_root,
        dataset_name="jaguar_splits_curated",
        train_processing_fn=None,
        val_processing_fn=val_processing_fn,
        include_duplicates=config["split"]["include_duplicates"],
        use_fiftyone=config["data"]["use_fiftyone"],
    )

    val_ds.transform = get_transforms(
        config,
        ctx_val.model.backbone_wrapper,
        is_training=False,
    )

    val_local_to_emb_row = build_local_to_emb_row(
        val_ds,
        ctx_val.split_df,
        split="val",
    )

    print("[DEBUG] build_val_query_for_setting before embeddings: setting: {setting}")

    query_embeddings = load_or_extract_jaguarid_embeddings(
        model=ctx_val.model,
        torch_ds=val_ds,
        split="val",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix=f"{setting}_val",
    )

    return {
        "query_embeddings": query_embeddings,
        "query_labels": np.asarray(val_ds.labels),
        "query_global_indices": val_local_to_emb_row,
        "val_ds": val_ds,
    }


def build_val_only_retrieval_for_setting(
    config: dict,
    ctx_val,
    gallery_embeddings_val: np.ndarray,
    gallery_labels_val: np.ndarray,
    gallery_global_indices_val: np.ndarray,
    setting: str,
) -> tuple[RetrievalState, dict]:
    """
    Build a full val-only retrieval state for one query setting.

    Query side:
        validation split under `setting`
    Gallery side:
        original validation split only
    """
    query = build_val_query_for_setting(
        config=config,
        ctx_val=ctx_val,
        setting=setting,
    )

    retrieval = build_query_gallery_retrieval_state(
        query_embeddings=query["query_embeddings"],
        gallery_embeddings=gallery_embeddings_val,
        query_global_indices=query["query_global_indices"],
        gallery_global_indices=gallery_global_indices_val,
        query_labels=query["query_labels"],
        gallery_labels=gallery_labels_val,
        split_df=ctx_val.split_df,
    )

    return retrieval, query


def build_val_gallery_base(
    config: dict,
    train_config: dict,
    checkpoint_dir: Path,
    eval_val_setting: str = "original",
) -> dict:
    """
    Build a val-only retrieval base where the validation split provides both queries and gallery candidates.
    """
    ctx_val = build_eval_context(
        config=config,
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        eval_val_setting=eval_val_setting,
    )

    val_embeddings = load_or_extract_jaguarid_embeddings(
        model=ctx_val.model,
        torch_ds=ctx_val.val_ds,
        split="val",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix=f"{eval_val_setting}_val_gallery",
    )

    val_labels = np.asarray(ctx_val.val_ds.labels)
    val_global_indices = ctx_val.val_local_to_emb_row
    val_files = [str(s["filename"]) for s in ctx_val.val_ds.samples]

    print(f"[DEBUG] val ds length in buidl_val_gallery-base: {len(ctx_val.val_ds)}")
    print(f"[DEBUG] global indices in buidl_val_gallery-base: {val_global_indices}")

    return {
        "ctx_val": ctx_val,
        "val_embeddings": val_embeddings,
        "val_labels": val_labels,
        "val_global_indices": val_global_indices,
        "val_files": val_files,
    }


def build_retrieval_diagnostics_per_query(
    retrieval: RetrievalState,
) -> pd.DataFrame:
    """
    Build per-query retrieval diagnostics using the shared strict protocol.

    Strict filtering:
    - excludes same-global-index matches
    - excludes same-burst matches
    """
    rows = []

    n_queries = len(retrieval.q_global)

    for i in tqdm(range(n_queries), desc="Building Retrieval Diagnostics"):
        q_idx_global = int(retrieval.q_global[i])
        q_label = retrieval.labels_q[i]
        q_burst = retrieval.burst_q[i]

        if pd.notna(q_burst):
            n_queries_with_query_burst += 1
        else:
            n_queries_without_query_burst += 1

        sims = retrieval.sim_matrix[i]
        g_idx = retrieval.g_global
        g_labels = retrieval.labels_g
        g_bursts = retrieval.burst_g

        valid_mask = np.ones(len(g_idx), dtype=bool)
        valid_mask &= (g_idx != q_idx_global)

        if q_burst != -1:
            valid_mask &= (g_bursts != q_burst)

        valid_positions = np.where(valid_mask)[0]
        if len(valid_positions) == 0:
            rows.append({
                "query_idx": q_idx_global,
                "query_label": q_label,
                "gold_rank": np.nan,
                "is_rank1": False,
                "is_rank5": False,
                "best_gold_similarity": np.nan,
                "best_impostor_similarity": np.nan,
                "margin_gold_minus_impostor": np.nan,
            })
            continue

        sims_valid = sims[valid_positions]
        order = np.argsort(-sims_valid)
        ranked_positions = valid_positions[order]

        ranked_labels = g_labels[ranked_positions]
        rels = (ranked_labels == q_label)

        if not rels.any():
            rows.append({
                "query_idx": q_idx_global,
                "query_label": q_label,
                "gold_rank": np.nan,
                "is_rank1": False,
                "is_rank5": False,
                "best_gold_similarity": np.nan,
                "best_impostor_similarity": np.nan,
                "margin_gold_minus_impostor": np.nan,
            })
            continue

        gold_scores = sims[valid_positions][g_labels[valid_positions] == q_label]
        impostor_scores = sims[valid_positions][g_labels[valid_positions] != q_label]

        best_gold_similarity = float(gold_scores.max()) if len(gold_scores) > 0 else np.nan
        best_impostor_similarity = float(impostor_scores.max()) if len(impostor_scores) > 0 else np.nan

        first_gold_rank = int(np.where(rels)[0][0] + 1)

        rows.append({
            "query_idx": q_idx_global,
            "query_label": q_label,
            "gold_rank": first_gold_rank,
            "is_rank1": first_gold_rank <= 1,
            "is_rank5": first_gold_rank <= 5,
            "best_gold_similarity": best_gold_similarity,
            "best_impostor_similarity": best_impostor_similarity,
            "margin_gold_minus_impostor": (
                best_gold_similarity - best_impostor_similarity
                if not np.isnan(best_impostor_similarity) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def save_retrieval_results_per_id(
    save_dir: Path,
    setting: str,
    query_df: pd.DataFrame,
    summary: dict,
    identity_df: pd.DataFrame | None = None,
) -> dict:
    """
    Save per-query, optional per-identity, and summary retrieval results for one evaluation setting.
    """
    out_dir = save_dir / setting
    ensure_dir(out_dir)

    query_df.to_csv(out_dir / "query_metrics.csv", index=False)

    if identity_df is not None:
        identity_df.to_csv(out_dir / "identity_metrics.csv", index=False)

    write_json(summary, out_dir / "summary.json")

    return {
        "query_df": query_df.assign(setting=setting),
        "summary": {"setting": setting, **summary},
        "identity_df": None if identity_df is None else identity_df.assign(setting=setting),
    }


def select_val_samples_from_emb_rows(ctx_orig, query_emb_rows: np.ndarray) -> list[dict]:
    """
    Resolve selected global validation emb_row ids back to the corresponding validation dataset samples.
    """
    val_local_idx = map_emb_rows_to_local_indices(
        query_emb_rows,
        ctx_orig.val_local_to_emb_row,
    )
    return [ctx_orig.val_ds.samples[int(i)] for i in val_local_idx]


def load_checkpoint_into_model(model: JaguarIDModel, checkpoint_path: Path) -> None:
    """
    Load checkpoint weights into a JaguarIDModel and switch it to eval mode on the configured device.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()