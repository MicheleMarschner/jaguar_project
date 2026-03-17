import numpy as np
import pandas as pd
from typing import Any


def build_ensemble_results_long_df(
    exp_name: str,
    experiment_group: str | None,
    gallery_protocol: str,
    per_model_metrics: dict[str, dict],
    fusion_metrics: dict[str, dict],
    oracle_summary: dict,
) -> pd.DataFrame:
    """Build one long-form results table for a single ensemble run."""
    rows = []

    member_names = list(per_model_metrics.keys())
    member_set = "+".join(member_names)
    n_members = len(member_names)

    best_single_map = max(float(metrics["mAP"]) for metrics in per_model_metrics.values())
    oracle_map = float(oracle_summary["oracle_mAP"])
    oracle_rank1 = float(oracle_summary["oracle_rank1"])

    for model_name, metrics in per_model_metrics.items():
        model_map = float(metrics["mAP"])
        model_rank1 = float(metrics["rank1"])

        rows.append({
            "experiment_name": exp_name,
            "experiment_group": experiment_group,
            "gallery_protocol": gallery_protocol,
            "method_name": model_name,
            "method_type": "single",
            "member_set": member_set,
            "n_members": n_members,
            "mAP": model_map,
            "rank1": model_rank1,
            "best_single_mAP": best_single_map,
            "gain_vs_best_single": model_map - best_single_map,
            "oracle_mAP": oracle_map,
            "oracle_rank1": oracle_rank1,
            "oracle_gap_mAP": oracle_map - model_map,
        })

    for fusion_name, metrics in fusion_metrics.items():
        fusion_map = float(metrics["mAP"])
        fusion_rank1 = float(metrics["rank1"])

        rows.append({
            "experiment_name": exp_name,
            "experiment_group": experiment_group,
            "gallery_protocol": gallery_protocol,
            "method_name": fusion_name,
            "method_type": "fusion",
            "member_set": member_set,
            "n_members": n_members,
            "mAP": fusion_map,
            "rank1": fusion_rank1,
            "best_single_mAP": best_single_map,
            "gain_vs_best_single": fusion_map - best_single_map,
            "oracle_mAP": oracle_map,
            "oracle_rank1": oracle_rank1,
            "oracle_gap_mAP": oracle_map - fusion_map,
        })

    rows.append({
        "experiment_name": exp_name,
        "experiment_group": experiment_group,
        "gallery_protocol": gallery_protocol,
        "method_name": "oracle",
        "method_type": "oracle",
        "member_set": member_set,
        "n_members": n_members,
        "mAP": oracle_map,
        "rank1": oracle_rank1,
        "best_single_mAP": best_single_map,
        "gain_vs_best_single": oracle_map - best_single_map,
        "oracle_mAP": oracle_map,
        "oracle_rank1": oracle_rank1,
        "oracle_gap_mAP": 0.0,
    })

    return pd.DataFrame(rows)


def build_protocol_comparison_df(results_long_df: pd.DataFrame) -> pd.DataFrame:
    """Compare the same method across gallery protocols."""
    required = {"gallery_protocol", "method_name", "method_type", "mAP", "rank1"}
    missing = required - set(results_long_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    pivot = (
        results_long_df.pivot_table(
            index=["experiment_name", "experiment_group", "method_name", "method_type"],
            columns="gallery_protocol",
            values=["mAP", "rank1"],
            aggfunc="first",
        )
        .reset_index()
    )

    pivot.columns = [
        "__".join(col).strip("_") if isinstance(col, tuple) else col
        for col in pivot.columns
    ]

    if {"mAP__trainval_gallery", "mAP__valonly_gallery"}.issubset(pivot.columns):
        pivot["delta_mAP__trainval_minus_valonly"] = (
            pivot["mAP__trainval_gallery"] - pivot["mAP__valonly_gallery"]
        )

    if {"rank1__trainval_gallery", "rank1__valonly_gallery"}.issubset(pivot.columns):
        pivot["delta_rank1__trainval_minus_valonly"] = (
            pivot["rank1__trainval_gallery"] - pivot["rank1__valonly_gallery"]
        )

    sort_col = "delta_mAP__trainval_minus_valonly"
    if sort_col in pivot.columns:
        pivot = pivot.sort_values(sort_col, ascending=False)

    return pivot


def build_per_query_comparison_df(
    per_model_query_dfs: dict[str, pd.DataFrame],
    fusion_query_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge per-query results from single models and fusion methods into one comparison table."""
    model_names = list(per_model_query_dfs.keys())

    if not model_names:
        raise ValueError("per_model_query_dfs must not be empty")

    base = per_model_query_dfs[model_names[0]][["query_idx", "query_label"]].copy()

    for name, df in per_model_query_dfs.items():
        base = base.merge(
            df[["query_idx", "ap", "rank1_correct", "first_pos_rank", "top1_idx", "top1_label", "top1_sim"]].rename(
                columns={
                    "ap": f"ap__{name}",
                    "rank1_correct": f"rank1__{name}",
                    "first_pos_rank": f"first_pos_rank__{name}",
                    "top1_idx": f"top1_idx__{name}",
                    "top1_label": f"top1_label__{name}",
                    "top1_sim": f"top1_sim__{name}",
                }
            ),
            on="query_idx",
            how="left",
        )

    for name, df in fusion_query_dfs.items():
        base = base.merge(
            df[["query_idx", "ap", "rank1_correct", "first_pos_rank", "top1_idx", "top1_label", "top1_sim"]].rename(
                columns={
                    "ap": f"ap__{name}",
                    "rank1_correct": f"rank1__{name}",
                    "first_pos_rank": f"first_pos_rank__{name}",
                    "top1_idx": f"top1_idx__{name}",
                    "top1_label": f"top1_label__{name}",
                    "top1_sim": f"top1_sim__{name}",
                }
            ),
            on="query_idx",
            how="left",
        )

    model_ap_cols = [f"ap__{name}" for name in model_names]
    model_rank1_cols = [f"rank1__{name}" for name in model_names]

    base["oracle_ap"] = base[model_ap_cols].max(axis=1, skipna=True)
    base["oracle_rank1"] = base[model_rank1_cols].any(axis=1)

    base["best_model"] = base[model_ap_cols].idxmax(axis=1).str.replace("ap__", "", regex=False)
    base["best_model_ap"] = base[model_ap_cols].max(axis=1, skipna=True)

    if "score_fusion" in fusion_query_dfs:
        base["score_fusion_delta_vs_best_model"] = base["ap__score_fusion"] - base["best_model_ap"]
        base["score_fusion_delta_vs_oracle"] = base["ap__score_fusion"] - base["oracle_ap"]
        base["score_fusion_beats_best_model"] = base["ap__score_fusion"] > base["best_model_ap"]
        base["score_fusion_matches_best_model"] = np.isclose(
            base["ap__score_fusion"], base["best_model_ap"], atol=1e-12
        )

    return base


def compute_oracle_from_query_dfs(
    per_model_query_dfs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    """Build an oracle summary from per-model per-query AP and rank-1."""
    model_names = list(per_model_query_dfs.keys())
    if not model_names:
        raise ValueError("per_model_query_dfs must not be empty")

    base = per_model_query_dfs[model_names[0]][["query_idx", "query_label"]].copy()

    for name, df in per_model_query_dfs.items():
        base = base.merge(
            df[["query_idx", "ap", "rank1_correct"]].rename(
                columns={
                    "ap": f"ap__{name}",
                    "rank1_correct": f"rank1__{name}",
                }
            ),
            on="query_idx",
            how="left",
        )

    ap_cols = [f"ap__{name}" for name in model_names]
    rank1_cols = [f"rank1__{name}" for name in model_names]

    base["oracle_ap"] = base[ap_cols].max(axis=1, skipna=True)
    base["oracle_rank1"] = base[rank1_cols].any(axis=1)

    summary = {
        "oracle_mAP": float(base["oracle_ap"].mean()),
        "oracle_rank1": float(base["oracle_rank1"].mean()),
    }

    return base, summary


def build_per_identity_gain_df(
    query_df_base: pd.DataFrame,
    query_df_target: pd.DataFrame,
    identity_col: str = "query_label",
    base_name: str = "best_single",
    target_name: str = "ensemble",
) -> pd.DataFrame:
    """Aggregate per-query gains to per-identity comparison."""
    base = query_df_base[["query_idx", identity_col, "ap", "rank1_correct"]].rename(
        columns={
            "ap": f"ap__{base_name}",
            "rank1_correct": f"rank1__{base_name}",
        }
    )
    target = query_df_target[["query_idx", "ap", "rank1_correct"]].rename(
        columns={
            "ap": f"ap__{target_name}",
            "rank1_correct": f"rank1__{target_name}",
        }
    )

    df = base.merge(target, on="query_idx", how="inner")

    grouped = (
        df.groupby(identity_col, dropna=False)
        .agg(
            queries=("query_idx", "size"),
            base_ap_mean=(f"ap__{base_name}", "mean"),
            target_ap_mean=(f"ap__{target_name}", "mean"),
            base_rank1_mean=(f"rank1__{base_name}", "mean"),
            target_rank1_mean=(f"rank1__{target_name}", "mean"),
        )
        .reset_index()
    )

    grouped["delta_ap"] = grouped["target_ap_mean"] - grouped["base_ap_mean"]
    grouped["delta_rank1"] = grouped["target_rank1_mean"] - grouped["base_rank1_mean"]
    return grouped.sort_values("delta_ap", ascending=False)


def build_compute_summary_df(
    out: dict,
    fusion_results: list[dict[str, Any]],
    members_cfg: list[dict],
) -> pd.DataFrame:
    """Build a compact compute proxy table for reporting."""
    member_rows = []
    for member_cfg in members_cfg:
        name = member_cfg["name"]
        member_out = out["member_outputs"][name]
        member_rows.append(
            {
                "kind": "member",
                "method_name": name,
                "family": "single_model",
                "n_members_used": 1,
                "embedding_dim": int(member_out["query_embeddings"].shape[1]),
            }
        )

    fusion_rows = []
    for res in fusion_results:
        fusion_rows.append(
            {
                "kind": "fusion",
                "method_name": res["name"],
                "family": res["meta"].get("family"),
                "n_members_used": int(res["meta"].get("n_members_used", len(out["member_outputs"]))),
                "embedding_dim": res["meta"].get("embedding_dim"),
            }
        )

    return pd.DataFrame(member_rows + fusion_rows)


def rank1_overlap_from_query_dfs(
    query_df_a: pd.DataFrame,
    query_df_b: pd.DataFrame,
    name_a: str,
    name_b: str,
) -> pd.DataFrame:
    """Summarize overlap of rank-1 correctness between two methods."""
    df = query_df_a[["query_idx", "rank1_correct"]].rename(columns={"rank1_correct": f"rank1__{name_a}"}).merge(
        query_df_b[["query_idx", "rank1_correct"]].rename(columns={"rank1_correct": f"rank1__{name_b}"}),
        on="query_idx",
        how="inner",
    )

    a = df[f"rank1__{name_a}"].astype(bool)
    b = df[f"rank1__{name_b}"].astype(bool)

    rows = [
        {"case": "both_correct", "count": int((a & b).sum())},
        {"case": f"only_{name_a}_correct", "count": int((a & ~b).sum())},
        {"case": f"only_{name_b}_correct", "count": int((~a & b).sum())},
        {"case": "both_wrong", "count": int((~a & ~b).sum())},
    ]
    out = pd.DataFrame(rows)
    out["fraction"] = out["count"] / len(df) if len(df) else np.nan
    return out