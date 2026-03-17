from __future__ import annotations

from itertools import combinations
from pathlib import Path

from jaguar.analysis.kaggle_ensemble.plot_ensemble_analysis import create_ensemble_plots
import numpy as np
import pandas as pd

from jaguar.utils.utils import ensure_dir


def find_run_dirs(group_dir: Path) -> list[Path]:
    """Find run directories that contain ensemble result summaries."""
    return sorted(
        p.parent
        for p in group_dir.rglob("results_long_all_protocols.csv")
        if p.is_file()
    )


def load_results_long(run_dir: Path) -> pd.DataFrame:
    """Load the per-run long results table across protocols."""
    path = run_dir / "results_long_all_protocols.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)
    df["run_dir"] = str(run_dir)
    df["run_name"] = run_dir.name
    return df


def load_per_query_comparison(run_dir: Path, protocol: str) -> pd.DataFrame:
    """Load per-query comparison table for one run and protocol."""
    path = run_dir / f"per_query_comparison__{protocol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def get_single_and_fusion_methods(
    results_long_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Extract single-model and fusion method names from one results-long slice."""
    singles = (
        results_long_df.loc[
            results_long_df["method_type"] == "single", "method_name"
        ]
        .dropna()
        .tolist()
    )
    fusions = (
        results_long_df.loc[
            results_long_df["method_type"] == "fusion", "method_name"
        ]
        .dropna()
        .tolist()
    )
    return singles, fusions


def build_run_protocol_summary_df(all_results_long_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize best single, best fusion, and oracle performance per run and protocol."""
    rows = []
    group_cols = [
        "run_name",
        "run_dir",
        "experiment_name",
        "experiment_group",
        "gallery_protocol",
    ]

    for keys, df_sub in all_results_long_df.groupby(group_cols, dropna=False):
        run_name, run_dir, experiment_name, experiment_group, protocol = keys

        singles = df_sub[df_sub["method_type"] == "single"].copy()
        fusions = df_sub[df_sub["method_type"] == "fusion"].copy()
        oracle = df_sub[df_sub["method_type"] == "oracle"].copy()

        if singles.empty:
            continue

        best_single = singles.sort_values("mAP", ascending=False).iloc[0]
        best_fusion = (
            fusions.sort_values("mAP", ascending=False).iloc[0]
            if not fusions.empty
            else None
        )
        oracle_row = oracle.iloc[0] if not oracle.empty else None

        row = {
            "run_name": run_name,
            "run_dir": run_dir,
            "experiment_name": experiment_name,
            "experiment_group": experiment_group,
            "gallery_protocol": protocol,
            "n_single_models": len(singles),
            "n_fusions": len(fusions),
            "best_single_name": best_single["method_name"],
            "best_single_mAP": float(best_single["mAP"]),
            "best_single_rank1": float(best_single["rank1"]),
        }

        if best_fusion is not None:
            row.update(
                {
                    "best_fusion_name": best_fusion["method_name"],
                    "best_fusion_mAP": float(best_fusion["mAP"]),
                    "best_fusion_rank1": float(best_fusion["rank1"]),
                    "fusion_gain_vs_best_single_mAP": float(
                        best_fusion["mAP"] - best_single["mAP"]
                    ),
                    "fusion_gain_vs_best_single_rank1": float(
                        best_fusion["rank1"] - best_single["rank1"]
                    ),
                }
            )
        else:
            row.update(
                {
                    "best_fusion_name": None,
                    "best_fusion_mAP": np.nan,
                    "best_fusion_rank1": np.nan,
                    "fusion_gain_vs_best_single_mAP": np.nan,
                    "fusion_gain_vs_best_single_rank1": np.nan,
                }
            )

        if oracle_row is not None:
            row.update(
                {
                    "oracle_mAP": float(oracle_row["mAP"]),
                    "oracle_rank1": float(oracle_row["rank1"]),
                    "oracle_gap_vs_best_single_mAP": float(
                        oracle_row["mAP"] - best_single["mAP"]
                    ),
                    "oracle_gap_vs_best_fusion_mAP": (
                        float(oracle_row["mAP"] - best_fusion["mAP"])
                        if best_fusion is not None
                        else np.nan
                    ),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["gallery_protocol", "best_fusion_mAP", "best_single_mAP"],
        ascending=[True, False, False],
    )


def build_global_method_leaderboard_df(
    all_results_long_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate method performance across runs for coarse comparison."""
    grouped = (
        all_results_long_df.groupby(
            ["gallery_protocol", "method_type", "method_name"],
            dropna=False,
        )
        .agg(
            runs=("experiment_name", "nunique"),
            mean_mAP=("mAP", "mean"),
            median_mAP=("mAP", "median"),
            max_mAP=("mAP", "max"),
            mean_rank1=("rank1", "mean"),
            max_rank1=("rank1", "max"),
        )
        .reset_index()
    )
    return grouped.sort_values(["gallery_protocol", "mean_mAP"], ascending=[True, False])


def build_error_overlap_df(
    run_dir: Path,
    protocol: str,
    single_names: list[str],
) -> pd.DataFrame:
    """Compute pairwise rank-1 overlap statistics between single models."""
    df = load_per_query_comparison(run_dir, protocol)
    rows = []

    for name_a, name_b in combinations(single_names, 2):
        col_a = f"rank1__{name_a}"
        col_b = f"rank1__{name_b}"

        if col_a not in df.columns or col_b not in df.columns:
            continue

        a = df[col_a].fillna(False).astype(bool)
        b = df[col_b].fillna(False).astype(bool)
        n = len(df)

        both_correct = int((a & b).sum())
        only_a = int((a & ~b).sum())
        only_b = int((~a & b).sum())
        both_wrong = int((~a & ~b).sum())

        rows.append(
            {
                "run_name": run_dir.name,
                "gallery_protocol": protocol,
                "model_a": name_a,
                "model_b": name_b,
                "n_queries": n,
                "both_correct": both_correct,
                "only_a_correct": only_a,
                "only_b_correct": only_b,
                "both_wrong": both_wrong,
                "fraction_both_correct": both_correct / n if n else np.nan,
                "fraction_only_a_correct": only_a / n if n else np.nan,
                "fraction_only_b_correct": only_b / n if n else np.nan,
                "fraction_both_wrong": both_wrong / n if n else np.nan,
                "fraction_disagree": (only_a + only_b) / n if n else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_all_error_overlaps_df(all_results_long_df: pd.DataFrame) -> pd.DataFrame:
    """Build pairwise single-model error-overlap table across all runs."""
    rows = []
    group_cols = ["run_name", "run_dir", "gallery_protocol"]

    for _, df_sub in all_results_long_df.groupby(group_cols, dropna=False):
        run_dir = Path(df_sub["run_dir"].iloc[0])
        protocol = df_sub["gallery_protocol"].iloc[0]
        singles, _ = get_single_and_fusion_methods(df_sub)

        overlap_df = build_error_overlap_df(run_dir, protocol, singles)
        if not overlap_df.empty:
            rows.append(overlap_df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["gallery_protocol", "fraction_disagree"], ascending=[True, False])


def build_per_identity_gain_df_for_best_fusion(
    run_dir: Path,
    protocol: str,
    best_single_name: str,
    best_fusion_name: str,
) -> pd.DataFrame:
    """Aggregate per-identity gains for the best fusion vs the best single model."""
    df = load_per_query_comparison(run_dir, protocol)

    required_cols = [
        "query_label",
        f"ap__{best_single_name}",
        f"rank1__{best_single_name}",
        f"ap__{best_fusion_name}",
        f"rank1__{best_fusion_name}",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {run_dir.name} / {protocol}: {missing}")

    grouped = (
        df.groupby("query_label", dropna=False)
        .agg(
            queries=("query_label", "size"),
            best_single_ap_mean=(f"ap__{best_single_name}", "mean"),
            best_fusion_ap_mean=(f"ap__{best_fusion_name}", "mean"),
            best_single_rank1_mean=(f"rank1__{best_single_name}", "mean"),
            best_fusion_rank1_mean=(f"rank1__{best_fusion_name}", "mean"),
        )
        .reset_index()
    )

    grouped["delta_ap"] = grouped["best_fusion_ap_mean"] - grouped["best_single_ap_mean"]
    grouped["delta_rank1"] = (
        grouped["best_fusion_rank1_mean"] - grouped["best_single_rank1_mean"]
    )
    grouped["run_name"] = run_dir.name
    grouped["gallery_protocol"] = protocol
    grouped["best_single_name"] = best_single_name
    grouped["best_fusion_name"] = best_fusion_name

    return grouped.sort_values("delta_ap", ascending=False)


def build_all_per_identity_gains_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-identity gain tables for the best fusion in each run and protocol."""
    rows = []

    for _, row in summary_df.iterrows():
        best_fusion_name = row["best_fusion_name"]
        if pd.isna(best_fusion_name):
            continue

        df_gain = build_per_identity_gain_df_for_best_fusion(
            run_dir=Path(row["run_dir"]),
            protocol=row["gallery_protocol"],
            best_single_name=row["best_single_name"],
            best_fusion_name=best_fusion_name,
        )
        rows.append(df_gain)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["gallery_protocol", "delta_ap"], ascending=[True, False])


def build_interesting_cases_df(
    run_dir: Path,
    protocol: str,
    best_single_name: str,
    best_fusion_name: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Extract qualitative cases for the best fusion vs the best single model."""
    df = load_per_query_comparison(run_dir, protocol).copy()

    single_ap_col = f"ap__{best_single_name}"
    single_rank1_col = f"rank1__{best_single_name}"
    fusion_ap_col = f"ap__{best_fusion_name}"
    fusion_rank1_col = f"rank1__{best_fusion_name}"
    fusion_first_pos_col = f"first_pos_rank__{best_fusion_name}"
    fusion_top1_label_col = f"top1_label__{best_fusion_name}"
    single_top1_label_col = f"top1_label__{best_single_name}"

    needed = [single_ap_col, single_rank1_col, fusion_ap_col, fusion_rank1_col, "oracle_ap", "best_model", "best_model_ap"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {run_dir.name} / {protocol}: {missing}")

    df["delta_ap__fusion_vs_best_single"] = df[fusion_ap_col] - df[single_ap_col]
    df["delta_rank1__fusion_vs_best_single"] = (
        df[fusion_rank1_col].astype(float) - df[single_rank1_col].astype(float)
    )
    df["oracle_gap__fusion"] = df["oracle_ap"] - df[fusion_ap_col]

    keep_cols = [
        "query_idx",
        "query_label",
        "best_model",
        "best_model_ap",
        "oracle_ap",
        single_ap_col,
        single_rank1_col,
        fusion_ap_col,
        fusion_rank1_col,
        "delta_ap__fusion_vs_best_single",
        "delta_rank1__fusion_vs_best_single",
        "oracle_gap__fusion",
    ]
    optional_cols = [
        single_top1_label_col,
        fusion_top1_label_col,
        fusion_first_pos_col,
    ]
    keep_cols = [c for c in keep_cols + optional_cols if c in df.columns]

    top_gains = df.sort_values("delta_ap__fusion_vs_best_single", ascending=False).head(top_n).copy()
    top_gains["case_type"] = "top_gain"

    top_losses = df.sort_values("delta_ap__fusion_vs_best_single", ascending=True).head(top_n).copy()
    top_losses["case_type"] = "top_loss"

    disagreement_cases = (
        df.loc[df["best_model"] != best_single_name]
        .sort_values("delta_ap__fusion_vs_best_single", ascending=False)
        .head(top_n)
        .copy()
    )
    disagreement_cases["case_type"] = "model_disagreement_gain"

    out = pd.concat(
        [
            top_gains[keep_cols + ["case_type"]],
            top_losses[keep_cols + ["case_type"]],
            disagreement_cases[keep_cols + ["case_type"]],
        ],
        ignore_index=True,
    )

    out["run_name"] = run_dir.name
    out["gallery_protocol"] = protocol
    out["best_single_name"] = best_single_name
    out["best_fusion_name"] = best_fusion_name

    front_cols = [
        "run_name",
        "gallery_protocol",
        "case_type",
        "best_single_name",
        "best_fusion_name",
    ]
    other_cols = [c for c in out.columns if c not in front_cols]
    return out[front_cols + other_cols]


def build_all_interesting_cases_df(summary_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Build qualitative interesting-case tables for the best fusion in each run and protocol."""
    rows = []

    for _, row in summary_df.iterrows():
        best_fusion_name = row["best_fusion_name"]
        if pd.isna(best_fusion_name):
            continue

        df_cases = build_interesting_cases_df(
            run_dir=Path(row["run_dir"]),
            protocol=row["gallery_protocol"],
            best_single_name=row["best_single_name"],
            best_fusion_name=best_fusion_name,
            top_n=top_n,
        )
        rows.append(df_cases)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(
        ["gallery_protocol", "case_type", "delta_ap__fusion_vs_best_single"],
        ascending=[True, True, False],
    )


def run(
    config: dict,
    root_dir: Path,
    run_dir: Path,
    save_dir: Path,
) -> None:
    """Run sweep-level analysis across all ensemble runs in one experiment group."""
    del config, run_dir

    ensure_dir(save_dir)

    run_dirs = find_run_dirs(root_dir)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run dirs with results_long_all_protocols.csv found in {root_dir}"
        )

    all_results = [load_results_long(rd) for rd in run_dirs]
    all_results_long_df = pd.concat(all_results, ignore_index=True)
    all_results_long_df.to_csv(save_dir / "results_long_all_runs.csv", index=False)

    summary_df = build_run_protocol_summary_df(all_results_long_df)
    summary_df.to_csv(save_dir / "run_protocol_summary.csv", index=False)

    leaderboard_df = build_global_method_leaderboard_df(all_results_long_df)
    leaderboard_df.to_csv(save_dir / "global_method_leaderboard.csv", index=False)

    overlap_df = build_all_error_overlaps_df(all_results_long_df)
    if not overlap_df.empty:
        overlap_df.to_csv(save_dir / "single_model_error_overlaps.csv", index=False)

    per_identity_gain_df = build_all_per_identity_gains_df(summary_df)
    if not per_identity_gain_df.empty:
        per_identity_gain_df.to_csv(
            save_dir / "best_fusion_per_identity_gains.csv",
            index=False,
        )

    interesting_cases_df = build_all_interesting_cases_df(summary_df, top_n=10)
    if not interesting_cases_df.empty:
        interesting_cases_df.to_csv(
            save_dir / "interesting_cases_best_fusion.csv",
            index=False,
        )

    print("\nSaved:")
    print(save_dir / "results_long_all_runs.csv")
    print(save_dir / "run_protocol_summary.csv")
    print(save_dir / "global_method_leaderboard.csv")
    if not overlap_df.empty:
        print(save_dir / "single_model_error_overlaps.csv")
    if not per_identity_gain_df.empty:
        print(save_dir / "best_fusion_per_identity_gains.csv")
    if not interesting_cases_df.empty:
        print(save_dir / "interesting_cases_best_fusion.csv")

    create_ensemble_plots(
        summary_df=summary_df,
        leaderboard_df=leaderboard_df,
        overlap_df=overlap_df,
        per_identity_gain_df=per_identity_gain_df,
        interesting_cases_df=interesting_cases_df,
        save_dir=save_dir,
    )

    print("\nTop run/protocol summaries:")
    cols = [
        "run_name",
        "gallery_protocol",
        "best_single_name",
        "best_single_mAP",
        "best_fusion_name",
        "best_fusion_mAP",
        "fusion_gain_vs_best_single_mAP",
        "oracle_gap_vs_best_fusion_mAP",
    ]
    print(
        summary_df[cols]
        .sort_values(["gallery_protocol", "best_fusion_mAP"], ascending=[True, False])
        .head(20)
        .to_string(index=False)
    )