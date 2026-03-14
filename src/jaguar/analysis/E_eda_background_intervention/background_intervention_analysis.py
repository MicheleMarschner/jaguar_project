from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from jaguar.config import PATHS
from jaguar.utils.utils import save_fig



def load_background_results(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load aggregate and per-query background-reliance outputs from one run folder.
    """
    summary_df = pd.read_parquet(run_dir / "background_summary_all.parquet")
    per_query_delta_df = pd.read_parquet(run_dir / "background_per_query_delta_vs_original.parquet")
    return summary_df, per_query_delta_df


# -----------------------------
# Tables
# -----------------------------

def build_background_main_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the main compact comparison table
    """
    cols = [
        "setting",
        "mAP",
        "rank1",
        "delta_mAP_vs_original",
        "delta_rank1_vs_original",
    ]
    table = summary_df[cols].copy()

    table = table.rename(
        columns={
            "rank1": "Rank-1",
            "delta_mAP_vs_original": "ΔmAP vs original",
            "delta_rank1_vs_original": "ΔRank-1 vs original",
        }
    )

    return table.sort_values("setting").reset_index(drop=True)


def build_background_per_query_diagnostics(per_query_delta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate concise per-query diagnostics per setting.
    """
    df = per_query_delta_df.copy()
    df = df[df["setting"] != "original"].copy()

    diag = (
        df.groupby("setting", as_index=False)
        .agg(
            mean_delta_ap_vs_original=("delta_ap_vs_original", "mean"),
            median_delta_ap_vs_original=("delta_ap_vs_original", "median"),
            rank1_flip_rate_vs_original=("rank1_flip_vs_original", "mean"),
            mean_delta_first_pos_rank_vs_original=("delta_first_pos_rank_vs_original", "mean"),
        )
    )

    return diag.sort_values("setting").reset_index(drop=True)


# -----------------------------
# Plots
# -----------------------------

def plot_background_main_metrics(summary_df: pd.DataFrame, save_path: str | Path) -> None:
    """
    Plot grouped bars for mAP and Rank-1 by setting.
    """
    plot_df = summary_df[["setting", "mAP", "rank1"]].copy()
    plot_df = plot_df.rename(columns={"rank1": "Rank-1"})
    plot_df = plot_df.melt(id_vars="setting", var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=plot_df, x="setting", y="value", hue="metric", ax=ax)

    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Background manipulation: aggregate retrieval performance")
    ax.set_ylim(0, min(1.0, max(0.05, plot_df["value"].max() * 1.05)))

    save_fig(fig, save_path)


def plot_background_deltas(summary_df: pd.DataFrame, save_path: str | Path) -> None:
    """
    Plot grouped bars for ΔmAP and ΔRank-1 vs original by setting.
    """
    plot_df = summary_df[
        ["setting", "delta_mAP_vs_original", "delta_rank1_vs_original"]
    ].copy()
    plot_df = plot_df[plot_df["setting"] != "original"].copy()
    plot_df = plot_df.rename(
        columns={
            "delta_mAP_vs_original": "ΔmAP vs original",
            "delta_rank1_vs_original": "ΔRank-1 vs original",
        }
    )
    plot_df = plot_df.melt(id_vars="setting", var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=plot_df, x="setting", y="value", hue="metric", ax=ax)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Delta")
    ax.set_title("Performance drop relative to original queries")

    save_fig(fig, save_path)


def plot_background_delta_ap_boxplot(per_query_delta_df: pd.DataFrame, save_path: str | Path) -> None:
    """
    Plot per-query AP change vs original as a boxplot by setting.
    """
    plot_df = per_query_delta_df.copy()
    plot_df = plot_df[plot_df["setting"] != "original"].copy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=plot_df, x="setting", y="delta_ap_vs_original", ax=ax)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("ΔAP vs original")
    ax.set_title("Per-query AP degradation under background manipulation")

    save_fig(fig, save_path)


# -----------------------------
# Sentences for README/report
# -----------------------------

def build_rank1_flip_sentence(per_query_diag_df: pd.DataFrame) -> str:
    """
    Create one short sentence summarizing rank-1 flip rates.
    """
    if per_query_diag_df.empty:
        return "Rank-1 flip rate could not be computed."

    best_row = per_query_diag_df.sort_values(
        "rank1_flip_rate_vs_original", ascending=False
    ).iloc[0]

    return (
        f"Rank-1 predictions were moderately stable overall, with the largest flip rate "
        f"under {best_row['setting']} ({best_row['rank1_flip_rate_vs_original']:.1%} of queries)."
    )


def run_background_analysis(run_dir: str | Path) -> dict[str, pd.DataFrame | str]:
    """
    Build all requested tables, plots, and short report sentences for one run folder.
    """
    summary_df, per_query_delta_df = load_background_results(run_dir)

    main_table = build_background_main_table(summary_df)
    per_query_diag = build_background_per_query_diagnostics(per_query_delta_df)

    main_table.to_csv(run_dir / "background_main_table.csv", index=False)
    per_query_diag.to_csv(run_dir / "background_per_query_diagnostics.csv", index=False)

    plot_background_main_metrics(summary_df, run_dir / "plot_background_main_metrics.png")
    plot_background_deltas(summary_df, run_dir / "plot_background_deltas.png")
    plot_background_delta_ap_boxplot(
        per_query_delta_df, run_dir / "plot_background_delta_ap_boxplot.png"
    )

    rank1_flip_sentence = build_rank1_flip_sentence(per_query_diag)

    return {
        "main_table": main_table,
        "per_query_diagnostics": per_query_diag,
        "rank1_flip_sentence": rank1_flip_sentence,
    }


if __name__ == "__main__":
    
    experiment_group = "eda_background"
    run_dir = PATHS.runs / experiment_group
    results = run_background_analysis()
    print(results["main_table"])
    print(results["per_query_diagnostics"])
    print(results["rank1_flip_sentence"])
    print(results["limitations_sentence"])