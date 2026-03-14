from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from jaguar.config import PATHS
from jaguar.utils.utils import ensure_dir, save_parquet
from jaguar.xai.foreground_contribution import (
    summarize_bg_vs_jaguar,
    summarize_embedding_stability,
    summarize_retrieval_variant,
)


def load_runs(experiments_dir: Path) -> list[dict]:
    runs = []

    for run_dir in sorted(p for p in experiments_dir.iterdir() if p.is_dir()):
        backbone_name, head_type, bg_part = run_dir.name.split("__")
        background = bg_part.replace("bg-", "", 1)

        analysis_path = next(run_dir.glob("analysis_merged__*.parquet"))
        classification_path = next(run_dir.glob("classification_sensitivity__*.parquet"))
        config_path = next(run_dir.glob("run_config__*.json"))

        with open(config_path, "r") as f:
            config = json.load(f)

        runs.append({
            "run_name": run_dir.name,
            "backbone_name": backbone_name,
            "head_type": head_type,
            "background": background,
            "run_dir": run_dir,
            "analysis_df": pd.read_parquet(analysis_path),
            "classification_df": pd.read_parquet(classification_path),
            "config": config,
        })

    return runs


def summarize_run_analysis(analysis_df: pd.DataFrame) -> dict:
    groups = {
        "all": analysis_df,
        "orig_rank1_correct": analysis_df[analysis_df["is_rank1_orig"]].copy(),
        "orig_rank1_wrong": analysis_df[~analysis_df["is_rank1_orig"]].copy(),
    }

    retrieval = {}
    stability = {}

    for group_name, df in groups.items():
        retrieval[group_name] = {
            "bg_vs_jaguar": summarize_bg_vs_jaguar(df),
            "variants": {
                "orig": summarize_retrieval_variant(df, "orig"),
                "jaguar_only": summarize_retrieval_variant(df, "jaguar_only"),
                "bg_only": summarize_retrieval_variant(df, "bg_only"),
            },
        }
        stability[group_name] = summarize_embedding_stability(df)

    return {
        "retrieval": retrieval,
        "stability": stability,
    }


def summarize_run(run_data: dict) -> dict:
    return {
        "meta": {
            "run_name": run_data["run_name"],
            "backbone_name": run_data["backbone_name"],
            "head_type": run_data["head_type"],
            "background": run_data["background"],
        },
        **summarize_run_analysis(run_data["analysis_df"]),
    }


def flatten_dict(d: dict, parent_key: str = "", sep: str = "__") -> dict:
    out = {}

    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep))
        else:
            out[key] = v

    return out


def build_master_summary_table(runs: list[dict]) -> pd.DataFrame:
    rows = [flatten_dict(summarize_run(run_data)) for run_data in runs]
    return pd.DataFrame(rows)


def build_main_retrieval_table(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main report table for foreground-vs-background contribution.
    One row per run/background condition.
    """
    df = master_df.copy()

    out = pd.DataFrame({
        "background": df["meta__background"],
        "backbone_name": df["meta__backbone_name"],
        "head_type": df["meta__head_type"],

        "orig_rank1": df["retrieval__all__variants__orig__rank1"],
        "jaguar_only_rank1": df["retrieval__all__variants__jaguar_only__rank1"],
        "bg_only_rank1": df["retrieval__all__variants__bg_only__rank1"],

        "rank1_gap_jag_minus_bg": (
            df["retrieval__all__variants__jaguar_only__rank1"]
            - df["retrieval__all__variants__bg_only__rank1"]
        ),
        "share_bg_better_rank1": df["retrieval__all__bg_vs_jaguar__share_bg_better_rank1"],

        "orig_mean_margin": df["retrieval__all__variants__orig__mean_margin"],
        "jaguar_only_mean_margin": df["retrieval__all__variants__jaguar_only__mean_margin"],
        "bg_only_mean_margin": df["retrieval__all__variants__bg_only__mean_margin"],

        "margin_gap_jag_minus_bg": (
            df["retrieval__all__variants__jaguar_only__mean_margin"]
            - df["retrieval__all__variants__bg_only__mean_margin"]
        ),
        "share_bg_better_margin": df["retrieval__all__bg_vs_jaguar__share_bg_better_margin"],
    })

    return out.sort_values(["backbone_name", "head_type", "background"]).reset_index(drop=True)



def save_grouped_master_summaries(runs: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}

    for run_data in runs:
        group_key = f"{run_data['backbone_name']}__{run_data['head_type']}"
        grouped.setdefault(group_key, []).append(run_data)

    out = {}

    for group_key, group_runs in grouped.items():
        master_df = build_master_summary_table(group_runs)
        summary_dir = PATHS.results / f"xai/background_sensitivity_summary/{group_key}"

        ensure_dir(summary_dir)
        master_df.to_csv(summary_dir / "master_summary.csv", index=False)
        save_parquet(summary_dir / "master_summary.parquet", master_df)

        out[group_key] = {
            "master_df": master_df,
            "summary_dir": summary_dir,
        }

    return out


def save_global_master_summary(runs: list[dict]) -> pd.DataFrame:
    master_df = build_master_summary_table(runs)
    summary_dir = PATHS.results / "xai/background_sensitivity_summary/_global"

    ensure_dir(summary_dir)
    master_df.to_csv(summary_dir / "master_summary.csv", index=False)
    save_parquet(summary_dir / "master_summary.parquet", master_df)

    return master_df


def plot_absolute_performance_by_variant(
    master_df: pd.DataFrame,
    summary_dir: Path,
    group_name: str = "all",
) -> None:
    df = master_df.sort_values("meta__background").copy()

    plot_df = pd.DataFrame({
        "background": list(df["meta__background"]) * 3,
        "variant": (
            ["orig"] * len(df)
            + ["jaguar_only"] * len(df)
            + ["bg_only"] * len(df)
        ),
        "rank1": (
            list(df[f"retrieval__{group_name}__variants__orig__rank1"])
            + list(df[f"retrieval__{group_name}__variants__jaguar_only__rank1"])
            + list(df[f"retrieval__{group_name}__variants__bg_only__rank1"])
        ),
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x="background",
        y="rank1",
        hue="variant",
        hue_order=["orig", "jaguar_only", "bg_only"],
    )
    plt.ylabel("Rank-1")
    plt.xlabel("Background setting")
    plt.title(f"Absolute retrieval performance ({group_name})")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(summary_dir / f"absolute_performance_by_variant__{group_name}.png", dpi=200)
    plt.close()


def plot_background_comparison_figure(
    master_df: pd.DataFrame,
    summary_dir: Path,
    group_name: str = "all",
) -> None:
    df = master_df.sort_values("meta__background").copy()

    palette = {
        "orig": "#4C72B0",
        "jaguar_only": "#DD8452",
        "bg_only": "#55A868",
    }

    plot_df_rank1 = pd.DataFrame({
        "background": list(df["meta__background"]) * 3,
        "variant": (
            ["orig"] * len(df)
            + ["jaguar_only"] * len(df)
            + ["bg_only"] * len(df)
        ),
        "value": (
            list(df[f"retrieval__{group_name}__variants__orig__rank1"])
            + list(df[f"retrieval__{group_name}__variants__jaguar_only__rank1"])
            + list(df[f"retrieval__{group_name}__variants__bg_only__rank1"])
        ),
    })

    plot_df_margin = pd.DataFrame({
        "background": list(df["meta__background"]) * 3,
        "variant": (
            ["orig"] * len(df)
            + ["jaguar_only"] * len(df)
            + ["bg_only"] * len(df)
        ),
        "value": (
            list(df[f"retrieval__{group_name}__variants__orig__mean_margin"])
            + list(df[f"retrieval__{group_name}__variants__jaguar_only__mean_margin"])
            + list(df[f"retrieval__{group_name}__variants__bg_only__mean_margin"])
        ),
    })

    plot_df_stability = pd.DataFrame({
        "background": list(df["meta__background"]) * 2,
        "variant": (
            ["jaguar_only"] * len(df)
            + ["bg_only"] * len(df)
        ),
        "value": (
            list(df[f"stability__{group_name}__mean_stability_jaguar_only"])
            + list(df[f"stability__{group_name}__mean_stability_bg_only"])
        ),
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(
        data=plot_df_rank1,
        x="background",
        y="value",
        hue="variant",
        hue_order=["orig", "jaguar_only", "bg_only"],
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_title(f"Rank-1 ({group_name})")
    axes[0].set_xlabel("Background setting")
    axes[0].set_ylabel("Rank-1")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(
        data=plot_df_margin,
        x="background",
        y="value",
        hue="variant",
        hue_order=["orig", "jaguar_only", "bg_only"],
        palette=palette,
        ax=axes[1],
    )
    axes[1].set_title(f"Mean margin ({group_name})")
    axes[1].set_xlabel("Background setting")
    axes[1].set_ylabel("Mean margin")
    axes[1].tick_params(axis="x", rotation=45)

    sns.barplot(
        data=plot_df_stability,
        x="background",
        y="value",
        hue="variant",
        hue_order=["jaguar_only", "bg_only"],
        palette=palette,
        ax=axes[2],
    )
    axes[2].set_title(f"Mean stability ({group_name})")
    axes[2].set_xlabel("Background setting")
    axes[2].set_ylabel("Mean cosine similarity")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(summary_dir / f"background_comparison__{group_name}.png", dpi=200)
    plt.close()


def plot_gap_figure(
    master_df: pd.DataFrame,
    summary_dir: Path,
    group_name: str = "all",
) -> None:
    df = master_df.sort_values("meta__background").copy()

    gap_df = pd.DataFrame({
        "background": list(df["meta__background"]),
        "rank1_gap_jag_minus_bg": (
            df[f"retrieval__{group_name}__variants__jaguar_only__rank1"]
            - df[f"retrieval__{group_name}__variants__bg_only__rank1"]
        ),
        "margin_gap_jag_minus_bg": (
            df[f"retrieval__{group_name}__variants__jaguar_only__mean_margin"]
            - df[f"retrieval__{group_name}__variants__bg_only__mean_margin"]
        ),
        "stability_gap_jag_minus_bg": (
            df[f"stability__{group_name}__mean_stability_jaguar_only"]
            - df[f"stability__{group_name}__mean_stability_bg_only"]
        ),
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(data=gap_df, x="background", y="rank1_gap_jag_minus_bg", color="#DD8452", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title(f"Rank-1 gap ({group_name})")
    axes[0].set_xlabel("Background setting")
    axes[0].set_ylabel("Jaguar - BG")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=gap_df, x="background", y="margin_gap_jag_minus_bg", color="#DD8452", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title(f"Mean margin gap ({group_name})")
    axes[1].set_xlabel("Background setting")
    axes[1].set_ylabel("Jaguar - BG")
    axes[1].tick_params(axis="x", rotation=45)

    sns.barplot(data=gap_df, x="background", y="stability_gap_jag_minus_bg", color="#DD8452", ax=axes[2])
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title(f"Mean stability gap ({group_name})")
    axes[2].set_xlabel("Background setting")
    axes[2].set_ylabel("Jaguar - BG")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(summary_dir / f"background_gap_figure__{group_name}.png", dpi=200)
    plt.close()


def plot_bg_better_share_figure(master_df: pd.DataFrame, summary_dir: Path) -> None:
    df = master_df.sort_values("meta__background").copy()

    plot_df = pd.DataFrame({
        "background": list(df["meta__background"]) * 3,
        "metric": (
            ["share_bg_better_rank"] * len(df)
            + ["share_bg_better_margin"] * len(df)
            + ["share_bg_more_stable"] * len(df)
        ),
        "value": (
            list(df["retrieval__all__bg_vs_jaguar__share_bg_better_rank"])
            + list(df["retrieval__all__bg_vs_jaguar__share_bg_better_margin"])
            + list(df["stability__all__share_bg_more_stable"])
        ),
    })

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x="background",
        y="value",
        hue="metric",
    )
    plt.ylabel("Share")
    plt.xlabel("Background setting")
    plt.title("Share of cases where background beats jaguar")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(summary_dir / "background_better_share_figure.png", dpi=200)
    plt.close()


def plot_bg_better_rank1_share(
    master_df: pd.DataFrame,
    summary_dir: Path,
    group_name: str = "all",
) -> None:
    df = master_df.sort_values("meta__background").copy()

    plot_df = pd.DataFrame({
        "background": df["meta__background"],
        "value": df[f"retrieval__{group_name}__bg_vs_jaguar__share_bg_better_rank1"],
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="background", y="value", color="#55A868")
    plt.ylabel("Share")
    plt.xlabel("Background setting")
    plt.title(f"Share BG better than jaguar (Rank-1) ({group_name})")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(summary_dir / f"bg_better_rank1_share__{group_name}.png", dpi=200)
    plt.close()

def plot_backbone_gap_for_background(
    global_master_df: pd.DataFrame,
    summary_dir: Path,
    background: str,
    group_name: str = "all",
) -> None:
    df = global_master_df[global_master_df["meta__background"] == background].copy()
    df = df.sort_values("meta__backbone_name")

    plot_df = pd.DataFrame({
        "backbone": df["meta__backbone_name"],
        "rank1_gap_jag_minus_bg": (
            df[f"retrieval__{group_name}__variants__jaguar_only__rank1"]
            - df[f"retrieval__{group_name}__variants__bg_only__rank1"]
        ),
        "margin_gap_jag_minus_bg": (
            df[f"retrieval__{group_name}__variants__jaguar_only__mean_margin"]
            - df[f"retrieval__{group_name}__variants__bg_only__mean_margin"]
        ),
        "stability_gap_jag_minus_bg": (
            df[f"stability__{group_name}__mean_stability_jaguar_only"]
            - df[f"stability__{group_name}__mean_stability_bg_only"]
        ),
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(data=plot_df, x="backbone", y="rank1_gap_jag_minus_bg", color="#DD8452", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title(f"Rank-1 gap | bg={background} | {group_name}")
    axes[0].set_xlabel("Backbone")
    axes[0].set_ylabel("Jaguar - BG")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=plot_df, x="backbone", y="margin_gap_jag_minus_bg", color="#DD8452", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title(f"Mean margin gap | bg={background} | {group_name}")
    axes[1].set_xlabel("Backbone")
    axes[1].set_ylabel("Jaguar - BG")
    axes[1].tick_params(axis="x", rotation=45)

    sns.barplot(data=plot_df, x="backbone", y="stability_gap_jag_minus_bg", color="#DD8452", ax=axes[2])
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title(f"Mean stability gap | bg={background} | {group_name}")
    axes[2].set_xlabel("Backbone")
    axes[2].set_ylabel("Jaguar - BG")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(summary_dir / f"backbone_gap__bg-{background}__{group_name}.png", dpi=200)
    plt.close()


def has_non_nan_group(master_df: pd.DataFrame, group_name: str, prefix: str = "retrieval") -> bool:
    cols = [c for c in master_df.columns if c.startswith(f"{prefix}__{group_name}__")]
    if not cols:
        return False
    return not master_df[cols].isna().all().all()


def build_error_split_table(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact report table for:
    all / orig_rank1_correct / orig_rank1_wrong
    """
    rows = []

    for _, row in master_df.iterrows():
        meta = {
            "background": row["meta__background"],
            "backbone_name": row["meta__backbone_name"],
            "head_type": row["meta__head_type"],
        }

        for group_name in ["all", "orig_rank1_correct", "orig_rank1_wrong"]:
            rows.append({
                **meta,
                "group": group_name,
                "jaguar_only_rank1": row[f"retrieval__{group_name}__variants__jaguar_only__rank1"],
                "bg_only_rank1": row[f"retrieval__{group_name}__variants__bg_only__rank1"],
                "share_bg_better_rank1": row[f"retrieval__{group_name}__bg_vs_jaguar__share_bg_better_rank1"],
            })

    return pd.DataFrame(rows).sort_values(
        ["backbone_name", "head_type", "background", "group"]
    ).reset_index(drop=True)


def save_report_tables(master_df: pd.DataFrame, summary_dir: Path) -> dict[str, Path]:
    main_table = build_main_retrieval_table(master_df)
    error_table = build_error_split_table(master_df)

    main_csv = summary_dir / "foreground_background_main_table.csv"
    error_csv = summary_dir / "foreground_background_error_split_table.csv"

    main_table.to_csv(main_csv, index=False)
    error_table.to_csv(error_csv, index=False)

    return {
        "main_table_csv": main_csv,
        "error_split_csv": error_csv,
    }


def run_foreground_background_analysis(
    experiments_dir: Path,
) -> dict[str, Path]:
    runs = load_runs(experiments_dir)
    if not runs:
        print(f"[ANALYSIS][WARN] No background-sensitivity runs found under: {experiments_dir}")
        return {}

    grouped_outputs = save_grouped_master_summaries(runs)

    for _, obj in grouped_outputs.items():
        plot_background_comparison_figure(obj["master_df"], obj["summary_dir"], group_name="all")
        plot_background_comparison_figure(obj["master_df"], obj["summary_dir"], group_name="orig_rank1_wrong")

        plot_gap_figure(obj["master_df"], obj["summary_dir"], group_name="all")
        plot_gap_figure(obj["master_df"], obj["summary_dir"], group_name="orig_rank1_wrong")

        plot_bg_better_rank1_share(obj["master_df"], obj["summary_dir"], group_name="all")
        plot_bg_better_rank1_share(obj["master_df"], obj["summary_dir"], group_name="orig_rank1_wrong")

    global_master_df = save_global_master_summary(runs)
    save_report_tables(global_master_df, global_summary_dir)

    global_summary_dir = PATHS.results / "xai/background_sensitivity_summary/_global"
    ensure_dir(global_summary_dir)

    for background in sorted(global_master_df["meta__background"].dropna().unique()):
        plot_backbone_gap_for_background(global_master_df, global_summary_dir, background, group_name="all")
        plot_backbone_gap_for_background(global_master_df, global_summary_dir, background, group_name="orig_rank1_wrong")

    return {
        "global_master_csv": global_summary_dir / "master_summary.csv",
        "global_master_parquet": global_summary_dir / "master_summary.parquet",
        "global_main_table_csv": global_summary_dir / "foreground_background_main_table.csv",
        "global_error_split_csv": global_summary_dir / "foreground_background_error_split_table.csv",
    }