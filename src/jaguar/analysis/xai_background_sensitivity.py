"""
Plot 1 - Absolute performance by background setting:
- Frage: Wie stark bleibt die Identitätsinformation im Original, im Jaguar allein und im Hintergrund allein erhalten?
- Interpretation: jaguar_only hoch, bg_only niedrig → Modell nutzt vor allem den Jaguar
- Abstand zwischen jaguar_only und bg_only über Trainingsbedingungen vergleichen

Plot 2 - Gap plot (jaguar_only - bg_only):
- Frage: Ist der Jaguar stärker als der Hintergrund, und wie stark?
- Interpretation: positiv → Jaguar trägt mehr; nahe 0 → beides ähnlich stark; negativ → Hintergrund trägt mehr

Plot 3 - Share BG better than Jaguar:
- Frage: Wie oft schlägt der Hintergrund den Jaguar tatsächlich auf Einzelfall-Ebene?
- zeigt Anteil von Fällen, in denen bg_only besser ist als jaguar_only
- Interpretation: 0 oder sehr niedrig → Hintergrund dominiert fast nie; höherer Anteil → es gibt mehr echte bg-dominante Fälle;gut, um seltene aber wichtige Problemfälle sichtbar zu machen


Zusatz - sobald dass Modell auch mal falsch liegt:
Mit orig_rank1_correct vs orig_rank1_wrong

Nutzt das Modell den Hintergrund vor allem bei eigentlich schwierigen Fehlerfällen?

Oder auch bei Fällen, die es im Original korrekt löst?

!TODO: unbedingt überprüfen was gerade query und was gallery - es darf keine leakage sein!! Somit über split dateien als truth source arbeiten!! (sofern ich das richtig verstehe?)

"""



from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from jaguar.config import PATHS
from jaguar.utils.utils import ensure_dir, save_parquet
from jaguar.xai.xai_classification import (
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



def plot_bg_better_margin_share(master_df: pd.DataFrame, summary_dir: Path, group_name: str = "all") -> None:
    df = master_df.sort_values("meta__background").copy()

    plot_df = pd.DataFrame({
        "background": df["meta__background"],
        "value": df[f"retrieval__{group_name}__bg_vs_jaguar__share_bg_better_margin"],
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="background", y="value", color="#55A868")
    plt.ylabel("Share")
    plt.xlabel("Background setting")
    plt.title(f"Share BG better than jaguar (margin) ({group_name})")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(summary_dir / f"bg_better_margin_share__{group_name}.png", dpi=200)
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


def run_xai_background_sensitivity(
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

        plot_bg_better_margin_share(obj["master_df"], obj["summary_dir"], group_name="all")
        plot_bg_better_margin_share(obj["master_df"], obj["summary_dir"], group_name="orig_rank1_wrong")

    global_master_df = save_global_master_summary(runs)

    global_summary_dir = PATHS.results / "xai/background_sensitivity_summary/_global"
    ensure_dir(global_summary_dir)

    for background in sorted(global_master_df["meta__background"].dropna().unique()):
        plot_backbone_gap_for_background(global_master_df, global_summary_dir, background, group_name="all")
        plot_backbone_gap_for_background(global_master_df, global_summary_dir, background, group_name="orig_rank1_wrong")

    return {
        "global_master_csv": global_summary_dir / "master_summary.csv",
        "global_master_parquet": global_summary_dir / "master_summary.parquet",
    }