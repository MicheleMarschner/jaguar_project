import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from jaguar.analysis.xai_metrics_analysis import (
    plot_faithfulness_barplot,
    plot_faithfulness_gap_distribution,
    run_significance_tests_independent,
)
from jaguar.config import PATHS
from jaguar.utils.utils import ensure_dir
from jaguar.utils.utils_experiments import load_toml_from_path
from jaguar.utils.utils_evaluation import build_eval_context
from jaguar.utils.utils_xai import build_val_resolver, overlay_heatmap


sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


# ============================================================
# Loading
# ============================================================

def load_class_metric_vectors(
    run_root: Path,
    summary_filename: str = "xai_class_summary_metrics.csv",
) -> pd.DataFrame:
    """
    Load class-attribution XAI metric vectors into one long dataframe.

    Output columns:
    - run_id
    - explainer
    - group
    - metric
    - sample_i
    - value
    """
    rows: list[dict[str, Any]] = []

    metric_specs = [
        ("sanity", "sanity_vec_path"),
        ("faith_topk", "faith_topk_vec_path"),
        ("faith_random", "faith_random_vec_path"),
        ("faith_gap", "faith_gap_vec_path"),
        ("complexity", "complexity_vec_path"),
    ]

    for summary_csv in run_root.rglob(summary_filename):
        metrics_dir = summary_csv.parent
        run_dir = metrics_dir.parent
        summary = pd.read_csv(summary_csv)

        for _, row in summary.iterrows():
            explainer = row["explainer"]
            group = row.get("group", "all")

            for metric_name, vec_col in metric_specs:
                if vec_col not in summary.columns:
                    continue
                if pd.isna(row.get(vec_col)):
                    continue

                vec_path = Path(row[vec_col])
                if not vec_path.is_absolute():
                    vec_path = metrics_dir / vec_path
                vec_path = vec_path.resolve()

                if not vec_path.exists():
                    print(f"[WARN] Missing vector path: {vec_path}")
                    continue

                vec = np.load(vec_path)
                for i, val in enumerate(vec):
                    rows.append(
                        {
                            "run_id": run_dir.name,
                            "explainer": explainer,
                            "group": group,
                            "metric": metric_name,
                            "sample_i": int(i),
                            "value": float(val),
                        }
                    )

    return pd.DataFrame(rows)


def load_class_summary_table(
    run_root: Path,
    summary_filename: str = "xai_class_summary_metrics.csv",
) -> pd.DataFrame:
    """
    Load and concatenate all class metric summary csv files.
    """
    out = []
    for summary_csv in run_root.rglob(summary_filename):
        metrics_dir = summary_csv.parent
        run_dir = metrics_dir.parent
        df = pd.read_csv(summary_csv).copy()
        df["run_id"] = run_dir.name
        df["metrics_dir"] = str(metrics_dir)
        out.append(df)

    if not out:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)


# ============================================================
# Tables
# ============================================================

def build_class_xai_main_table(df_vec: pd.DataFrame) -> pd.DataFrame:
    """
    Main report table:
    run_id × explainer × group with mean metric values.
    """
    metric_order = ["sanity", "faith_topk", "faith_random", "faith_gap", "complexity"]
    sub = df_vec[df_vec["metric"].isin(metric_order)].copy()

    summary = (
        sub.groupby(["run_id", "explainer", "group", "metric"])["value"]
        .mean()
        .reset_index()
        .pivot(index=["run_id", "explainer", "group"], columns="metric", values="value")
        .reset_index()
    )

    rename_map = {
        "sanity": "sanity_mean",
        "faith_topk": "faith_topk_mean",
        "faith_random": "faith_random_mean",
        "faith_gap": "faith_gap_mean",
        "complexity": "complexity_mean",
    }
    summary = summary.rename(columns={k: v for k, v in rename_map.items() if k in summary.columns})
    return summary.sort_values(["run_id", "explainer", "group"]).reset_index(drop=True)


def save_class_xai_main_table(df_vec: pd.DataFrame, save_dir: Path) -> Path:
    out = save_dir / "xai_class_main_table.csv"
    table = build_class_xai_main_table(df_vec)
    table.to_csv(out, index=False)
    return out


def build_class_group_comparison_table(main_table: pd.DataFrame) -> pd.DataFrame:
    """
    Group comparison table with IG and GradCAM side by side.
    """
    metric_cols = [
        c for c in [
            "sanity_mean",
            "faith_topk_mean",
            "faith_random_mean",
            "faith_gap_mean",
            "complexity_mean",
        ]
        if c in main_table.columns
    ]

    if main_table.empty:
        return main_table.copy()

    pivot = main_table.pivot_table(
        index=["run_id", "group"],
        columns="explainer",
        values=metric_cols,
    )

    pivot.columns = [f"{metric}__{explainer}" for metric, explainer in pivot.columns]
    out = pivot.reset_index()

    for metric in metric_cols:
        ig_col = f"{metric}__IG"
        gc_col = f"{metric}__GradCAM"
        if ig_col in out.columns and gc_col in out.columns:
            out[f"{metric}__GradCAM_minus_IG"] = out[gc_col] - out[ig_col]

    return out.sort_values(["run_id", "group"]).reset_index(drop=True)


def save_class_group_comparison_table(main_table: pd.DataFrame, save_dir: Path) -> Path:
    out = save_dir / "xai_class_group_comparison_table.csv"
    table = build_class_group_comparison_table(main_table)
    table.to_csv(out, index=False)
    return out


# ============================================================
# Plots
# ============================================================

def plot_class_metric_means(
    df_vec: pd.DataFrame,
    save_dir: Path,
    filename: str = "xai_class_metric_means.png",
) -> None:
    """
    Compact mean plot for sanity / faith_gap / complexity by group and explainer.
    """
    keep_metrics = [m for m in ["sanity", "faith_gap", "complexity"] if m in df_vec["metric"].unique()]
    sub = df_vec[df_vec["metric"].isin(keep_metrics)].copy()
    if sub.empty:
        return

    agg = (
        sub.groupby(["run_id", "metric", "group", "explainer"], as_index=False)["value"]
        .mean()
    )

    g = sns.catplot(
        data=agg,
        x="group",
        y="value",
        hue="explainer",
        col="metric",
        row="run_id",
        kind="bar",
        sharey=False,
        height=4.0,
        aspect=1.1,
    )
    g.set_axis_labels("group", "mean")
    g.set_titles("{row_name} | {col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=35)
    g.fig.suptitle("Class attribution metric means by group", y=1.03)
    g.savefig(save_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def plot_class_metric_boxplots(
    df_vec: pd.DataFrame,
    save_dir: Path,
    filename: str = "xai_class_metric_boxplots.png",
) -> None:
    """
    Boxplots of per-sample metric distributions by group and explainer.
    """
    keep_metrics = [m for m in ["sanity", "faith_gap", "complexity"] if m in df_vec["metric"].unique()]
    sub = df_vec[df_vec["metric"].isin(keep_metrics)].copy()
    if sub.empty:
        return

    g = sns.catplot(
        data=sub,
        x="group",
        y="value",
        hue="explainer",
        col="metric",
        row="run_id",
        kind="box",
        sharey=False,
        height=4.0,
        aspect=1.1,
    )
    g.set_axis_labels("group", "value")
    g.set_titles("{row_name} | {col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=35)
        if "faith_gap" in ax.get_title():
            ax.axhline(0.0, color="black", linewidth=1)
    g.fig.suptitle("Class attribution metric distributions by group", y=1.03)
    g.savefig(save_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


# ============================================================
# Qualitative helpers
# ============================================================

def _tensor_to_display_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert image tensor [3,H,W] to displayable [H,W,3] in [0,1].
    Uses per-image min-max scaling for visualization only.
    """
    x = x.detach().cpu().float()
    if x.ndim == 4:
        x = x[0]
    img = x.permute(1, 2, 0).numpy()
    img = img - img.min()
    denom = img.max()
    if denom > 0:
        img = img / denom
    return np.clip(img, 0.0, 1.0)


def build_metric_lookup(df_vec: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long metric vectors to one row per sample_i.
    Keys:
    - run_id
    - explainer
    - group
    - sample_i
    """
    if df_vec.empty:
        return pd.DataFrame()

    wide = (
        df_vec.pivot_table(
            index=["run_id", "explainer", "group", "sample_i"],
            columns="metric",
            values="value",
        )
        .reset_index()
    )
    return wide


def load_class_artifact(
    run_root: Path,
    explainer: str,
    group: str,
) -> dict:
    """
    Load Stage-2 class saliency artifact for one explainer/group.
    """
    path = run_root / "explanations" / explainer / f"sal__{group}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing class artifact: {path}")
    return torch.load(path, map_location="cpu")


def select_representative_samples(
    metric_lookup: pd.DataFrame,
    group: str,
    selector_explainer: str = "IG",
    top_k: int = 3,
    bottom_k: int = 3,
) -> pd.DataFrame:
    """
    Select representative samples for one group using faith_gap.
    """
    sub = metric_lookup[
        (metric_lookup["group"] == group) &
        (metric_lookup["explainer"] == selector_explainer)
    ].copy()

    if sub.empty:
        return sub

    if "faith_gap" not in sub.columns:
        return sub.head(0)

    sub = sub.sort_values("faith_gap", ascending=False).reset_index(drop=True)

    top = sub.head(top_k).copy()
    top["selection"] = "top_faith_gap"

    bottom = sub.tail(bottom_k).copy()
    bottom["selection"] = "bottom_faith_gap"

    chosen = pd.concat([top, bottom], ignore_index=True)
    chosen = chosen.drop_duplicates(subset=["sample_i"]).reset_index(drop=True)
    return chosen


def save_qualitative_grid_for_group(
    run_root: Path,
    ctx,
    resolve_sample,
    group: str,
    metric_lookup: pd.DataFrame,
    save_dir: Path,
    explainers: list[str] | None = None,
    selector_explainer: str = "IG",
    top_k: int = 3,
    bottom_k: int = 3,
) -> Path | None:
    """
    Save a qualitative grid:
    rows = selected samples
    cols = original + one overlay per explainer
    """
    if explainers is None:
        explainers = ["IG", "GradCAM"]

    chosen = select_representative_samples(
        metric_lookup=metric_lookup,
        group=group,
        selector_explainer=selector_explainer,
        top_k=top_k,
        bottom_k=bottom_k,
    )
    if chosen.empty:
        print(f"[WARN] No qualitative samples selected for group='{group}'")
        return None

    artifacts = {}
    for explainer in explainers:
        try:
            artifacts[explainer] = load_class_artifact(run_root, explainer, group)
        except FileNotFoundError:
            continue

    if not artifacts:
        print(f"[WARN] No artifacts found for qualitative grid group='{group}'")
        return None

    n_rows = len(chosen)
    n_cols = 1 + len(artifacts)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    explainer_order = [e for e in explainers if e in artifacts]

    for row_i, (_, row) in enumerate(chosen.iterrows()):
        sample_pos = int(row["sample_i"])

        selector_art = artifacts[selector_explainer]
        sample_idx = int(selector_art["sample_indices"][sample_pos].item())

        ds, local_idx, _ = resolve_sample(sample_idx)
        sample = ds[local_idx]

        img_t = sample["img"]
        img_np = _tensor_to_display_image(img_t)

        gold_idx = int(selector_art["gold_idx"][sample_pos].item())
        pred_idx = int(selector_art["pred_orig"][sample_pos].item())
        is_correct = bool(selector_art["is_correct_orig"][sample_pos].item())

        ax = axes[row_i, 0]
        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(
            f"original\n"
            f"{row['selection']} | gold={gold_idx} pred={pred_idx} "
            f"| correct={is_correct}"
        )

        for col_j, explainer in enumerate(explainer_order, start=1):
            art = artifacts[explainer]
            sal = art["saliency"][sample_pos].detach().cpu().numpy()
            if explainer == "IG":
                overlay = overlay_heatmap(
                    img_rgb=img_np,
                    saliency_2d=sal,
                    alpha=0.62,
                    percentile=97.0,
                    threshold=0.10,
                )
            else:
                overlay = overlay_heatmap(
                    img_rgb=img_np,
                    saliency_2d=sal,
                    alpha=0.45,
                    percentile=99.0,
                    threshold=0.25,
                )
            ax = axes[row_i, col_j]
            ax.imshow(overlay)
            ax.axis("off")

            faith_gap = row["faith_gap"] if "faith_gap" in row else np.nan
            sanity = row["sanity"] if "sanity" in row else np.nan
            complexity = row["complexity"] if "complexity" in row else np.nan

            ax.set_title(
                f"{explainer}\n"
                f"faith_gap={faith_gap:.3f} | sanity={sanity:.3f} | complexity={complexity:.3f}"
            )

    fig.suptitle(f"Qualitative class attribution overlays — group={group}", y=1.01)
    fig.tight_layout()

    out = save_dir / f"qualitative_grid__{group}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def save_qualitative_summary(
    run_root: Path,
    config: dict,
    save_dir: Path,
    metric_lookup: pd.DataFrame,
    groups: list[str],
    explainers: list[str],
) -> Path:
    """
    Build qualitative grids for the requested groups and save a small manifest json.
    """
    checkpoint_dir = PATHS.checkpoints / config["evaluation"]["checkpoint_dir"]
    train_config = load_toml_from_path(checkpoint_dir / "config_leaderboard_exp.toml")

    ctx = build_eval_context(
        config=config,
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        eval_val_setting="original",
    )
    resolve_sample = build_val_resolver(ctx)

    qual_dir = save_dir / "qualitative"
    ensure_dir(qual_dir)

    saved = {}
    for group in groups:
        out = save_qualitative_grid_for_group(
            run_root=run_root,
            ctx=ctx,
            resolve_sample=resolve_sample,
            group=group,
            metric_lookup=metric_lookup,
            save_dir=qual_dir,
            explainers=explainers,
            selector_explainer="IG" if "IG" in explainers else explainers[0],
            top_k=3,
            bottom_k=3,
        )
        saved[group] = None if out is None else str(out)

    manifest = {
        "groups": groups,
        "explainers": explainers,
        "saved_files": saved,
    }

    manifest_path = save_dir / "qualitative_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


# ============================================================
# Main runner
# ============================================================

def run_class_xai_analysis(
    config: dict,
    run_root: Path,
    save_root: Path,
) -> pd.DataFrame:
    """
    Run quantitative and qualitative class-attribution XAI analysis.
    """
    ensure_dir(save_root)

    df_vec = load_class_metric_vectors(run_root=run_root)
    if df_vec.empty:
        print(f"[WARN] No class-attribution metric vectors found in {run_root}")
        return df_vec

    main_table = build_class_xai_main_table(df_vec)
    main_table.to_csv(save_root / "xai_class_main_table.csv", index=False)

    group_table = build_class_group_comparison_table(main_table)
    group_table.to_csv(save_root / "xai_class_group_comparison_table.csv", index=False)

    plot_faithfulness_barplot(
        df_vec=df_vec,
        save_path=save_root,
        x_col="group",
        row_col="explainer",
        col_col="run_id",
        filename="xai_class_faithfulness_barplot.png",
        y_label="Mean score drop",
        title="Class attribution faithfulness: top-k vs random masking",
    )

    plot_faithfulness_gap_distribution(
        df_vec=df_vec,
        save_path=save_root,
        x_col="group",
        hue_col="explainer",
        col_col="run_id",
        filename="xai_class_faithfulness_gap_distribution.png",
        y_label="Faithfulness gap",
        title="Class attribution faithfulness gap",
    )

    plot_class_metric_means(
        df_vec=df_vec,
        save_dir=save_root,
        filename="xai_class_metric_means.png",
    )

    plot_class_metric_boxplots(
        df_vec=df_vec,
        save_dir=save_root,
        filename="xai_class_metric_boxplots.png",
    )

    run_significance_tests_independent(
        df_vec=df_vec,
        save_path=save_root,
        slice_col="group",
        model_col="run_id",
        filename="xai_class_significance_tests_mannwhitney.csv",
    )

    metric_lookup = build_metric_lookup(df_vec)
    if not metric_lookup.empty:
        explainers = sorted(metric_lookup["explainer"].unique().tolist())
        groups = config["xai"].get("groups", [])
        if not groups:
            groups = sorted(metric_lookup["group"].unique().tolist())

        save_qualitative_summary(
            run_root=run_root,
            config=config,
            save_dir=save_root,
            metric_lookup=metric_lookup,
            groups=list(groups),
            explainers=explainers,
        )

    return df_vec


def run(
    config: dict,
    save_dir: Path,
    root_dir: Path | None = None,
    run_dir: Path | None = None,
    exemplar_run_dir: Path | None = None,
    **kwargs,
) -> None:
    """
    Analysis entry point for class-attribution XAI metrics.
    """
    if run_dir is None:
        raise ValueError("run_dir must be provided for class_xai_analysis")

    save_root = save_dir / "class_xai_analysis"
    ensure_dir(save_root)

    run_class_xai_analysis(
        config=config,
        run_root=run_dir,
        save_root=save_root,
    )