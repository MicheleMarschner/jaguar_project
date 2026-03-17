from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_NAME_MAP = {
    "eva_miew_convnext_weighted_globalminmax_sq": "EVA + MiewID + ConvNeXt (weighted, global)",
    "eva_miew_convnext_miewheavy_globalminmax_sq": "EVA + MiewID + ConvNeXt (Miew-heavy, global)",
    "eva_miew_convnext_equal_globalminmax_sq": "EVA + MiewID + ConvNeXt (equal, global)",
    "eva_miew_convnext_equal_rowminmax_sq": "EVA + MiewID + ConvNeXt (equal, rowwise)",
    "eva_mega_convnext_evaheavy_globalminmax_sq": "EVA + Mega + ConvNeXt (EVA-heavy, global)",
    "eva_miew_dinov2base_equal_globalminmax_sq": "EVA + MiewID + DINOv2-B (equal, global)",
}

PROTOCOL_NAME_MAP = {
    "trainval_gallery": "Train+Val gallery",
    "valonly_gallery": "Val-only gallery",
}

METHOD_NAME_MAP = {
    "score_fusion": "Score fusion",
    "embedding_concat": "Embedding concatenation",
    "oracle": "Oracle upper bound",
}


RUN_NAME_MAP = {
    "eva_miew_convnext_weighted_globalminmax_sq": "EVA + MiewID + ConvNeXt\n(weighted, global)",
    "eva_miew_convnext_miewheavy_globalminmax_sq": "EVA + MiewID + ConvNeXt\n(Miew-heavy, global)",
    "eva_miew_convnext_equal_globalminmax_sq": "EVA + MiewID + ConvNeXt\n(equal, global)",
    "eva_miew_convnext_equal_rowminmax_sq": "EVA + MiewID + ConvNeXt\n(equal, rowwise)",
    "eva_mega_convnext_evaheavy_globalminmax_sq": "EVA + MegaDescriptor + ConvNeXt\n(EVA-heavy, global)",
    "eva_miew_dinov2base_equal_globalminmax_sq": "EVA + MiewID + DINOv2-Base\n(equal, global)",
}

PROTOCOL_NAME_MAP = {
    "trainval_gallery": "Train+Val gallery",
    "valonly_gallery": "Val-only gallery",
}

METHOD_NAME_MAP = {
    "score_fusion": "Score fusion",
    "embedding_concat": "Embedding concatenation",
    "oracle": "Oracle upper bound",
    "EVA-02": "EVA-02",
    "MiewID": "MiewID",
    "ConvNeXt-V2": "ConvNeXt-V2",
    "DINOv2-Base": "DINOv2-Base",
}


def _pretty_run_name(name: str) -> str:
    """Map raw run names to readable plot labels."""
    return RUN_NAME_MAP.get(str(name), str(name))


def _pretty_protocol_name(name: str) -> str:
    """Map raw protocol names to readable plot labels."""
    return PROTOCOL_NAME_MAP.get(str(name), str(name))


def _pretty_method_name(name: str) -> str:
    """Map raw method names to readable plot labels."""
    return METHOD_NAME_MAP.get(str(name), str(name))


def _pretty_metric_name(metric: str) -> str:
    """Map internal metric keys to readable plot labels."""
    return {"mAP": "mAP", "rank1": "Rank-1"}.get(metric, metric)


def _pretty_overlap_name(value_col: str) -> str:
    """Map overlap statistic keys to readable plot labels."""
    return {
        "fraction_disagree": "Single-model disagreement rate",
        "fraction_both_wrong": "Shared error rate",
    }.get(value_col, value_col)


def _sanitize_filename(text: str) -> str:
    """Convert arbitrary text into a filesystem-friendly filename fragment."""
    return (
        str(text)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
    )


def _save_current_fig(out_path: Path) -> None:
    """Save the current matplotlib figure and close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_run_level_metric(
    summary_df: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    """Plot best single, best fusion, and oracle for one metric across runs."""
    if summary_df.empty:
        return

    df = summary_df.copy()
    value_cols = [
        f"best_single_{metric}",
        f"best_fusion_{metric}",
        f"oracle_{metric}",
    ]
    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        return

    df["label"] = (
        df["run_name"].map(_pretty_run_name)
        + "\n"
        + df["gallery_protocol"].map(_pretty_protocol_name)
    )
    df = df.sort_values(
        ["gallery_protocol", f"best_fusion_{metric}"],
        ascending=[True, False],
    )

    x = np.arange(len(df))
    width = 0.26
    metric_label = _pretty_metric_name(metric)

    plt.figure(figsize=(max(12, len(df) * 1.0), 5.8))
    plt.bar(x - width, df[f"best_single_{metric}"], width=width, label="Best single")
    plt.bar(x, df[f"best_fusion_{metric}"], width=width, label="Best fusion")
    plt.bar(x + width, df[f"oracle_{metric}"], width=width, label="Oracle upper bound")

    plt.xticks(x, df["label"], rotation=35, ha="right")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label}: best single vs best fusion vs oracle")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    _save_current_fig(out_path)

def _plot_fusion_gain_metric(
    summary_df: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    """Plot fusion gain over the best single model across runs."""
    if summary_df.empty:
        return

    col = f"fusion_gain_vs_best_single_{metric}"
    if col not in summary_df.columns:
        return

    df = summary_df.copy()
    df["label"] = (
        df["run_name"].map(_pretty_run_name)
        + "\n"
        + df["gallery_protocol"].map(_pretty_protocol_name)
    )
    df = df.sort_values(["gallery_protocol", col], ascending=[True, False])

    x = np.arange(len(df))
    metric_label = _pretty_metric_name(metric)

    plt.figure(figsize=(max(12, len(df) * 0.95), 5.0))
    plt.bar(x, df[col])
    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(x, df["label"], rotation=35, ha="right")
    plt.ylabel(f"Gain in {metric_label}")
    plt.title(f"Gain of best fusion over best single ({metric_label})")
    plt.grid(axis="y", alpha=0.3)

    _save_current_fig(out_path)

def _plot_method_leaderboard(
    leaderboard_df: pd.DataFrame,
    protocol: str,
    metric: str,
    out_path: Path,
) -> None:
    """Plot mean method performance across runs for one protocol."""
    if leaderboard_df.empty:
        return

    df = leaderboard_df[leaderboard_df["gallery_protocol"] == protocol].copy()
    if df.empty:
        return

    value_col = f"mean_{metric}"
    if value_col not in df.columns:
        return

    df = df.sort_values(value_col, ascending=False)
    df["pretty_method"] = df["method_name"].map(_pretty_method_name)
    df["pretty_type"] = df["method_type"].replace(
        {
            "single": "single",
            "fusion": "fusion",
            "oracle": "oracle",
        }
    )
    labels = df["pretty_method"] + " [" + df["pretty_type"].astype(str) + "]"

    x = np.arange(len(df))
    metric_label = _pretty_metric_name(metric)
    protocol_label = _pretty_protocol_name(protocol)

    plt.figure(figsize=(max(9, len(df) * 0.9), 5.0))
    plt.bar(x, df[value_col])
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel(f"Mean {metric_label}")
    plt.title(f"Average {metric_label} across runs ({protocol_label})")
    plt.grid(axis="y", alpha=0.3)

    _save_current_fig(out_path)

def _plot_error_overlap_heatmap(
    overlap_df: pd.DataFrame,
    protocol: str,
    value_col: str,
    out_path: Path,
) -> None:
    """Plot a heatmap of pairwise single-model overlap statistics."""
    if overlap_df.empty:
        return

    df = overlap_df[overlap_df["gallery_protocol"] == protocol].copy()
    if df.empty or value_col not in df.columns:
        return

    df["pair"] = (
        df["model_a"].map(_pretty_method_name).astype(str)
        + " vs "
        + df["model_b"].map(_pretty_method_name).astype(str)
    )
    df["pretty_run_name"] = df["run_name"].map(_pretty_run_name)

    pivot = (
        df.pivot_table(
            index="pretty_run_name",
            columns="pair",
            values=value_col,
            aggfunc="mean",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    if pivot.empty:
        return

    protocol_label = _pretty_protocol_name(protocol)
    overlap_label = _pretty_overlap_name(value_col)

    plt.figure(figsize=(max(8, pivot.shape[1] * 1.4), max(4, pivot.shape[0] * 0.9)))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label=overlap_label)
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=35, ha="right")
    plt.yticks(np.arange(pivot.shape[0]), pivot.index)
    plt.title(f"{overlap_label} ({protocol_label})")

    _save_current_fig(out_path)

def _plot_identity_gains_top_bottom(
    per_identity_gain_df: pd.DataFrame,
    protocol: str,
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Plot the largest positive and negative identity-level AP gains."""
    if per_identity_gain_df.empty:
        return

    df = per_identity_gain_df[per_identity_gain_df["gallery_protocol"] == protocol].copy()
    if df.empty or "delta_ap" not in df.columns:
        return

    top_df = df.sort_values("delta_ap", ascending=False).head(top_n).copy()
    bot_df = df.sort_values("delta_ap", ascending=True).head(top_n).copy()
    plot_df = pd.concat([top_df, bot_df], ignore_index=True)

    plot_df["label"] = (
        plot_df["query_label"].astype(str)
        + " | "
        + plot_df["run_name"].map(_pretty_run_name).astype(str)
    )
    plot_df = plot_df.sort_values("delta_ap", ascending=True)

    y = np.arange(len(plot_df))
    protocol_label = _pretty_protocol_name(protocol)

    plt.figure(figsize=(12, max(6, len(plot_df) * 0.35)))
    plt.barh(y, plot_df["delta_ap"])
    plt.yticks(y, plot_df["label"])
    plt.axvline(0.0, linewidth=1.0)
    plt.xlabel("Delta AP (best fusion - best single)")
    plt.title(f"Largest positive and negative identity-level gains ({protocol_label})")
    plt.grid(axis="x", alpha=0.3)

    _save_current_fig(out_path)

def _plot_interesting_cases_scatter(
    interesting_cases_df: pd.DataFrame,
    protocol: str,
    out_path: Path,
) -> None:
    """Plot per-query best-single AP against best-fusion AP for interesting cases."""
    if interesting_cases_df.empty:
        return

    df = interesting_cases_df[interesting_cases_df["gallery_protocol"] == protocol].copy()
    if df.empty:
        return

    single_col = None
    fusion_col = None
    if "best_single_name" in df.columns and "best_fusion_name" in df.columns:
        row0 = df.iloc[0]
        single_candidate = f"ap__{row0['best_single_name']}"
        fusion_candidate = f"ap__{row0['best_fusion_name']}"
        if single_candidate in df.columns:
            single_col = single_candidate
        if fusion_candidate in df.columns:
            fusion_col = fusion_candidate

    if single_col is None or fusion_col is None:
        return

    single_label = _pretty_method_name(single_col.replace("ap__", ""))
    fusion_label = _pretty_method_name(fusion_col.replace("ap__", ""))
    protocol_label = _pretty_protocol_name(protocol)

    case_label_map = {
        "top_gain": "Top gains",
        "top_loss": "Top losses",
        "model_disagreement_gain": "Model-disagreement gains",
    }

    plt.figure(figsize=(6.8, 6.0))
    for case_type in sorted(df["case_type"].dropna().unique()):
        sub = df[df["case_type"] == case_type]
        plt.scatter(
            sub[single_col],
            sub[fusion_col],
            label=case_label_map.get(case_type, case_type),
            alpha=0.8,
        )

    lim_min = min(df[single_col].min(), df[fusion_col].min())
    lim_max = max(df[single_col].max(), df[fusion_col].max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1.0)

    plt.xlabel(f"AP of best single ({single_label})")
    plt.ylabel(f"AP of best fusion ({fusion_label})")
    plt.title(f"Interesting cases: best single vs best fusion ({protocol_label})")
    plt.legend()
    plt.grid(alpha=0.3)

    _save_current_fig(out_path)


def create_ensemble_plots(
    summary_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    per_identity_gain_df: pd.DataFrame,
    interesting_cases_df: pd.DataFrame,
    save_dir: Path,
) -> None:
    """Create all sweep-level ensemble plots from precomputed analysis tables."""
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_run_level_metric(summary_df, "mAP", plots_dir / "run_level_map.png")
    _plot_run_level_metric(summary_df, "rank1", plots_dir / "run_level_rank1.png")
    _plot_fusion_gain_metric(summary_df, "mAP", plots_dir / "fusion_gain_map.png")
    _plot_fusion_gain_metric(summary_df, "rank1", plots_dir / "fusion_gain_rank1.png")

    protocols = []
    if not summary_df.empty and "gallery_protocol" in summary_df.columns:
        protocols.extend(summary_df["gallery_protocol"].dropna().unique().tolist())
    if not overlap_df.empty and "gallery_protocol" in overlap_df.columns:
        protocols.extend(overlap_df["gallery_protocol"].dropna().unique().tolist())
    if not per_identity_gain_df.empty and "gallery_protocol" in per_identity_gain_df.columns:
        protocols.extend(per_identity_gain_df["gallery_protocol"].dropna().unique().tolist())
    if not interesting_cases_df.empty and "gallery_protocol" in interesting_cases_df.columns:
        protocols.extend(interesting_cases_df["gallery_protocol"].dropna().unique().tolist())

    protocols = sorted(set(protocols))

    for protocol in protocols:
        tag = _sanitize_filename(protocol)

        _plot_method_leaderboard(
            leaderboard_df=leaderboard_df,
            protocol=protocol,
            metric="mAP",
            out_path=plots_dir / f"method_leaderboard_map__{tag}.png",
        )
        _plot_method_leaderboard(
            leaderboard_df=leaderboard_df,
            protocol=protocol,
            metric="rank1",
            out_path=plots_dir / f"method_leaderboard_rank1__{tag}.png",
        )
        _plot_error_overlap_heatmap(
            overlap_df=overlap_df,
            protocol=protocol,
            value_col="fraction_disagree",
            out_path=plots_dir / f"error_overlap_disagree__{tag}.png",
        )
        _plot_error_overlap_heatmap(
            overlap_df=overlap_df,
            protocol=protocol,
            value_col="fraction_both_wrong",
            out_path=plots_dir / f"error_overlap_both_wrong__{tag}.png",
        )
        _plot_identity_gains_top_bottom(
            per_identity_gain_df=per_identity_gain_df,
            protocol=protocol,
            out_path=plots_dir / f"identity_gains_top_bottom__{tag}.png",
            top_n=15,
        )
        _plot_interesting_cases_scatter(
            interesting_cases_df=interesting_cases_df,
            protocol=protocol,
            out_path=plots_dir / f"interesting_cases_scatter__{tag}.png",
        )