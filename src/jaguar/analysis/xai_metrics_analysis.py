import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from jaguar.utils.utils import ensure_dir
from jaguar.utils.utils_xai import resolve_vec_path


sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def load_xai_metric_vectors(
    run_root: Path,
    summary_filename: str,
    slice_col: str,
    metric_specs: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Generic long-format XAI metric loader.

    Output columns:
    - run_id
    - explainer
    - slice_name
    - slice_kind
    - metric
    - sample_i
    - value
    """
    out = []

    for summary_csv in run_root.rglob(summary_filename):
        metrics_dir = summary_csv.parent
        run_dir = metrics_dir.parent
        summary = pd.read_csv(summary_csv)

        for _, row in summary.iterrows():
            explainer = row["explainer"]
            slice_name = row.get(slice_col, "all")

            for metric_name, vec_col in metric_specs:
                vec_path = resolve_vec_path(row[vec_col], metrics_dir)
                v = np.load(vec_path)

                for i, val in enumerate(v):
                    out.append({
                        "run_id": run_dir.name,
                        "explainer": explainer,
                        "slice_name": slice_name,
                        "slice_kind": slice_col,
                        "metric": metric_name,
                        "sample_i": int(i),
                        "value": float(val),
                    })

    return pd.DataFrame(out)


def build_xai_main_table(
    df_vec: pd.DataFrame,
    id_cols: list[str] | None = None,
    metric_order: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Generic mean summary table from long-format vectors.
    """
    if id_cols is None:
        id_cols = ["run_id", "explainer", "slice_name"]

    if metric_order is None:
        metric_order = ["sanity", "faith_topk", "faith_random", "faith_gap"]

    if rename_map is None:
        rename_map = {
            "sanity": "sanity_mean",
            "faith_topk": "faith_topk_mean",
            "faith_random": "faith_random_mean",
            "faith_gap": "faith_gap_mean",
        }

    sub = df_vec[df_vec["metric"].isin(metric_order)].copy()

    summary = (
        sub.groupby(id_cols + ["metric"])["value"]
        .mean()
        .reset_index()
        .pivot(index=id_cols, columns="metric", values="value")
        .reset_index()
    )

    existing = {k: v for k, v in rename_map.items() if k in summary.columns}
    summary = summary.rename(columns=existing)

    return summary.sort_values(id_cols).reset_index(drop=True)


def save_xai_main_table(
    df_vec: pd.DataFrame,
    save_dir: Path,
    filename: str,
    id_cols: list[str] | None = None,
    metric_order: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
) -> Path:
    table = build_xai_main_table(
        df_vec=df_vec,
        id_cols=id_cols,
        metric_order=metric_order,
        rename_map=rename_map,
    )
    out = save_dir / filename
    table.to_csv(out, index=False)
    return out


# ============================================================
# Metric Distributions
# ============================================================


def plot_faithfulness_barplot(
    df_vec: pd.DataFrame,
    save_path: Path,
    x_col: str = "slice_name",
    row_col: str = "explainer",
    col_col: str = "run_id",
    filename: str = "faithfulness_barplot__topk_vs_random.png",
    y_label: str = "Mean score",
    title: str = "Faithfulness: salient vs random masking",
) -> None:
    """
    Generic top-k vs random faithfulness bar plot.
    Main report plot:
    mean deletion AUC for salient vs random masking by explainer and pair_type.
    Lower is better.
    """
    sub = df_vec[df_vec["metric"].isin(["faith_topk", "faith_random"])].copy()
    if sub.empty:
        return

    sub["mask_type"] = sub["metric"].map({
        "faith_topk": "topk_salient",
        "faith_random": "random",
    })

    agg = (
        sub.groupby([col_col, row_col, x_col, "mask_type"], as_index=False)["value"]
        .mean()
    )

    g = sns.catplot(
        data=agg,
        x=x_col,
        y="value",
        hue="mask_type",
        col=col_col,
        row=row_col,
        kind="bar",
        sharey=False,
        height=4,
        aspect=1.0,
    )
    g.set_axis_labels(x_col, y_label)
    g.set_titles("{row_name} | {col_name}")
    g.fig.suptitle(title, y=1.02)
    g.savefig(save_path / filename, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def plot_faithfulness_gap_distribution(
    df_vec: pd.DataFrame,
    save_path: Path,
    x_col: str = "slice_name",
    hue_col: str = "explainer",
    col_col: str = "run_id",
    filename: str = "faithfulness_gap_distribution.png",
    y_label: str = "Faithfulness gap",
    title: str = "Faithfulness gap",
) -> None:
    """
    Distribution plot for faithfulness gap (random - salient).
    Positive values mean salient masking hurts more than random masking.
    """
    sub = df_vec[df_vec["metric"] == "faith_gap"].copy()
    if sub.empty:
        return

    g = sns.catplot(
        data=sub,
        x=x_col,
        y="value",
        hue=hue_col,
        col=col_col,
        kind="box",
        sharey=False,
        height=4,
        aspect=1.0,
    )
    for ax in g.axes.flat:
        ax.axhline(0.0, color="black", linewidth=1)
    g.set_axis_labels(x_col, y_label)
    g.set_titles("{col_name}")
    g.fig.suptitle(title, y=1.02)
    g.savefig(save_path / filename, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def plot_metric_distributions_by_model(
    df_vec: pd.DataFrame,
    save_path: Path,
    metrics: list[tuple[str, str]],
    x_col: str = "explainer",
    hue_col: str = "slice_name",
    col_col: str = "run_id",
) -> None:
    """
    RQ3 (GradCAM vs IG) + RQ4 (sanity/faithfulness compliance) (recommended main plot).

    Purpose: Boxplots of per-sample metric distributions with separate panels per model.
    Output: One plot per metric, faceted by model; within each panel compares explainer × pair_type.
    """
    for metric_name, title in metrics:
        sub = df_vec[df_vec["metric"] == metric_name].dropna(subset=["value"])
        if sub.empty:
            continue

        g = sns.catplot(
            data=sub,
            x=x_col,
            y="value",
            hue=hue_col,
            col=col_col,
            kind="box",
            height=4,
            aspect=0.9,
            sharey=False,
        )
        g.fig.suptitle(title, y=1.05)
        g.savefig(
            save_path / f"boxplot_{metric_name}__by_model.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(g.fig)



def plot_metric_distributions_by_explainer(df_vec, save_path):
    """
    RQ3 (GradCAM vs IG) + RQ4 (sanity/faithfulness compliance).

    Purpose: Boxplots of per-sample metric distributions, grouped by explainer and pair_type,
    pooled across models (high-level overview only).
    Output: One plot per metric (faith/sanity/complexity).
    Why: Quickly answers “Do IG and GradCAM differ, and does that depend on pair_type?”
    """
    metrics = [
        ("faith", "Faithfulness (Deletion AUC) [Lower is Better]"),
        ("sanity", "Sanity Check (Spearman) [Lower is Better]"),
        ("complexity", "Sparseness [Higher is Better]"),
    ]

    for metric_name, title in metrics:
        sub = df_vec[df_vec["metric"] == metric_name].dropna(subset=["value"]).copy()
        if sub.empty:
            print(f"[Warn] No rows for metric='{metric_name}'")
            continue

        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=sub,
            x="explainer",
            y="value",
            hue="pair_type",
            palette="Set2",
            gap=.1,
        )
        plt.title(f"{title}\n(All Models Combined)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / f"boxplot_{metric_name}.png", dpi=300)
        plt.close()


# ============================================================
# Significance Tests
# ============================================================

def run_significance_tests_independent(
    df_vec: pd.DataFrame,
    save_path: Path,
    slice_col: str = "slice_name",
    model_col: str = "run_id",
    filename: str = "significance_tests_mannwhitney.csv",
) -> pd.DataFrame:
    """
    RQ3 (GradCAM vs IG) + RQ4

    Independent-samples test (Mann–Whitney U) to check whether IG and GradCAM metric distributions differ
    within each (model, pair_type, metric).
    Expects df_vec columns: model, pair_type, explainer, metric, value
    Why: Adds quantitative evidence beyond boxplots. Uses independent test even if cases are paired;
    conservative when pairing exists.
    """
    results = []

    for (model, slice_name, metric), sub in df_vec.groupby([model_col, slice_col, "metric"]):
        ig = sub[sub["explainer"] == "IG"]["value"].dropna().to_numpy()
        gc = sub[sub["explainer"] == "GradCAM"]["value"].dropna().to_numpy()

        if len(ig) < 2 or len(gc) < 2:
            continue

        u_stat, p = stats.mannwhitneyu(ig, gc, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        results.append({
            model_col: model,
            slice_col: slice_name,
            "metric": metric,
            "n_IG": int(len(ig)),
            "n_GradCAM": int(len(gc)),
            "p_value": float(p),
            "significance": sig,
            "mean_IG": float(np.mean(ig)),
            "mean_GradCAM": float(np.mean(gc)),
            "median_IG": float(np.median(ig)),
            "median_GradCAM": float(np.median(gc)),
            "u_stat": float(u_stat),
        })

    res_df = pd.DataFrame(results).sort_values(["metric", model_col, slice_col])
    res_df.to_csv(save_path / filename, index=False)
    return res_df




def run_xai_similarity_metrics_analysis(
    run_root: Path,
    save_root: Path,
) -> pd.DataFrame:
    ensure_dir(save_root)

    df_vec = load_xai_metric_vectors(
        run_root=run_root,
        summary_filename="xai_summary_metrics.csv",
        slice_col="pair_type",
        metric_specs=[
            ("sanity", "sanity_vec_path"),
            ("faith_topk", "faith_topk_vec_path"),
            ("faith_random", "faith_random_vec_path"),
            ("faith_gap", "faith_gap_vec_path"),
            ("complexity", "complexity_vec_path"),
        ],
    )

    if df_vec.empty:
        print(f"[WARN] No similarity metric vectors found in {run_root}")
        return df_vec

    save_xai_main_table(
        df_vec=df_vec,
        save_dir=save_root,
        filename="xai_similarity_main_table.csv",
        id_cols=["run_id", "explainer", "slice_name"],
        rename_map={
            "sanity": "sanity_mean",
            "faith_topk": "faith_topk_mean",
            "faith_random": "faith_random_mean",
            "faith_gap": "faith_gap_mean",
            "complexity": "complexity_mean",
        },
    )

    plot_metric_distributions_by_model(
        df_vec=df_vec,
        save_path=save_root,
        metrics=[
            ("faith_topk", "Faithfulness Top-k"),
            ("faith_random", "Faithfulness Random"),
            ("faith_gap", "Faithfulness Gap"),
            ("sanity", "Sanity"),
            ("complexity", "Complexity"),
        ],
    )

    plot_faithfulness_barplot(
        df_vec=df_vec,
        save_path=save_root,
        x_col="slice_name",
        row_col="explainer",
        col_col="run_id",
        filename="faithfulness_barplot__topk_vs_random.png",
        y_label="Mean deletion AUC",
        title="Faithfulness: salient vs random masking",
    )

    plot_faithfulness_gap_distribution(
        df_vec=df_vec,
        save_path=save_root,
        x_col="slice_name",
        hue_col="explainer",
        col_col="run_id",
        filename="faithfulness_gap_distribution__by_model.png",
        y_label="Faithfulness gap",
        title="Faithfulness gap by pair type",
    )

    run_significance_tests_independent(
        df_vec=df_vec,
        save_path=save_root,
        slice_col="slice_name",
        model_col="run_id",
        filename="significance_tests_mannwhitney.csv",
    )

    return df_vec


def run_xai_class_metrics_analysis(
    run_root: Path,
    save_root: Path,
) -> pd.DataFrame:
    ensure_dir(save_root)

    df_vec = load_xai_metric_vectors(
        run_root=run_root,
        summary_filename="xai_class_summary_metrics.csv",
        slice_col="group",
        metric_specs=[
            ("sanity", "sanity_vec_path"),
            ("faith_topk", "faith_topk_vec_path"),
            ("faith_random", "faith_random_vec_path"),
            ("faith_gap", "faith_gap_vec_path"),
        ],
    )

    if df_vec.empty:
        print(f"[WARN] No class-attribution metric vectors found in {run_root}")
        return df_vec

    save_xai_main_table(
        df_vec=df_vec,
        save_dir=save_root,
        filename="xai_class_main_table.csv",
        id_cols=["run_id", "explainer", "slice_name"],
        rename_map={
            "sanity": "sanity_mean",
            "faith_topk": "mean_topk_drop",
            "faith_random": "mean_random_drop",
            "faith_gap": "faithfulness_gap",
        },
    )

    plot_metric_distributions_by_model(
        df_vec=df_vec,
        save_path=save_root,
        metrics=[
            ("faith_topk", "Faithfulness Top-k Drop"),
            ("faith_random", "Faithfulness Random Drop"),
            ("faith_gap", "Faithfulness Gap"),
            ("sanity", "Sanity"),
        ],
    )

    plot_faithfulness_barplot(
        df_vec=df_vec,
        save_path=save_root,
        x_col="slice_name",
        row_col="explainer",
        col_col="run_id",
        filename="xai_class_faithfulness_barplot.png",
        y_label="Mean score drop",
        title="Class attribution faithfulness: top-k vs random masking",
    )

    plot_faithfulness_gap_distribution(
        df_vec=df_vec,
        save_path=save_root,
        x_col="slice_name",
        hue_col="explainer",
        col_col="run_id",
        filename="xai_class_faithfulness_gap_distribution.png",
        y_label="Faithfulness gap",
        title="Class attribution faithfulness gap",
    )

    run_significance_tests_independent(
        df_vec=df_vec,
        save_path=save_root,
        slice_col="slice_name",
        model_col="run_id",
        filename="xai_class_significance_tests_mannwhitney.csv",
    )

    return df_vec