import pandas as pd
from jaguar.utils.utils import ensure_dir
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import re

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ============================================================
# Load and aggregate all results
# ============================================================

def resolve_vec_path(vec_path_raw: str, metrics_dir: Path) -> Path:
    """
    Resolves a metric vector path from the summary CSV.

    Supports:
    1) Absolute path that exists
    2) Absolute path that used to be under .../xai/<run>/... and is now under .../xai/similarity/<run>/...
    3) Relative path (or filename) relative to metrics_dir
    """
    if not isinstance(vec_path_raw, str) or not vec_path_raw:
        raise FileNotFoundError(f"Empty vec path: {vec_path_raw}")

    p = Path(vec_path_raw)

    # (A) Absolute path exists
    if p.is_absolute() and p.exists():
        return p

    # (B) Relative to metrics_dir
    cand = metrics_dir / p
    if cand.exists():
        return cand

    # (C) If absolute but old location, rewrite .../xai/<run>/... -> .../xai/similarity/<run>/...
    if p.is_absolute():
        parts = list(p.parts)
        try:
            xai_i = parts.index("xai")
        except ValueError:
            pass
        else:
            # if it already has similarity right after xai, nothing to rewrite
            if xai_i + 1 < len(parts) and parts[xai_i + 1] != "similarity":
                rewritten = Path(*parts[: xai_i + 1], "similarity", *parts[xai_i + 1 :])
                if rewritten.exists():
                    return rewritten

    raise FileNotFoundError(
        f"Could not resolve vec path:\n"
        f"  vec_path_raw: {vec_path_raw}\n"
        f"  metrics_dir : {metrics_dir}"
    )


_RUN_RE = re.compile(r"^(?P<model>.+)__(?P<split>.+)__n(?P<n>\d+)__seed(?P<seed>\d+)$")


def load_all_vectors(run_root: Path) -> pd.DataFrame:
    """
    Returns long dataframe with per-sample metric values:
    model, split, n_samples, seed, run_id, explainer, pair_type, metric, sample_i, value
    """
    out = []

    for summary_csv in run_root.rglob("metrics/xai_summary_metrics.csv"):
        metrics_dir = summary_csv.parent          # .../metrics
        run_dir = metrics_dir.parent              # .../<model>__<split>__n..__seed..
        m = _RUN_RE.match(run_dir.name)
        if m is None:
            continue

        model = m.group("model")
        split = m.group("split")
        n_samples = int(m.group("n"))
        seed = int(m.group("seed"))

        summary = pd.read_csv(summary_csv)

        for _, row in summary.iterrows():
            explainer = row["explainer"]
            pair_type = row["pair_type"]

            for metric_name, vec_col in [
                ("sanity", "sanity_vec_path"),
                ("faith", "faith_vec_path"),
                ("complexity", "complexity_vec_path"),
            ]:
                vec_path_raw = row.get(vec_col, "")
                vec_path = resolve_vec_path(vec_path_raw, metrics_dir=metrics_dir)

                v = np.load(vec_path)  # shape [N]

                for i, val in enumerate(v):
                    out.append({
                        "model": model,
                        "split": split,
                        "n_samples": n_samples,
                        "seed": seed,
                        "run_id": run_dir.name,
                        "explainer": explainer,
                        "pair_type": pair_type,
                        "metric": metric_name,
                        "sample_i": int(i),
                        "value": float(val),
                        "vec_path": str(vec_path),
                    })

    return pd.DataFrame(out)

# ============================================================
# Generate summary table
# ============================================================

def print_summary_table(df_vec: pd.DataFrame, save_path: Path):
    """
    RQ3 (GradCAM vs IG) + RQ4 (sanity/faithfulness compliance).

    Purpose: Compares metrics across model × explainer × pair_type
    Produce a compact “Mean ± Std” summary table from df_vec (per-sample metrics).
    Why: complements boxplots (shows exact values).
    """
    print("\n=== Summary Statistics (Mean ± Std) ===")

    summary = (
        df_vec
        .groupby(["model","explainer","pair_type","metric"])["value"]
        .agg(mean="mean", std="std")
        .assign(formatted=lambda d: d["mean"].map(lambda x: f"{x:.3f}") + " ± " + d["std"].map(lambda x: f"{x:.3f}"))
        .reset_index()
        .pivot(index=["model","explainer","pair_type"], columns="metric", values="formatted")
    )

    '''
    summary = pd.pivot_table(
        df_vec,
        index=["model", "explainer", "pair_type"],
        columns="metric",
        values="value",
        aggfunc=["mean", "std", "count"],
    )

    # pivot_table gives columns (stat, metric). Flip to (metric, stat)
    summary = summary.swaplevel(0, 1, axis=1)

    # rename + order
    summary = summary.rename(columns={
        "faith": "faith_mean",
        "sanity": "sanity_mean",
        "complexity": "complexity_mean",
    }, level=0)

    summary = summary.reindex(["faith_mean","sanity_mean","complexity_mean"], axis=1, level=0)
    summary = summary.reindex(["mean","std"], axis=1, level=1)
    '''
    print(summary)
    summary.to_csv(save_path / "xai_metrics_summary_table.csv")
    
    
# ============================================================
# Metric Distributions
# ============================================================

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


def plot_metric_distributions_by_model(df_vec: pd.DataFrame, save_path: Path):
    """
    RQ3 (GradCAM vs IG) + RQ4 (sanity/faithfulness compliance) (recommended main plot).

    Purpose: Boxplots of per-sample metric distributions with separate panels per model.
    Output: One plot per metric, faceted by model; within each panel compares explainer × pair_type.
    """

    metrics = [("faith","Faithfulness (Deletion AUC)"), ("sanity","Sanity (Spearman)"), ("complexity","Sparseness")]
    
    for metric_name, title in metrics:
       
        sub = df_vec[df_vec["metric"] == metric_name].dropna(subset=["value"])
        if sub.empty:
            continue
        
        g = sns.catplot(
            data=sub,
            x="explainer",
            y="value",
            hue="pair_type",
            col="model",
            kind="box",
            height=4,
            aspect=0.9,
            sharey=False,
        )
        g.fig.suptitle(title, y=1.05)
        g.savefig(save_path / f"boxplot_{metric_name}__by_model.png", dpi=300, bbox_inches="tight")
        plt.close(g.fig)


# ============================================================
# Significance Tests
# ============================================================

def run_significance_tests_independent(df_vec: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """
    RQ3 (GradCAM vs IG) + RQ4

    Independent-samples test (Mann–Whitney U) to check whether IG and GradCAM metric distributions differ
    within each (model, pair_type, metric).
    Expects df_vec columns: model, pair_type, explainer, metric, value
    Why: Adds quantitative evidence beyond boxplots. Uses independent test even if cases are paired;
    conservative when pairing exists.
    """
    results = []

    for (model, pair_type, metric), sub in df_vec.groupby(["model", "pair_type", "metric"]):
        ig = sub[sub["explainer"] == "IG"]["value"].dropna().to_numpy()
        gc = sub[sub["explainer"] == "GradCAM"]["value"].dropna().to_numpy()

        if len(ig) < 2 or len(gc) < 2:
            continue

        # two-sided Mann–Whitney U (robust default)
        u_stat, p = stats.mannwhitneyu(ig, gc, alternative="two-sided")

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        results.append({
            "model": model,
            "pair_type": pair_type,
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
    res_df = pd.DataFrame(results).sort_values(["metric", "model", "pair_type"])

    file_path = save_path / "significance_tests_mannwhitney.csv"
    res_df.to_csv(file_path, index=False)
    print(res_df)
    return res_df


def run_xai_metrics_analysis(
    run_root: Path,
    save_dir: Path,
) -> dict[str, Path]:
    
    ensure_dir(save_dir)

    df_vec = load_all_vectors(run_root)
    if df_vec.empty:
        print(f"[ANALYSIS][WARN] No XAI metric vectors found under: {run_root}")
        return {}

    print_summary_table(df_vec, save_dir)
    plot_metric_distributions_by_model(df_vec, save_dir)
    run_significance_tests_independent(df_vec, save_dir)

    out = {
        "summary_table": save_dir / "xai_metrics_summary_table.csv",
        "significance_tests": save_dir / "significance_tests_mannwhitney.csv",
    }

    for metric_name in ["faith", "sanity", "complexity"]:
        plot_path = save_dir / f"boxplot_{metric_name}__by_model.png"
        if plot_path.exists():
            out[f"boxplot_{metric_name}_by_model"] = plot_path

    return out

