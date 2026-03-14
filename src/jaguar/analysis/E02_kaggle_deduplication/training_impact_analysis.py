from pathlib import Path
from typing import Any
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from jaguar.utils.utils import read_json_if_exists


def load_run(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Load one experiment run from:
    - metrics.json
    - train_history.json
    """
    metrics = read_json_if_exists(run_dir / "metrics.json")
    history = pd.DataFrame(read_json_if_exists(run_dir / "train_history.json"))

    if "epoch" in history.columns:
        history = history.sort_values("epoch").reset_index(drop=True)

    return metrics, history


# ----------------------------
# Metric extraction
# ----------------------------

def extract_summary(metrics: dict[str, Any], history: pd.DataFrame, label: str) -> dict[str, Any]:
    """
    Extract only the main comparison metrics for one run.
    """
    metric_block = metrics.get("metrics", {})

    best_epoch = metrics.get("best_epoch")
    best_pairwise = float(metrics.get("best_score", float("nan")))

    final_pairwise = float(history["val_pairwise_AP"].iloc[-1]) if "val_pairwise_AP" in history.columns else float("nan")
    post_peak_drop_pairwise = final_pairwise - best_pairwise

    return {
        "condition": label,
        "experiment_name": metrics.get("experiment_name"),
        "id_balanced_mAP": metric_block.get("id_balanced_mAP"),
        "pairwise_AP": metric_block.get("pairwise_AP"),
        "rank1": metric_block.get("rank1"),
        "sim_gap": metric_block.get("sim_gap"),
        "best_epoch": best_epoch,
        "best_pairwise_AP": best_pairwise,
        "final_pairwise_AP": final_pairwise,
        "post_peak_drop_pairwise_AP": post_peak_drop_pairwise,
    }


def build_main_comparison_table(
    full_summary: dict[str, Any],
    curated_summary: dict[str, Any],
) -> pd.DataFrame:
    """
    Build the main comparison table plus delta row:
    delta = curated - full
    """
    cols = [
        "condition",
        "id_balanced_mAP",
        "pairwise_AP",
        "rank1",
        "sim_gap",
        "best_epoch",
    ]

    full_row = {k: full_summary[k] for k in cols}
    curated_row = {k: curated_summary[k] for k in cols}

    delta_row = {"condition": "delta_curated_minus_full"}
    for k in cols:
        if k == "condition":
            continue
        delta_row[k] = curated_summary[k] - full_summary[k]

    return pd.DataFrame([full_row, curated_row, delta_row])


def build_training_dynamics_table(
    full_summary: dict[str, Any],
    curated_summary: dict[str, Any],
) -> pd.DataFrame:
    """
    Build the training dynamics table plus delta row:
    delta = curated - full
    """
    cols = [
        "condition",
        "best_pairwise_AP",
        "final_pairwise_AP",
        "post_peak_drop_pairwise_AP",
    ]

    full_row = {k: full_summary[k] for k in cols}
    curated_row = {k: curated_summary[k] for k in cols}

    delta_row = {"condition": "delta_curated_minus_full"}
    for k in cols:
        if k == "condition":
            continue
        delta_row[k] = curated_summary[k] - full_summary[k]

    return pd.DataFrame([full_row, curated_row, delta_row])


# ----------------------------
# Plot preparation
# ----------------------------

def prepare_history_for_plot(history: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Keep only the series we want to plot and attach condition label.
    """
    keep_cols = ["epoch"]
    for c in ["train_loss", "val_mAP"]:
        if c in history.columns:
            keep_cols.append(c)

    out = history[keep_cols].copy()
    out["condition"] = label
    return out


# ----------------------------
# Plotting
# ----------------------------

def plot_train_loss_comparison(
    full_hist: pd.DataFrame,
    curated_hist: pd.DataFrame,
    save_path: Path,
) -> Path:
    """
    Plot train loss for full vs curated.
    """
    df = pd.concat(
        [
            prepare_history_for_plot(full_hist, "full"),
            prepare_history_for_plot(curated_hist, "curated"),
        ],
        ignore_index=True,
    )

    if "train_loss" not in df.columns:
        raise ValueError("train_loss not found in train_history.json")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))

    sns.lineplot(
        data=df,
        x="epoch",
        y="train_loss",
        hue="condition",
        marker="o",
    )

    plt.title("Train Loss: Full vs Curated")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return save_path


def plot_val_map_comparison(
    full_hist: pd.DataFrame,
    curated_hist: pd.DataFrame,
    save_path: Path,
) -> Path:
    """
    Plot validation mAP for full vs curated.
    """
    df = pd.concat(
        [
            prepare_history_for_plot(full_hist, "full"),
            prepare_history_for_plot(curated_hist, "curated"),
        ],
        ignore_index=True,
    )

    if "val_mAP" not in df.columns:
        raise ValueError("val_mAP not found in train_history.json")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))

    sns.lineplot(
        data=df,
        x="epoch",
        y="val_mAP",
        hue="condition",
        marker="o",
    )

    plt.title("Validation mAP: Full vs Curated")
    plt.xlabel("Epoch")
    plt.ylabel("Validation mAP")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return save_path


def plot_loss_and_map_overlay(
    full_hist: pd.DataFrame,
    curated_hist: pd.DataFrame,
    save_path: Path,
) -> Path:
    """
    Optional combined plot with twin y-axis:
    - left axis: train loss
    - right axis: val mAP
    """
    full_df = prepare_history_for_plot(full_hist, "full")
    curated_df = prepare_history_for_plot(curated_hist, "curated")
    df = pd.concat([full_df, curated_df], ignore_index=True)

    if "train_loss" not in df.columns or "val_mAP" not in df.columns:
        raise ValueError("Need both train_loss and val_mAP for overlay plot")

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    for condition, sub in df.groupby("condition"):
        sns.lineplot(
            data=sub,
            x="epoch",
            y="train_loss",
            marker="o",
            ax=ax1,
            label=f"{condition} train_loss",
        )
        sns.lineplot(
            data=sub,
            x="epoch",
            y="val_mAP",
            marker="o",
            ax=ax2,
            label=f"{condition} val_mAP",
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax2.set_ylabel("Validation mAP")
    ax1.set_title("Train Loss and Validation mAP: Full vs Curated")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return save_path


def build_stage3_summary_table(
    run_specs: list[dict[str, Any]],
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build one compact summary table across multiple Stage-3 runs
    """
    rows = []

    for spec in run_specs:
        run_dir = Path(spec["run_dir"])
        metrics = read_json_if_exists(run_dir / "metrics.json")

        metric_block = metrics.get("metrics", {})

        rows.append(
            {
                "condition": spec["condition"],
                "run_name": metrics.get("experiment_name", run_dir.name),
                "train_k": spec.get("train_k"),
                "val_k": spec.get("val_k"),
                "id_balanced_mAP": metric_block.get("id_balanced_mAP"),
                "pairwise_AP": metric_block.get("pairwise_AP"),
                "rank1": metric_block.get("rank1"),
                "sim_gap": metric_block.get("sim_gap"),
                "best_epoch": metrics.get("best_epoch"),
            }
        )

    df = pd.DataFrame(rows)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df



def run_duplicate_impact_report(
    full_run_dir: str | Path,
    curated_run_dir: str | Path,
    save_dir: str | Path,
    make_overlay_plot: bool = False,
) -> dict[str, Path]:

    full_metrics, full_hist = load_run(full_run_dir)
    curated_metrics, curated_hist = load_run(curated_run_dir)

    full_summary = extract_summary(full_metrics, full_hist, label="full")
    curated_summary = extract_summary(curated_metrics, curated_hist, label="curated")

    main_table = build_main_comparison_table(full_summary, curated_summary)
    training_table = build_training_dynamics_table(full_summary, curated_summary)

    main_table_path = save_dir / "main_comparison.csv"
    training_table_path = save_dir / "training_dynamics.csv"

    main_table.to_csv(main_table_path, index=False)
    training_table.to_csv(training_table_path, index=False)

    train_loss_plot_path = plot_train_loss_comparison(
        full_hist=full_hist,
        curated_hist=curated_hist,
        save_path=save_dir / "train_loss_comparison.png",
    )

    val_map_plot_path = plot_val_map_comparison(
        full_hist=full_hist,
        curated_hist=curated_hist,
        save_path=save_dir / "val_map_comparison.png",
    )

    outputs = {
        "train_loss_plot": train_loss_plot_path,
        "val_map_plot": val_map_plot_path,
        "main_table": main_table_path,
        "training_table": training_table_path,
    }

    if make_overlay_plot:
        overlay_path = plot_loss_and_map_overlay(
            full_hist=full_hist,
            curated_hist=curated_hist,
            save_path=save_dir / "loss_and_map_overlay.png",
        )
        outputs["overlay_plot"] = overlay_path

    summary_df = build_stage3_summary_table(
        run_specs=[
            {"run_dir": "runs/full", "condition": "full", "train_k": None, "val_k": None},
            {"run_dir": "runs/curated_k1", "condition": "curated_k1", "train_k": 1, "val_k": 1},
            {"run_dir": "runs/curated_k3", "condition": "curated_k3", "train_k": 3, "val_k": 3},
        ],
        save_path="analysis/stage3_summary.csv",
    )

    return outputs