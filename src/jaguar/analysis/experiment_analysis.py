from pathlib import Path
import pandas as pd
from jaguar.utils.utils import ensure_dir
import matplotlib.pyplot as plt




def run_backbone_analysis(df: pd.DataFrame, save_dir: Path) -> None:
    if df.empty:
        print("[ANALYSIS][WARN] backbone summary is empty")
        return

    required_cols = {"backbone_name", "mAP"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ANALYSIS][WARN] backbone summary missing columns: {sorted(missing)}")
        return

    plot_df = df.copy()
    plot_df = plot_df.sort_values("mAP", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["backbone_name"], plot_df["mAP"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mAP")
    plt.title("Backbone Comparison")
    plt.tight_layout()

    out_path = save_dir / "backbone_map_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[ANALYSIS] Saved backbone plot: {out_path}")


def run_loss_analysis(df: pd.DataFrame, save_dir: Path) -> None:
    if df.empty:
        print("[ANALYSIS][WARN] loss summary is empty")
        return

    required_cols = {"head_type", "mAP"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ANALYSIS][WARN] loss summary missing columns: {sorted(missing)}")
        return

    plot_df = df.copy()
    plot_df["label"] = plot_df["head_type"].astype(str)

    if "s" in plot_df.columns:
        plot_df["label"] = plot_df["label"] + " | s=" + plot_df["s"].astype(str)
    if "m" in plot_df.columns:
        plot_df["label"] = plot_df["label"] + " | m=" + plot_df["m"].astype(str)

    plot_df = plot_df.sort_values("mAP", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["label"], plot_df["mAP"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mAP")
    plt.title("Loss Comparison")
    plt.tight_layout()

    out_path = save_dir / "loss_map_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[ANALYSIS] Saved loss plot: {out_path}")


def run_deduplication_analysis(df: pd.DataFrame, save_dir: Path) -> None:
    if df.empty:
        print("[ANALYSIS][WARN] deduplication summary is empty")
        return

    required_cols = {"experiment_name", "mAP"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ANALYSIS][WARN] deduplication summary missing columns: {sorted(missing)}")
        return

    plot_df = df.copy()
    plot_df = plot_df.sort_values("mAP", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["experiment_name"], plot_df["mAP"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mAP")
    plt.title("Deduplication Comparison")
    plt.tight_layout()

    out_path = save_dir / "deduplication_map_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[ANALYSIS] Saved deduplication plot: {out_path}")


def run_experiment_analysis(
    *,
    experiment_group: str,
    output_profile: str,
    summary_path: Path,
) -> None:
        if not summary_path.exists():
            print(f"[ANALYSIS][WARN] Missing summary for {experiment_group}: {summary_path}")
            return

        df = pd.read_csv(summary_path)

        save_dir = summary_path.parent / "analysis"
        ensure_dir(save_dir)

        if output_profile == "deduplication":
            run_deduplication_analysis(df, save_dir)
            return

        if output_profile == "backbone":
            run_backbone_analysis(df, save_dir)
            return

        if output_profile == "loss":
            run_loss_analysis(df, save_dir)
            return

        if output_profile == "optim_sched":
            print(f"[ANALYSIS] optim_sched analysis not implemented yet: {experiment_group}")
            return

        print(
            f"[ANALYSIS][WARN] No analysis module implemented for "
            f"experiment_group={experiment_group}, output_profile={output_profile}"
        )

