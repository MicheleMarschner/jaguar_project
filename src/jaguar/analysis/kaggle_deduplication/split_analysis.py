import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from jaguar.utils.utils import ensure_dir, to_abs
from jaguar.utils.utils_datasets import load_full_jaguar_from_FO_export


def plot_split_identity_histograms(
    split_df: pd.DataFrame,
    save_dir: Path,
    dedup_policy: str,
    identity_col: str = "identity_id",
    split_col: str = "split_final",
    filename: str | None = None,
):
    """
    Plot side-by-side histograms (train vs val) of image counts per jaguar identity.
    """
    df = split_df.copy()
    identity_order = sorted(df[identity_col].astype(str).unique().tolist())

    df["_is_kept_like"] = df["keep_curated"].astype(bool)
    df["_is_duplicate"] = ~df["_is_kept_like"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    for ax, split_name in zip(axes, ["train", "val"]):
        part = df[df[split_col] == split_name]

        if dedup_policy == "drop_duplicates":
            # --- STACKED ---
            kept = (
                part.loc[part["_is_kept_like"], identity_col]
                .value_counts()
                .reindex(identity_order, fill_value=0)
            )
            dups = (
                part.loc[part["_is_duplicate"], identity_col]
                .value_counts()
                .reindex(identity_order, fill_value=0)
            )

            x = np.arange(len(identity_order))
            ax.bar(x, kept.values, label="Kept Images")
            ax.bar(x, dups.values, bottom=kept.values, label="Dropped Duplicates")

        else:
            # --- SIMPLE BARPLOT ---
            counts_df = (
                part[identity_col]
                .value_counts()
                .reindex(identity_order, fill_value=0)
                .reset_index(name="count")
                .rename(columns={"index": identity_col})
            )
            sns.barplot(data=counts_df, x=identity_col, y="count", ax=ax)

        # ----- SHARED CONFIGURATION (only once!) -----
        ax.set_title(f"{split_name.title()} Split ({dedup_policy})")
        ax.set_xticks(np.arange(len(identity_order)))
        ax.set_xticklabels(identity_order)
        plt.setp(ax.get_xticklabels(), rotation=90)

        ax.set_xlabel("Jaguar Identity")
        ax.set_ylabel("Image Count")
        ax.grid(axis="y", alpha=0.25)

    # Legend only for stacked charts
    if dedup_policy == "drop_duplicates":
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    fig.suptitle("Train vs Val Image Counts per Jaguar Identity", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    filename = f"split_hist_train_val__{dedup_policy}.png"
    out_fp = save_dir / filename
    fig.savefig(out_fp, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_fp


def check_split_with_closed_set_policy(
    split_df: pd.DataFrame,
    val_split_size: float = 0.2,
    identity_col: str = "identity_id",
):
    df = split_df.copy()
    total_images = len(df)

    # ---------------------------------
    # Basic split assignment validation
    # ---------------------------------
    if df["split_final"].isna().any():
        raise RuntimeError(f"{df['split_final'].isna().sum()} rows have no split_final assignment")

    valid_splits = {"train", "val"}
    found_splits = set(df["split_final"].astype(str).unique())
    invalid = found_splits - valid_splits
    if invalid:
        raise RuntimeError(f"Invalid split labels found: {invalid}")

    # ---------------------------------
    # Row-level image counts
    # ---------------------------------
    n_train = (df["split_final"] == "train").sum()
    n_val   = (df["split_final"] == "val").sum()

    print(f"[ClosedSet] Images: train={n_train} ({n_train/total_images:.2%}) | "
          f"val={n_val} ({n_val/total_images:.2%})")
    print(f"[ClosedSet] Target val_ratio={val_split_size:.2%} | achieved={n_val/total_images:.2%}")

    # ---------------------------------
    # Identity counts (overlap allowed)
    # ---------------------------------
    train_ids = set(df.loc[df["split_final"]=="train", identity_col].astype(str))
    val_ids   = set(df.loc[df["split_final"]=="val",   identity_col].astype(str))
    overlap   = train_ids & val_ids

    print(f"[ClosedSet] Identities: train={len(train_ids)} | val={len(val_ids)} | "
          f"overlap={len(overlap)} (allowed)")

    # ---------------------------------
    # Duplicate statistics (keep_curated)
    # ---------------------------------
    if "keep_curated" in df.columns:
        kept_train = df[(df["split_final"]=="train") & (df["keep_curated"]==True)]
        dup_train  = df[(df["split_final"]=="train") & (df["keep_curated"]==False)]

        kept_val   = df[(df["split_final"]=="val") & (df["keep_curated"]==True)]
        dup_val    = df[(df["split_final"]=="val") & (df["keep_curated"]==False)]

        print(f"[ClosedSet] Train: kept={len(kept_train)} | duplicates_removed={len(dup_train)}")
        print(f"[ClosedSet] Val:   kept={len(kept_val)} | duplicates_removed={len(dup_val)}")

    # ---------------------------------
    # Burst integrity check
    # ---------------------------------
    if "burst_group_id" in df.columns:
        per_group_split = df.groupby("burst_group_id")["split_final"].nunique()
        if (per_group_split > 1).any():
            bad = per_group_split[per_group_split > 1].index.tolist()[:10]
            raise RuntimeError(f"Burst integrity violation: groups split across train/val: {bad}")
        print("[ClosedSet] Burst-group integrity OK.")
    else:
        print("[ClosedSet] No burst_group_id → burst integrity check skipped.")


def check_split_with_open_set_policy(
    split_df: pd.DataFrame,
    val_split_size: float = 0.2,
    identity_col: str = "identity_id",
):
    df = split_df.copy()
    total_images = len(df)

    if df["split_final"].isna().any():
        raise RuntimeError(f"{df['split_final'].isna().sum()} rows have no split_final assignment")

    # ---------------------------------
    # Identity disjointness
    # ---------------------------------
    train_ids = set(df.loc[df["split_final"]=="train", identity_col].astype(str))
    val_ids   = set(df.loc[df["split_final"]=="val",   identity_col].astype(str))
    overlap   = train_ids & val_ids

    if overlap:
        raise RuntimeError(f"Identity overlap in open-set split: {list(overlap)[:10]}")

    # ---------------------------------
    # Counts & diagnostics
    # ---------------------------------
    n_train = (df["split_final"] == "train").sum()
    n_val   = (df["split_final"] == "val").sum()

    print(f"[OpenSet] Images: train={n_train} ({n_train/total_images:.2%}) | "
          f"val={n_val} ({n_val/total_images:.2%})")
    print(f"[OpenSet] Identities: train={len(train_ids)} | val={len(val_ids)} (disjoint OK)")
    print(f"[OpenSet] Target val_ratio={val_split_size:.2%} | achieved={n_val/total_images:.2%}")

    # ---------------------------------
    # Duplicate statistics
    # ---------------------------------
    if "keep_curated" in df.columns:
        kept_train = df[(df["split_final"]=="train") & (df["keep_curated"]==True)]
        dup_train  = df[(df["split_final"]=="train") & (df["keep_curated"]==False)]

        kept_val   = df[(df["split_final"]=="val") & (df["keep_curated"]==True)]
        dup_val    = df[(df["split_final"]=="val") & (df["keep_curated"]==False)]

        print(f"[OpenSet] Train: kept={len(kept_train)} | duplicates_removed={len(dup_train)}")
        print(f"[OpenSet] Val:   kept={len(kept_val)} | duplicates_removed={len(dup_val)}")

    # ---------------------------------
    # Burst integrity check
    # ---------------------------------
    if "burst_group_id" in df.columns:
        per_group_split = df.groupby("burst_group_id")["split_final"].nunique()
        if (per_group_split > 1).any():
            bad = per_group_split[per_group_split > 1].index.tolist()[:10]
            raise RuntimeError(f"Burst integrity violation: groups split across train/val: {bad}")
        print("[OpenSet] Burst-group integrity OK.")
    else:
        print("[OpenSet] No burst_group_id → burst integrity check skipped.")


def find_largest_duplicate_group(df):
    # keep only rows that have a duplicate_cluster_id
    dup = df.dropna(subset=["duplicate_cluster_id"]).copy()

    # count sizes of all duplicate clusters
    sizes = dup["duplicate_cluster_id"].value_counts()

    if sizes.empty:
        raise ValueError("No duplicate clusters found.")

    # pick the largest duplicate cluster
    cluster_id = sizes.index[0]

    # extract rows belonging to this cluster
    cluster = df[df["duplicate_cluster_id"] == cluster_id].copy()

    # identify the representative
    reps = cluster[cluster["keep_curated"] == True]
    if len(reps) != 1:
        raise RuntimeError(f"Cluster {cluster_id} must have exactly one representative.")

    rep = reps.iloc[0]  # Series

    # sort so representative is first
    dups = cluster[cluster["keep_curated"] == False].copy()
    cluster_sorted = pd.concat([rep.to_frame().T, dups], ignore_index=True)

    # print jaguar name + cluster length
    jaguar = rep["identity_id"]
    print(f"Largest duplicate group: Jaguar '{jaguar}' | size={len(cluster_sorted)}")

    return cluster_sorted


def plot_duplicate_group(cluster_df, data_root, save_dir):
    
    df = cluster_df.copy()
    # take first 5 images (rep is already first)
    df = df.head(5)

    n = len(df)

    # Extract metadata for title
    jaguar_name = str(df["identity_id"].iloc[0])
    cluster_id = str(df["duplicate_cluster_id"].iloc[0])
    cluster_size = len(cluster_df)   # full group size, not just 5

    plt.figure(figsize=(3 * n, 3))

    for i, row in enumerate(df.itertuples(index=False), start=1):
        if hasattr(row, "filepath_root") and hasattr(row, "filepath_rel"):
            fp = to_abs(row.filepath_root, row.filepath_rel)
        else:
            # Fallback: resolve by filename under the train image root
            fp = Path(data_root) / Path(row.filename).name

        img = Image.open(fp).convert("RGB")

        ax = plt.subplot(1, n, i)
        ax.imshow(img)
        ax.axis("off")

        role = "REP" if row.keep_curated else "DUP"
        title = f"{role}\n{row.filename}"
        ax.set_title(title, fontsize=8)

    plt.suptitle(f"Jaguar: {jaguar_name} | Duplicate Group Size: {cluster_size} | Cluster ID: {cluster_id}", fontsize=14, y=1.05)
    plt.tight_layout()

    file_path = save_dir / f"duplicate_group_{jaguar_name}_n{cluster_size}.png"
    plt.savefig(file_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {save_dir}")

    return file_path



def summarize_curation_sweep(
    artifacts_dir: Path,
    save_dir: Path,
) -> dict[str, Path]:
    ensure_dir(save_dir)
    sweep_df = pd.read_parquet(artifacts_dir / "pHash_threshold_sweep.parquet").sort_values("phash_threshold").copy()
    config_path = artifacts_dir / "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    chosen_threshold = int(config["intra_burst_phash_threshold"])
    train_k = int(config["train_k"])
    val_k = int(config["val_k"])

    keep_cols = [
        "phash_threshold",
        "train_kept",
        "train_dropped",
        "train_drop_pct",
        "val_kept",
        "val_dropped",
        "val_drop_pct",
    ]
    summary_df = sweep_df[keep_cols].copy()

    chosen_row = summary_df.loc[summary_df["phash_threshold"] == chosen_threshold]
    if len(chosen_row) == 0:
        raise ValueError(f"Chosen threshold {chosen_threshold} not found in sweep file.")
    chosen_row = chosen_row.iloc[0]

    print(f"[Stage2] Loaded sweep from: {artifacts_dir}")
    print(f"[Stage2] Loaded config from: {config_path}")
    print(f"[Stage2] Chosen intra-burst pHash threshold: {chosen_threshold}")
    print(f"[Stage2] train_k={train_k}, val_k={val_k}")
    print(
        f"[Stage2] At threshold {chosen_threshold}: "
        f"train_dropped={int(chosen_row['train_dropped'])} ({float(chosen_row['train_drop_pct']):.2%}), "
        f"val_dropped={int(chosen_row['val_dropped'])} ({float(chosen_row['val_drop_pct']):.2%})"
    )

    # Save compact CSV
    csv_path = save_dir / "curation_sweep_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    # Save markdown table
    md_path = save_dir / "curation_sweep_summary.md"
    with open(md_path, "w") as f:
        f.write(summary_df.to_markdown(index=False))

    # Minimal plot
    plt.figure(figsize=(6, 4))
    plt.plot(summary_df["phash_threshold"], summary_df["train_drop_pct"], marker="o", label="Train drop %")
    plt.plot(summary_df["phash_threshold"], summary_df["val_drop_pct"], marker="o", label="Val drop %")
    plt.axvline(chosen_threshold, linestyle="--", label=f"Chosen = {chosen_threshold}")
    plt.xlabel("Intra-burst pHash threshold")
    plt.ylabel("Dropped fraction")
    plt.title("Curation sweep")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = save_dir / "curation_sweep.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Stage2] Saved CSV to: {csv_path}")
    print(f"[Stage2] Saved markdown table to: {md_path}")
    print(f"[Stage2] Saved plot to: {plot_path}")

    return {
        "csv": csv_path,
        "markdown": md_path,
        "plot": plot_path,
    }


def run_split_diagnostics(
    curated_file_path: Path,
    save_dir: Path,
    img_root: Path,
    manifest_dir: Path,
    dataset_name: str = "jaguar_curated",
    use_fiftyone: bool = True,
) -> dict[str, Path]:
    
    curated_dir = Path(curated_file_path).parent
    #full_dir = (curated_dir.parent) / #mit dup_true
    
    #full_split_path = config["data"]["split_data_path"]
    #full_artifacts_dir = PATHS.runs / full_split_path

    # curated_split_path = Path(curated_split_path)
    # curated_artifacts_dir = curated_split_path.parent
    split_df = pd.read_parquet(curated_file_path)

    load_full_jaguar_from_FO_export(
        manifest_dir,
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=False,
        use_fiftyone=use_fiftyone
    )

    with open(curated_dir / "config.json", "r") as f:
        config = json.load(f)

    strategy = config["strategy"]
    dedup_policy = config["dedup_policy"]

    ensure_dir(save_dir)

    ### !TODO if not visible yet plothistogramm of full split (after bursts)

    hist_path = plot_split_identity_histograms(split_df, save_dir, dedup_policy)

    if strategy == "closed_set":
        check_split_with_closed_set_policy(split_df)
    elif strategy == "open_set":
        check_split_with_open_set_policy(split_df)
    else:
        raise RuntimeError(f"Unknown strategy: {strategy}")

    cluster = find_largest_duplicate_group(split_df)
    duplicate_group_path = plot_duplicate_group(
        cluster,
        data_root=img_root,
        save_dir=save_dir,
    )

    out = summarize_curation_sweep(
        artifacts_dir=curated_dir,
        save_dir=save_dir
    )

    return {
        "split_histogram": hist_path,
        "largest_duplicate_group_plot": duplicate_group_path,
    }