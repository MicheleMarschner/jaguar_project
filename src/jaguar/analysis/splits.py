import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from jaguar.config import DATA_STORE, EXPERIMENTS_STORE, PATHS
from jaguar.utils.utils import ensure_dir, resolve_path, to_abs
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

    file_path = save_dir / f"duplicate_group_{jaguar_name}_n{cluster_size}"
    plt.savefig(file_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    artifacts_dir = resolve_path("splits/jaguar_burst__str_closed_set__pol_drop_duplicates__k1", EXPERIMENTS_STORE)
    save_dir = PATHS.results / "splits"
    ensure_dir(save_dir)
    img_root = PATHS.data_train
    manifest_dir = resolve_path("fiftyone/splits_curated", DATA_STORE)
    dataset_name = "jaguar_curated"
    
    split_df = pd.read_parquet(artifacts_dir / "full_split.parquet")

    fo_wrapper, torch_ds = load_full_jaguar_from_FO_export(
        manifest_dir,
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=False,
    )

    with open(artifacts_dir / "config.json", "r") as f:
        config = json.load(f)

    strategy = config["strategy"]
    dedup_policy = config['dedup_policy']

    plot_split_identity_histograms(split_df, save_dir, dedup_policy)
    if strategy == "closed_set":
        check_split_with_closed_set_policy(split_df)
    elif strategy == "open_set":
        check_split_with_open_set_policy(split_df)
    else:
        raise RuntimeError(f"Unknown strategy: {strategy}")
    
    cluster = find_largest_duplicate_group(split_df)
    plot_duplicate_group(
        cluster,
        data_root=img_root,
        save_dir=save_dir
    )