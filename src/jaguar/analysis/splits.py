import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from jaguar.config import PATHS
from jaguar.utils.utils import ensure_dir
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export


def check_split_with_closed_set_policy(
    split_df: pd.DataFrame,
    val_size: float = 0.2,
    identity_col: str = "identity_id",
):
    total_images = int(len(split_df))

    if split_df["split_final"].isna().any():
        n_missing = int(split_df["split_final"].isna().sum())
        raise RuntimeError(f"{n_missing} rows have no split_final assignment")

    valid_splits = {"train", "val"}
    found_splits = set(split_df["split_final"].astype(str).unique().tolist())
    invalid = found_splits - valid_splits
    if invalid:
        raise RuntimeError(f"Invalid split labels found: {invalid}")

    # Row-level counts
    n_train_img = int((split_df["split_final"] == "train").sum())
    n_val_img = int((split_df["split_final"] == "val").sum())

    print(
        f"[ClosedSet] Images: train={n_train_img} ({n_train_img / total_images:.2%}) | "
        f"val={n_val_img} ({n_val_img / total_images:.2%})"
    )
    print(
        f"[ClosedSet] Target val image ratio={val_size:.2%}, "
        f"achieved={n_val_img / total_images:.2%}"
    )

    # Identity overlap is allowed in closed-set, but we report it
    train_identity_set = set(
        split_df.loc[split_df["split_final"] == "train", identity_col].astype(str).unique().tolist()
    )
    val_identity_set = set(
        split_df.loc[split_df["split_final"] == "val", identity_col].astype(str).unique().tolist()
    )
    overlap = train_identity_set & val_identity_set

    print(
        f"[ClosedSet] Identities: train={len(train_identity_set)} | "
        f"val={len(val_identity_set)} | overlap={len(overlap)} (closed-set overlap allowed)"
    )

    # Optional burst/group integrity check (only if helper column was saved)
    if "_split_group_id" in split_df.columns:
        per_group_split_nunique = split_df.groupby("_split_group_id")["split_final"].nunique()
        if (per_group_split_nunique > 1).any():
            bad_groups = per_group_split_nunique[per_group_split_nunique > 1].index.tolist()[:10]
            raise RuntimeError(
                f"Burst/group integrity violated: group spans multiple splits. Examples: {bad_groups}"
            )
        print("[ClosedSet] [OK] Burst groups kept intact.")
    else:
        print("[ClosedSet] [Info] No _split_group_id column found; burst integrity check skipped.")


def check_split_with_open_set_policy(
    split_df: pd.DataFrame,
    val_size: float = 0.2,
    identity_col: str = "identity_id",
):
    total_images = int(len(split_df))

    if split_df["split_final"].isna().any():
        n_missing = int(split_df["split_final"].isna().sum())
        raise RuntimeError(f"{n_missing} rows have no split_final assignment")

    # Safety check: identity-disjointness
    train_identity_set = set(
        split_df.loc[split_df["split_final"] == "train", identity_col].astype(str).unique().tolist()
    )
    val_identity_set = set(
        split_df.loc[split_df["split_final"] == "val", identity_col].astype(str).unique().tolist()
    )
    overlap = train_identity_set & val_identity_set
    if overlap:
        raise RuntimeError(f"Identity overlap detected in open-set split: {list(overlap)[:10]}")

    # Diagnostics
    n_train_img = int((split_df["split_final"] == "train").sum())
    n_val_img = int((split_df["split_final"] == "val").sum())

    n_train_ids = len(train_identity_set)
    n_val_ids = len(val_identity_set)

    print(
        f"[OpenSet] Images: train={n_train_img} ({n_train_img / total_images:.2%}) | "
        f"val={n_val_img} ({n_val_img / total_images:.2%})"
    )
    print(
        f"[OpenSet] Identities: train={n_train_ids} | val={n_val_ids} "
        f"(identity-disjoint enforced)"
    )
    print(
        f"[OpenSet] Target val image ratio={val_size:.2%}, "
        f"achieved={n_val_img / total_images:.2%}"
    )


def plot_split_identity_histograms(
    split_df: pd.DataFrame,
    save_dir,
    dedup_policy: str,
    identity_col: str = "identity_id",
    split_col: str = "split_final",
    filename: str | None = None,
):
    """
    Plot side-by-side histograms (train vs val) of image counts per jaguar identity.
    """
    ensure_dir(save_dir)
    df = split_df.copy()
    
    # stable identity order across both subplots
    identity_order = sorted(df[identity_col].unique().tolist())


    is_duplicate = df["burst_role"].fillna("singleton").astype(str).eq("duplicate")
    df["_is_duplicate"] = is_duplicate
    df["_is_kept_like"] = ~df["_is_duplicate"]


    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    split_names = ["train", "val"]

    for ax, sp in zip(axes, split_names):
        part = df[df[split_col] == sp].copy()

        if dedup_policy == "keep_all":
            # compute kept + duplicate counts per identity (stacked)
            kept_counts = (
                part.loc[part["_is_kept_like"], identity_col]
                .value_counts()
                .reindex(identity_order, fill_value=0)
            )
            dup_counts = (
                part.loc[part["_is_duplicate"], identity_col]
                .value_counts()
                .reindex(identity_order, fill_value=0)
            )

            x = range(len(identity_order))
            ax.bar(x, kept_counts.values, label="Kept Images")
            ax.bar(x, dup_counts.values, bottom=kept_counts.values, label="Burst Duplicates")

            ax.set_title(f"{sp.title()} Split (keep_all)")
            ax.set_xticks(list(x))
            ax.set_xticklabels(identity_order, rotation=90)

        else:
            # single-bar counts per identity
            counts_df = (
                part[identity_col]
                .value_counts()
                .reindex(identity_order, fill_value=0)
                .rename_axis(identity_col)
                .reset_index(name="count")
            )

            sns.barplot(
                data=counts_df,
                x=identity_col,
                y="count",
                ax=ax,
            )
            ax.set_title(f"{sp.title()} Split ({dedup_policy})")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        ax.set_xlabel("Jaguar Identity")
        ax.set_ylabel("Image Count")
        ax.grid(axis="y", alpha=0.25)

    # legend only once (keep_all mode)
    if dedup_policy == "keep_all":
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)

    fig.suptitle("Train vs Val Image Counts per Jaguar Identity", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if filename is None:
        filename = f"split_hist_train_val__{dedup_policy}.png"

    out_fp = save_dir / filename
    fig.savefig(out_fp, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_fp


if __name__ == "__main__":
    artifacts_dir = PATHS.runs / "splits" / "jaguar_after_dedup__open_set__dedup-keep_all__seed51"
    split_df = pd.read_parquet(artifacts_dir / "split_assignments.parquet")
    save_dir = PATHS.results / "splits"
    manifest_dir = PATHS.data_export / "dedup"
    dataset_name = "jaguar_after_dedup"

    fo_wrapper, torch_ds = load_jaguar_from_FO_export(
        manifest_dir,
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=False,
    )

    with open(artifacts_dir / "split_config.json", "r") as f:
        config = json.load(f)

    strategy = config["strategy"]
    val_size = float(config["val_size"])
    dedup_policy = config['dedup_policy']

    if strategy == "closed_set":
        check_split_with_closed_set_policy(split_df, val_size=val_size)
    elif strategy == "open_set":
        check_split_with_open_set_policy(split_df, val_size=val_size)
    else:
        raise RuntimeError(f"Unknown strategy: {strategy}")
    
    filename = f"split_identity_histograms__{strategy}__{dedup_policy}.png"
    out_fp = plot_split_identity_histograms(
        split_df=split_df,
        save_dir=save_dir,
        dedup_policy=dedup_policy,
        filename=filename
    )
    print(f"Saved plot to: {out_fp}")