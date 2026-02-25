"""
Split generation pipeline for the Jaguar re-ID project.

Purpose in the project:
- takes the exported dataset manifest (single source of truth per image),
- applies the chosen dedup policy,
- creates reproducible train/val splits (closed-set or open-set),
- saves split assignments + config + summary as artifacts for downstream training/evaluation.

This script defines *data protocol* artifacts, not model training.
"""
from jaguar.utils.utils import ensure_dir
from pathlib import Path
import pandas as pd
from typing import Literal

from jaguar.config import PATHS, SEED
from jaguar.utils.utils_datasets import get_group_aware_stratified_train_val_split, load_jaguar_from_FO_export
from jaguar.utils.utils_split_data import (
    build_split_table_from_torch_dataset, 
    filter_rows_by_dedup_policy, 
    print_split_assignment_stats, 
    print_stage_delta, print_stage_stats,
    save_split_artifacts
)


# ============================================================
# Split strategies
# ============================================================
def make_closed_set_splits(
    df: pd.DataFrame,
    val_size: float = 0.2,
    seed: int = 51,
    identity_col: str = "identity_id",
) -> pd.DataFrame:
    """
    Closed-set re-ID setting:
    Same identity may appear in both train and val, but exact images/bursts stay separated. Useful for measuring generalization 
    to new images of known identities.

    in detail:
      - image-disjoint
      - identity-overlapping allowed
      - burst groups kept together (if burst_group_id is present)
      - stratified approximately by identity at group level
      - singleton-label groups are kept in train to avoid losing classes
    """
    train_idx, val_idx, out, groups_df = get_group_aware_stratified_train_val_split(
        df=df,
        val_split=val_size,
        seed=seed,
        identity_col=identity_col,
        burst_group_col="burst_group_id",
        filepath_col="filepath",
    )

    out["split_final"] = None
    out.loc[train_idx, "split_final"] = "train"
    out.loc[val_idx, "split_final"] = "val"

    if out["split_final"].isna().any():
        raise RuntimeError("Some rows were not assigned a split")

    return out


def make_open_set_splits(
    df: pd.DataFrame,
    val_size: float = 0.2,
    seed: int = 51,
    identity_col: str = "identity_id",
) -> pd.DataFrame:
    """
    Open-set re-ID setting:
    Identities are the split unit (not images), so validation measures generalization to entirely unseen jaguars.

    in detail (identity-disjoint):
      - all images of an identity go to the same split
      - val_size targets the fraction of IMAGES in val (approximately)
      - exact ratio may not be achievable because identities are indivisible units
    """
    out = df.copy().reset_index(drop=True)
    out["split_final"] = None

    out_ids = out[identity_col].astype(str)

    # image counts per identity (identity = indivisible unit in open-set)
    id_counts = (
        out_ids.value_counts()
        .rename_axis(identity_col)
        .reset_index(name="n_images")
    )

    total_images = int(len(out))
    target_val_images = int(round(total_images * val_size))
    if target_val_images <= 0 or target_val_images >= total_images:
        raise ValueError("Split sizes too aggressive for number of images/identities")

    # reproducible randomization before greedy packing
    # (shuffle first, then sort by size desc with stable sort to break ties reproducibly)
    id_counts = id_counts.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    id_counts = id_counts.sort_values("n_images", ascending=False, kind="stable").reset_index(drop=True)

    # We approximate the requested val image ratio by selecting whole identities (greedy selection). Exact val_size is usually 
    # impossible because identities are indivisible.
    val_ids = []
    current_val_images = 0

    for _, row in id_counts.iterrows():
        ident = str(row[identity_col])
        n_img = int(row["n_images"])

        diff_without = abs(target_val_images - current_val_images)
        diff_with = abs(target_val_images - (current_val_images + n_img))

        if diff_with <= diff_without:
            val_ids.append(ident)
            current_val_images += n_img

    # Fallback: ensure val is non-empty
    if len(val_ids) == 0:
        smallest = id_counts.sort_values("n_images", ascending=True).iloc[0]
        val_ids = [str(smallest[identity_col])]
        current_val_images = int(smallest["n_images"])

    val_ids = set(val_ids)
    all_ids = set(id_counts[identity_col].astype(str).tolist())
    train_ids = all_ids - val_ids

    if len(train_ids) == 0:
        raise ValueError("Open-set split failed: all identities assigned to val. Reduce val_size.")

    # Assign splits by identity membership (strict identity-disjointness)
    out.loc[out_ids.isin(train_ids), "split_final"] = "train"
    out.loc[out_ids.isin(val_ids), "split_final"] = "val"

    if out["split_final"].isna().any():
        missing = out.loc[out["split_final"].isna(), identity_col].unique().tolist()
        raise RuntimeError(f"Some rows were not assigned a split. Missing identities: {missing[:10]}")
    
    return out


# ============================================================
# Orchestration: load -> build table -> dedup filter -> split -> save
# ============================================================
def create_and_save_splits(
    manifest_dir: str | Path,
    dataset_name: str,
    out_root: str | Path,
    strategy: Literal["closed_set", "open_set"] = "closed_set",
    dedup_policy: Literal["keep_all", "drop_duplicates"] = "drop_duplicates",
    val_size: float = 0.2,
    seed: int = 51,
    overwrite_db: bool = False,
):
    """
    Pipeline:
      1) load dedup-exported dataset (manifest)
      2) build split table
      3) apply dedup filtering policy
      4) create splits (closed/open)
      5) save parquet + config + summary

    Prints stage-wise changes but only saves final artifacts.
    """
    save_dir = out_root / f"{dataset_name}__{strategy}__dedup-{dedup_policy}__seed{seed}"
    ensure_dir(save_dir)

    #  Step 1: load exported manifest (post-EDA / post-dedup annotation stage)
    #  This is the project-wide source of truth for image-level metadata.
    fo_wrapper, torch_ds = load_jaguar_from_FO_export(
        manifest_dir,
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=overwrite_db,
    )

    print(f"\n[load] dataset_name={dataset_name}")
    print(f"  manifest_dir : {manifest_dir}")
    print(f"  torch_ds size : {len(torch_ds)}")

    # Step 2: convert dataset manifest into a split-ready metadata table (one row = one image, including 
    # identity and dedup annotations).
    df_manifest_all = build_split_table_from_torch_dataset(torch_ds)
    print_stage_stats("stage1: manifest -> split_table (full)", df_manifest_all)

    # Step 3: choose which images are eligible for splitting based on dedup policy. This controls whether 
    # training/validation sees all burst members or only deduplicated data.
    df_split_candidates = filter_rows_by_dedup_policy(df_manifest_all, dedup_policy=dedup_policy)
    print_stage_stats(f"stage2: after dedup_policy={dedup_policy}", df_split_candidates)
    print_stage_delta(
        prev_df=df_manifest_all,
        next_df=df_split_candidates,
        prev_name="stage1 full",
        next_name=f"stage2 dedup={dedup_policy}",
    )

    # Step 4: assign train/val according to the evaluation protocol (closed-set = known identities; open-set = unseen identities).
    if strategy == "closed_set":
        split_df = make_closed_set_splits(
            df_split_candidates,
            val_size=val_size,
            seed=seed,
        )
        strategy_desc = "closed_set (image-disjoint; identity overlap allowed across splits)"
    elif strategy == "open_set":
        split_df = make_open_set_splits(
            df_split_candidates,
            val_size=val_size,
            seed=seed,
        )
        strategy_desc = "open_set (identity-disjoint; all images of an identity stay in one split)"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print_split_assignment_stats(split_df, strategy=strategy)

    split_config = {
        "dataset_name": dataset_name,
        "manifest_dir": str(manifest_dir),
        "strategy": strategy,
        "strategy_description": strategy_desc,
        "dedup_policy": dedup_policy,
        "val_size": float(val_size),
        "seed": int(seed),
    }

    # Step 5: persist split assignments as reusable experiment artifacts.
    save_split_artifacts(
        out_dir=save_dir,
        split_df=split_df,
        split_config=split_config,
    )

    return fo_wrapper, torch_ds, split_df, save_dir


if __name__ == "__main__":
    dataset_name = "jaguar_after_dedup"
    manifest_dir = PATHS.data_export / "dedup"
    dedup_policy = "drop_duplicates"        # "keep_all", "drop_duplicates"
    strategy = "open_set"                 # "closed_set" or "open_set"
    out_root = PATHS.runs / "splits"

    
    fo_wrapper, torch_ds, split_df, save_dir = create_and_save_splits(
        manifest_dir=PATHS.data_export / "dedup",
        dataset_name=dataset_name,
        out_root=out_root,
        strategy=strategy,                
        dedup_policy=dedup_policy,
        val_size=0.2,
        seed=SEED,
        overwrite_db=False,
    )
    
    print(f"\nDone. Saved to: {save_dir}")