from __future__ import annotations
from jaguar.utils.utils import ensure_dir
from pathlib import Path
import json
import pandas as pd
from typing import Literal

from jaguar.config import PATHS, SEED
from jaguar.utils.utils_datasets import get_group_aware_stratified_train_val_split, load_jaguar_from_FO_export


# ============================================================
# Load exported manifest into metadata table (one row per image)
# ============================================================

def build_split_table_from_torch_dataset(torch_ds) -> pd.DataFrame:
    """
    Build one-row-per-image table from JaguarDataset manifest (source of truth).
    Assumes dedup fields were exported into the manifest and are present in samples.
    """
    rows = []

    for i, s in enumerate(torch_ds.samples):
        gt = s.get("ground_truth", {}) if isinstance(s, dict) else {}
        label = gt.get("label", None)

        row = {
            "emb_row": i,  # aligns with dataset order if needed later
            "filepath": str(torch_ds._resolve_path(s["filepath"])),
            "filename": s.get("filename", Path(s["filepath"]).name),
            "identity_id": (str(label) if label is not None else None),
            "split_original": s.get("split", None),

            # dedup fields if present
            "burst_group_id": s.get("burst_group_id", None),
            "burst_cluster_size": s.get("burst_cluster_size", None),
            "burst_role": s.get("burst_role", None),  # singleton / duplicate / representative
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # normalize missing burst roles (if field absent)
    if "burst_role" in df.columns:
        df["burst_role"] = df["burst_role"].fillna("singleton")

    return df


# ============================================================
# Stage-wise logging helpers
# ============================================================

def _n_identities(df: pd.DataFrame, identity_col: str = "identity_id") -> int:
    if identity_col not in df.columns:
        return 0
    return int(df[identity_col].dropna().astype(str).nunique())


def print_stage_stats(
    stage_name: str,
    df: pd.DataFrame,
    identity_col: str = "identity_id",
) -> None:
    print(f"\n[{stage_name}]")
    print(f"  images     : {len(df)}")
    print(f"  identities : {_n_identities(df, identity_col)}")

    # optional dedup-role breakdown if present
    if "burst_role" in df.columns:
        role_counts = df["burst_role"].fillna("missing").value_counts(dropna=False).to_dict()
        print(f"  burst_role counts: {role_counts}")


def print_stage_delta(
    prev_df: pd.DataFrame,
    next_df: pd.DataFrame,
    prev_name: str,
    next_name: str,
    identity_col: str = "identity_id",
) -> None:
    prev_n = len(prev_df)
    next_n = len(next_df)
    removed_n = prev_n - next_n

    prev_ids = set(prev_df[identity_col].dropna().astype(str).unique()) if identity_col in prev_df.columns else set()
    next_ids = set(next_df[identity_col].dropna().astype(str).unique()) if identity_col in next_df.columns else set()

    removed_ids = prev_ids - next_ids

    print(f"\n[{prev_name} -> {next_name}]")
    print(f"  images removed    : {removed_n} ({removed_n / prev_n:.2%} of previous)" if prev_n > 0 else "  images removed    : 0")
    print(f"  identities removed: {len(removed_ids)}")


def print_split_assignment_stats(
    df_split: pd.DataFrame,
    strategy: str,
    identity_col: str = "identity_id",
    split_col: str = "split_final",
) -> None:
    print(f"\n[split assignment: {strategy}]")
    total_images = len(df_split)
    total_ids = _n_identities(df_split, identity_col)
    print(f"  total images     : {total_images}")
    print(f"  total identities : {total_ids}")

    for sp in sorted(df_split[split_col].dropna().unique().tolist()):
        part = df_split[df_split[split_col] == sp]
        n_img = len(part)
        n_ids = _n_identities(part, identity_col)
        print(
            f"  - {sp:<5} images={n_img:>6} ({(n_img / total_images):6.2%}) | "
            f"identities={n_ids:>5} ({(n_ids / total_ids):6.2%})" if total_images > 0 and total_ids > 0
            else f"  - {sp:<5} images={n_img} | identities={n_ids}"
        )

    # strategy validation checks
    if strategy == "open_set":
        split_to_ids = {
            sp: set(df_split.loc[df_split[split_col] == sp, identity_col].dropna().astype(str).unique())
            for sp in df_split[split_col].dropna().unique()
        }
        splits = sorted(split_to_ids.keys())
        any_overlap = False
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                ov = split_to_ids[splits[i]] & split_to_ids[splits[j]]
                if ov:
                    any_overlap = True
                    print(f"  [WARN] identity overlap between {splits[i]} and {splits[j]}: {len(ov)}")
        if not any_overlap:
            print("  [OK] open-set identity disjointness verified")

    elif strategy == "closed_set":
        # Not required, but informative: report identity overlaps
        split_to_ids = {
            sp: set(df_split.loc[df_split[split_col] == sp, identity_col].dropna().astype(str).unique())
            for sp in df_split[split_col].dropna().unique()
        }
        splits = sorted(split_to_ids.keys())
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                ov = split_to_ids[splits[i]] & split_to_ids[splits[j]]
                print(f"  identity overlap {splits[i]}↔{splits[j]}: {len(ov)} (allowed in closed-set)")


# ============================================================
# Dedup filtering policy before splitting
# ============================================================

def filter_rows_by_dedup_policy(
    df: pd.DataFrame,
    dedup_policy: Literal["keep_all", "drop_duplicates"] = "drop_duplicates",
) -> pd.DataFrame:
    """
    dedup_policy:
      - keep_all: keep all rows
      - drop_duplicates: remove rows marked as duplicates
      - representatives_only: keep only singleton + representative
    """
    out = df.copy()

    if dedup_policy == "keep_all":
        return out.reset_index(drop=True)

    if dedup_policy == "drop_duplicates":
        # robust: use burst_role if present, fallback to duplicate tag
        if "burst_role" in out.columns:
            mask = out["burst_role"].fillna("singleton") != "duplicate"
        return out.loc[mask].reset_index(drop=True)

    raise ValueError(f"Unknown dedup_policy: {dedup_policy}")


# ============================================================
# Split strategies
# ============================================================
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split



def make_closed_set_splits(
    df: pd.DataFrame,
    val_size: float = 0.2,
    seed: int = 51,
    identity_col: str = "identity_id",
) -> pd.DataFrame:
    """
    Closed-set split (group-aware / burst-preserving):
      - image-disjoint
      - identity-overlapping allowed
      - burst groups kept together (if burst_group_id is present)
      - stratified approximately by identity at group level
      - singleton-label groups are kept in train to avoid losing classes

    Returns df with new column: split_final in {'train','val'}.
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
    Open-set split (identity-disjoint):
      - all images of an identity go to the same split
      - val_size targets the fraction of IMAGES in val (approximately)
      - exact ratio may not be achievable because identities are indivisible units

    Returns df with new column: split_final in {'train','val'}.
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

    # Greedy identity selection for val to get close to target image count
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
# Save artifacts
# ============================================================

def summarize_splits(
    df_split: pd.DataFrame,
    identity_col: str = "identity_id",
    split_col: str = "split_final",
) -> dict:
    summary = {
        "num_images_total": int(len(df_split)),
        "num_identities_total": int(df_split[identity_col].dropna().astype(str).nunique()),
        "images_per_split": {},
        "identities_per_split": {},
    }

    for sp in sorted(df_split[split_col].dropna().unique().tolist()):
        part = df_split[df_split[split_col] == sp]
        summary["images_per_split"][sp] = int(len(part))
        summary["identities_per_split"][sp] = int(part[identity_col].dropna().astype(str).nunique())

    return summary


def save_split_artifacts(
    out_dir: str | Path,
    split_df: pd.DataFrame,
    split_config: dict,
    file_stem: str = "split_assignments",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_df.to_parquet(out_dir / f"{file_stem}.parquet", index=False)

    summary = summarize_splits(split_df)

    with open(out_dir / "split_config.json", "w") as f:
        json.dump(split_config, f, indent=2)

    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Saved split artifacts to {out_dir}")
    print(json.dumps(summary, indent=2))


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

    Returns: (fo_wrapper, torch_ds, split_df, save_dir)
    """
    # save dir naming
    save_dir = out_root / f"{dataset_name}__{strategy}__dedup-{dedup_policy}__seed{seed}"
    ensure_dir(save_dir)

    #  Load exported manifest
    fo_wrapper, torch_ds = load_jaguar_from_FO_export(
        manifest_dir,
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=overwrite_db,
    )

    print(f"\n[load] dataset_name={dataset_name}")
    print(f"  manifest_dir : {manifest_dir}")
    print(f"  torch_ds size : {len(torch_ds)}")

    # Build table from manifest (full dataset)
    df_all = build_split_table_from_torch_dataset(torch_ds)
    print_stage_stats("stage1: manifest -> split_table (full)", df_all)

    # Dedup dataset by applying filtering policy
    df_split_input = filter_rows_by_dedup_policy(df_all, dedup_policy=dedup_policy)
    print_stage_stats(f"stage2: after dedup_policy={dedup_policy}", df_split_input)
    print_stage_delta(
        prev_df=df_all,
        next_df=df_split_input,
        prev_name="stage1 full",
        next_name=f"stage2 dedup={dedup_policy}",
    )

    # Split assignment
    if strategy == "closed_set":
        split_df = make_closed_set_splits(
            df_split_input,
            val_size=val_size,
            seed=seed,
        )
        strategy_desc = "closed_set (image-disjoint; identity overlap allowed across splits)"
    elif strategy == "open_set":
        split_df = make_open_set_splits(
            df_split_input,
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