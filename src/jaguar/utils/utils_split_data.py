import json
from pathlib import Path
from typing import Literal
import pandas as pd

# ============================================================
# Load exported manifest into metadata table (one row per image)
# ============================================================
def build_split_table_from_torch_dataset(torch_ds) -> pd.DataFrame:
    """
    Build one-row-per-image table from JaguarDataset manifest (source of truth).
    Flatten per-sample manifest entries into a tabular split table so later split logic works with explicit metadata 
    columns (identity, dedup tags, file references).
    """
    rows = []

    for i, s in enumerate(torch_ds.samples):
        gt = s.get("ground_truth", {}) if isinstance(s, dict) else {}
        label = gt.get("label", None)

        row = {
            "emb_row": i,  # stable row index to align with dataset-order-based artifacts if needed
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
    Dedup is applied *before* splitting so near-duplicate bursts do not distort split statistics or leak overly similar 
    images across train/val

    Current project policies:
      - keep_all: keep all rows
      - drop_duplicates: remove rows marked as duplicates
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
    '''
    Save both assignments and provenance (config/summary) so any model run can be traced back to the exact split protocol used.
    '''
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

