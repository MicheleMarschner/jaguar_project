'''
wandb:
job_type="eda"

cfg_overrides = {
    "meta": {
        "experiment_family": "eda",
        "study_id": "dataset_eda__round1_train",
        "job_type": "eda",
    },
    "data": {
        "dataset_name": "jaguar_init",
        "dataset_version": "jaguar_stage0_v1",
        "data_root": str(PATHS.data_train),
    },
    "eda": {
        "analysis_name": "initial_dataset_eda",
        "identity_filter_thresholds": thresholds,
        "n_examples_resolution_gallery": n_examples,
        "merge_key": "filename",
        "merge_validate": "one_to_one",
    },
    "inputs": {
        "meta_img_features_parquet": str(PATHS.runs / "deduplication" / "meta_img_features.parquet"),
    },
    "outputs": {
        "save_dir": str(save_dir),
    },
}

'''


from pathlib import Path
import re
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Iterable


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# Fast "first-pass" sanity report before any expensive image scanning.
# Goal: catch schema/missing-value issues immediately.
def basic_integrity_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    report = {
        "train_shape": tuple(train_df.shape),
        "test_shape": tuple(test_df.shape),
        "train_columns": train_df.columns.tolist(),
        "test_columns": test_df.columns.tolist(),
        "train_missing": train_df.isnull().sum().to_dict(),
        "test_missing": test_df.isnull().sum().to_dict(),
    }

    print_section("TRAIN SET INFO")
    print(f"Shape: {train_df.shape}")
    print(f"Columns: {train_df.columns.tolist()}")
    print("Missing values:")
    print(train_df.isnull().sum())
    print("\nFirst 3 rows:")
    print(train_df.head(3).to_string(index=False))

    print_section("TEST SET INFO")
    print(f"Shape: {test_df.shape}")
    if "query_image" in test_df.columns:
        print(f"Unique query images: {test_df['query_image'].nunique()}")
    if "gallery_image" in test_df.columns:
        print(f"Unique gallery images: {test_df['gallery_image'].nunique()}")
    print("\nFirst 3 rows:")
    print(test_df.head(3).to_string(index=False))

    return report


# ----------------------------
# EDA: class distribution
# ----------------------------

# Identity frequency distribution drives almost every downstream decision:
# split protocol, rebalancing, filtering thresholds, and evaluation interpretation.
def class_distribution(train_df: pd.DataFrame) -> tuple[pd.Series, dict]:
    counts = train_df["ground_truth"].value_counts().sort_values(ascending=False)
    desc = counts.describe()

    summary = {
        "number of unique identities": int(len(counts)),
        "max images per identity": float(desc["max"]),
        "min images per identity": float(desc["min"]),
        "mean images per identity": float(desc["mean"]),
        "median images per identity": float(desc["50%"]),
    }

    print_section("IDENTITY DISTRIBUTION")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Identify identities that may need careful handling (few samples)
    min_samples_for_split = 2  
    low_sample_identities = counts[counts < min_samples_for_split]

    if len(low_sample_identities) > 0:   
        print(f"\nWarning: {len(low_sample_identities)} identities have fewer than {min_samples_for_split} images")

    print("\nTop-10 IDs by count:")
    print(counts.head(10))

    return counts, summary

# Scenario analysis only: summarizes consequences of minimum-images-per-identity rules
# without modifying the dataset yet.
def identity_filter_summary(
    identity_counts: pd.Series,
    thresholds: Iterable[int],
    out_dir: Path,
) -> pd.DataFrame:
    # ---- summary table ----
    rows = []
    for x in thresholds:
        keep = identity_counts[identity_counts >= x]
        kept_ids = int(len(keep))
        kept_samples = int(keep.sum())
        removed_ids = int((identity_counts < x).sum())
        removed_samples = int(identity_counts[identity_counts < x].sum())

        if kept_ids > 0:
            arr = keep.values
            row = dict(
                min_imgs_per_id=int(arr.min()),
                median_imgs_per_id=float(np.median(arr)),
                mean_imgs_per_id=float(arr.mean()),
                max_imgs_per_id=int(arr.max()),
            )
        else:
            row = dict(
                min_imgs_per_id=np.nan,
                median_imgs_per_id=np.nan,
                mean_imgs_per_id=np.nan,
                max_imgs_per_id=np.nan,
            )

        rows.append(
            dict(
                threshold=x,
                kept_ids=kept_ids,
                kept_samples=kept_samples,
                removed_ids=removed_ids,
                removed_samples=removed_samples,
                **row,
            )
        )

    summary = pd.DataFrame(rows)

    print("\n=== FILTER SUMMARY (keep IDs with >= x images) ===")
    print(summary.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "identity_filter_summary.csv", index=False)
    print("Saved:", out_dir / "identity_filter_summary.csv")

    return summary



# Reads image headers (and mode) to build a lightweight technical profile of the dataset.
# Failed loads are tracked to surface corrupted/missing files.
def analyze_images(
    df: pd.DataFrame,
    img_dir: Path,
    img_col: str = "filename",
) -> pd.DataFrame:
    widths: list[int] = []
    heights: list[int] = []
    modes: list[str] = []
    filenames_ok: list[str] = []
    filenames_err: list[str] = []

    filenames = df[img_col].tolist()

    for fn in tqdm(filenames, desc="Analyzing images"):
        path = img_dir / fn
        try:
            with Image.open(path) as img:
                widths.append(int(img.width))
                heights.append(int(img.height))
                modes.append(str(img.mode))
                filenames_ok.append(fn)
        except Exception:
            filenames_err.append(fn)

    stats_df = pd.DataFrame(
        {
            "filename": filenames_ok,
            "width": widths,
            "height": heights,
            "mode": modes,
        }
    )

    print_section("IMAGE PROPERTIES")
    if not stats_df.empty:
        print(stats_df[["width", "height"]].describe())
        print("\nMode distribution:")
        print(stats_df["mode"].value_counts())
    print(f"\nFailed to load images: {len(filenames_err)}")

    return stats_df

# Protects against silent train/CSV mismatches that would later break EDA, dedup, or training.
def check_filename_and_folder_consistency(
    train_df: pd.DataFrame,
    data_path: Path,
    filename_col: str = "filename",
) -> None:
    # filename format + index coverage
    pat = re.compile(r"^train_(\d{4})\.png$", re.IGNORECASE)
    ok = train_df[filename_col].astype(str).apply(lambda s: bool(pat.match(s)))
    print("filename format train_####.png valid:", f"{ok.mean()*100:.1f}%")
    if (~ok).any():
        print("bad filename examples:", train_df.loc[~ok, filename_col].head(10).tolist())

    nums = (
        train_df[filename_col]
        .astype(str)
        .str.extract(r"train_(\d{4})\.png", expand=False)
        .dropna()
        .astype(int)
    )
    if len(nums) > 0:
        mn, mx = int(nums.min()), int(nums.max())
        missing_idx = sorted(set(range(mn, mx + 1)) - set(nums.tolist()))
        print(f"index coverage: {mn}..{mx} | missing indices:", len(missing_idx))
        if missing_idx[:10]:
            print("missing index examples:", missing_idx[:10])
    else:
        print("index coverage: no valid indices extracted")

    # --- CSV vs folder check ---
    print("\n=== CSV ↔ TRAIN FOLDER CONSISTENCY ===")
    if not data_path.exists():
        print("TRAIN_DIR does not exist:", data_path)
    else:
        csv_files = set(train_df[filename_col].astype(str))
        disk_files = set(p.name for p in data_path.glob("*") if p.is_file())

        missing_on_disk = sorted(csv_files - disk_files)
        extra_on_disk = sorted(disk_files - csv_files)

        print("train folder files:", len(disk_files))
        print("missing on disk (in CSV, not in folder):", len(missing_on_disk))
        print("extra on disk (in folder, not in CSV):", len(extra_on_disk))

        if missing_on_disk[:10]:
            print("example missing:", missing_on_disk[:10])
        if extra_on_disk[:10]:
            print("example extra:", extra_on_disk[:10])

# Joins upstream feature artifacts (sharpness) with freshly computed image metadata
# so we can analyze quality signals together with size/resolution.
def merge_sharpness_with_image_stats(
    df_sharp: pd.DataFrame,
    img_stats_df: pd.DataFrame,
    key: str = "filename",
    validate: str = "one_to_one",
) -> pd.DataFrame:
    """
    Merge sharpness/features table with image stats (width/height/mode),
    then add derived size columns.
    """
    merged = df_sharp.merge(
        img_stats_df,
        on=key,
        how="left",
        validate=validate,
    ).copy()

    merged["resolution_px"] = merged["width"] * merged["height"]
    merged["short_side"] = merged[["width", "height"]].min(axis=1)
    merged["long_side"] = merged[["width", "height"]].max(axis=1)

    return merged

# Utility for qualitative spot checks of extremes (useful for debugging dataset artifacts).
def get_top_bottom_by_column(
    df: pd.DataFrame,
    col: str,
    n: int = 10,
):
    """
    Returns (top_n_df, bottom_n_df) sorted by column.
    """
    tmp = df.copy()
    tmp = tmp.dropna(subset=[col])

    top_n = tmp.sort_values(col, ascending=False).head(n).copy()
    bottom_n = tmp.sort_values(col, ascending=True).head(n).copy()
    return top_n, bottom_n



def alpha_background_table_scene(
    train_df: pd.DataFrame,
    img_dir: Path,
    filename_col: str = "filename",
    alpha0_thresh: float = 0.01,      # >1% transparent => cutout
    min_bg_mean: float = 5.0,         # bg not black
    min_bg_std: float = 10.0,         # bg has texture/variation
) -> pd.DataFrame:
    rows = []
    for _, r in tqdm(train_df.iterrows(), total=len(train_df), desc="EDA alpha/bg scene"):
        fp = img_dir / str(r[filename_col])
        try:
            rgba = Image.open(fp).convert("RGBA")
            arr = np.array(rgba)
            rgb = arr[..., :3].astype(np.float32)
            a   = arr[..., 3]

            alpha0_frac = float((a == 0).mean())
            is_cutout = alpha0_frac > alpha0_thresh

            bg_present_scene = False
            bg_mean = np.nan
            bg_std  = np.nan

            if is_cutout:
                bg = rgb[a == 0]  # Nx3
                if bg.size:
                    bg_mean = float(bg.mean())
                    bg_std  = float(bg.std())  # variation across bg pixels/channels
                    bg_present_scene = (bg_mean >= min_bg_mean) and (bg_std >= min_bg_std)

            label = (
                "no_cutout" if not is_cutout else
                ("cutout_bg_present" if bg_present_scene else "cutout_bg_missing")
            )

            rows.append({
                "filename": fp.name,
                "alpha0_frac": alpha0_frac,
                "label": label,
                "bg_mean": bg_mean,
                "bg_std": bg_std,
            })
        except Exception as e:
            rows.append({
                "filename": fp.name,
                "alpha0_frac": np.nan,
                "label": "error",
                "bg_mean": np.nan,
                "bg_std": np.nan,
                "error": str(e),
            })

    return pd.DataFrame(rows)



def plot_bg_label_counts(alpha_df, save_path):
    """
    Bar chart for alpha/background categories.
    Expects alpha_df to have a 'label' column like:
      - 'no_cutout'
      - 'cutout_bg_present'
      - 'cutout_bg_missing'
      - (optional) 'error'
    """
    counts = alpha_df["label"].value_counts()

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.ylabel("Number of images")
    plt.title("Alpha cutout vs background availability")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()