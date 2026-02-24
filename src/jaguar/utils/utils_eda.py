from pathlib import Path
import re
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Iterable


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


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

# ----------------------------
# EDA: image properties
# ----------------------------

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


