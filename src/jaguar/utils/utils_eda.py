from pathlib import Path
import re
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import math

from jaguar.utils.utils import ensure_dir
from typing import Iterable, Optional

sns.set_theme(style="whitegrid", palette="muted")


def print_section(title: str) -> None:
    """Print a formatted section header for console output."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def basic_integrity_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Print basic schema and missing-value checks for train and test tables."""
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


def class_distribution(train_df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Summarize the number of images per identity in the training set."""
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
    """Create a summary table for different minimum-images-per-identity thresholds."""
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


def analyze_images(
    df: pd.DataFrame,
    img_dir: Path,
    img_col: str = "filename",
) -> pd.DataFrame:
    """Read image size and mode metadata for all files listed in the dataframe."""
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
    """Check filename format, index continuity, and CSV-to-folder consistency."""
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


def merge_sharpness_with_image_stats(
    df_sharp: pd.DataFrame,
    img_stats_df: pd.DataFrame,
    key: str = "filename",
    validate: str = "one_to_one",
) -> pd.DataFrame:
    """Merge sharpness features with image metadata and add derived size columns."""
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


def get_top_bottom_by_column(
    df: pd.DataFrame,
    col: str,
    n: int = 10,
):
    """Return the top-n and bottom-n rows sorted by a given column."""
    tmp = df.copy()
    tmp = tmp.dropna(subset=[col])

    top_n = tmp.sort_values(col, ascending=False).head(n).copy()
    bottom_n = tmp.sort_values(col, ascending=True).head(n).copy()
    return top_n, bottom_n


def alpha_background_table_scene(
    train_df: pd.DataFrame,
    img_dir: Path,
    filename_col: str = "filename",
    alpha0_thresh: float = 0.01,
    min_bg_mean: float = 5.0,
    min_bg_std: float = 10.0,
) -> pd.DataFrame:
    """Classify images by alpha cutout presence and visible background content."""
    rows = []
    for _, r in tqdm(train_df.iterrows(), total=len(train_df), desc="EDA alpha/bg scene"):
        fp = img_dir / str(r[filename_col])
        try:
            rgba = Image.open(fp).convert("RGBA")
            arr = np.array(rgba)
            rgb = arr[..., :3].astype(np.float32)
            a = arr[..., 3]

            alpha0_frac = float((a == 0).mean())
            is_cutout = alpha0_frac > alpha0_thresh

            bg_present_scene = False
            bg_mean = np.nan
            bg_std = np.nan

            if is_cutout:
                bg = rgb[a == 0]
                if bg.size:
                    bg_mean = float(bg.mean())
                    bg_std = float(bg.std())
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
    """Plot the number of images in each alpha/background label category."""
    counts = alpha_df["label"].value_counts()

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.ylabel("Number of images")
    plt.title("Alpha cutout vs background availability")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# EDA Plots
# ============================================================

def plot_image_dimensions(stats_df: pd.DataFrame, save_path: Optional[Path] = None, show: bool = False) -> None:
    fig = plt.figure(figsize=(8, 4))
    plt.hist(stats_df["width"], bins=50, alpha=0.6, label="width")
    plt.hist(stats_df["height"], bins=50, alpha=0.6, label="height")
    plt.legend()
    plt.title("Distribution of image widths and heights (train)")
    plt.xlabel("Pixels")
    plt.ylabel("Count")
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# Left: per-identity counts (sorted) to reveal long-tail imbalance.
# Right: histogram of class sizes to summarize the distribution shape.
def plot_identity_distribution(counts: pd.Series, save_path: Path) -> None:
    plt.figure(figsize=(12, 6))

    # Bar chart
    ax = plt.subplot(1, 2, 1)
    color_palette = ['red' if x < 20 else 'skyblue' for x in counts.values]
    sns.barplot(x=counts.values, y=counts.index, palette=color_palette, ax=ax)

    ax.set_title("Training Data: Image Count per Jaguar", fontsize=14)
    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Class index (sorted)")
    ax.tick_params(axis='y', rotation=30)

    # for horizontal bars, median count should be a VERTICAL line (x-axis), not hline
    ax.axvline(x=counts.median(), color='red', linestyle='--', label=f'Median: {counts.median():.1f}')
    for p in ax.patches:
        w = p.get_width()   # horizontal bar length
        y = p.get_y() + p.get_height() / 2
        ax.annotate(
            f"{int(w)}",
            (w, y),
            ha="left",
            va="center",
            fontsize=8,
            xytext=(3, 0),
            textcoords="offset points",
        )
    ax.legend()

    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(counts.values, bins=15, kde=True, color="darkgreen", alpha=0.5)
    plt.title("Distribution of Class Sizes", fontsize=14)
    plt.xlabel("Images per Jaguar")
    plt.ylabel("Count")

    plt.tight_layout()

    ensure_dir(save_path.parent)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


# Sharpness spans orders of magnitude, so we use a log-scaled x-axis and log-spaced bins.
def sharpness_histogramm(artifacts_dir, save_dir, filename="sharpness_histogram.png"):
    df = pd.read_parquet(artifacts_dir / "meta_img_features.parquet").copy()

    x = df["sharpness"].dropna()
    x = x[x >= 0]  
    median = x.median()
    p95 = x.quantile(0.95)

    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 70)  

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(x, bins=bins, kde=False, ax=ax)

    ax.set_xscale("log")
    ax.set_title("Sharpness Histogram (log x-scale)")
    ax.set_xlabel("Sharpness (log scale)")
    ax.set_ylabel("Count")

    ax.axvline(median, color="green", linestyle="--", linewidth=2, label=f"median={median:.1f}")
    ax.axvline(p95, color="green", linestyle=":", linewidth=2, label=f"p95={p95:.1f}")
    ax.legend(frameon=False)

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:g}"))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 3, 5)))
    ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:g}"))

    ax.tick_params(axis="x", which="major", labelsize=10, length=4)
    ax.tick_params(axis="x", which="minor", labelsize=10, length=2)

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_dir/filename, dpi=200, bbox_inches="tight")


# Resolution also spans a wide range; log-x avoids compressing low-resolution bins.
def plot_resolution_histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    save_path: str | Path | None = None,
    bins_n: int = 50,
    add_quantile_lines: bool = False,
):
    """
    Log-x histogram with log-spaced bins.
    """
    x = pd.to_numeric(values, errors="coerce").dropna()
    x = x[x > 0]
    if len(x) == 0:
        raise ValueError("No positive values to plot on log scale.")

    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), bins_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(x, bins=bins, kde=False, ax=ax)

    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    # log ticks
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, pos: f"{int(v):,}" if v >= 1 else f"{v:g}")
    )
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 3, 5)))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.grid(False)

    if add_quantile_lines:
        median = float(x.median())
        p95 = float(x.quantile(0.95))
        ax.axvline(median, color="green", linestyle="--", linewidth=2, label=f"median={median:.1f}")
        ax.axvline(p95, color="green", linestyle=":", linewidth=2, label=f"p95={p95:.1f}")
        ax.legend(frameon=False)

    plt.tight_layout()

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[Plot] Saved to: {save_path}")

    return fig, ax


# Qualitative inspection helper: visualize extreme examples with identity + size metadata.
def show_image_gallery(
    df_subset: pd.DataFrame,
    image_root: str | Path | None = None,   # NEW
    title: str = "",
    n_cols: int = 5,
    figsize_per_img: float = 3.2,
    save_path: str | Path | None = None,
):
    df_subset = df_subset.reset_index(drop=True)
    n_rows = math.ceil(len(df_subset) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_img, n_rows * figsize_per_img))

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for ax in axes[len(df_subset):]:
        ax.axis("off")

    for i, (_, row) in enumerate(df_subset.iterrows()):
        ax = axes[i]
        fname = str(row.get("filename"))
        fp = image_root / fname

        with Image.open(fp) as img:
            img = img.convert("RGB")
            ax.imshow(img)

        fname = str(row.get("filename", fp.name))
        jag_id = str(row.get("identity_id", "NA"))

        title_line = jag_id
        title_line += f" | {int(row['width'])}×{int(row['height'])}"
        title_line += f" | {int(row['resolution_px']):,} px"

        ax.set_title(title_line, fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.88, wspace=0.05, hspace=0.35)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[Gallery] Saved to: {save_path}")

