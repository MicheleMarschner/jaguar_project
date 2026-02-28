"""
Initial dataset EDA for the Jaguar re-ID project.

Project role:
- validates raw train/test CSVs and train image folder consistency
- computes image-level statistics (dimensions, modes, sharpness-derived merges)
- inspects identity imbalance and threshold-based filtering scenarios
- saves reusable EDA artifacts/plots for later split design and preprocessing decisions

This script is analysis-only (no training / no model evaluation).
"""

from pathlib import Path
import pandas as pd

from jaguar.config import PATHS
from jaguar.datasets.FiftyOneDataset import FODataset, get_or_create_manifest_dataset
from jaguar.utils.utils_eda import (
    analyze_images, 
    basic_integrity_report, 
    check_filename_and_folder_consistency, 
    class_distribution, 
    identity_filter_summary,
    merge_sharpness_with_image_stats,
    get_top_bottom_by_column
)
from jaguar.utils.utils_visualization import (
    plot_identity_distribution, 
    plot_image_dimensions, 
    sharpness_histogramm, 
    plot_resolution_histogram, 
    show_image_gallery
)

FO_DATASET_NAME = "jaguar_init"

# ----------------------------
# EDA Analysis
# ----------------------------

def run_eda(data_path: Path, train_file: Path, test_file: Path) -> None:
    # Load raw metadata tables (train labels + benchmark test pairing file)
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    artifacts_dir = PATHS.runs / "deduplication"   # upstream image-feature artifacts (e.g., sharpness)
    save_dir = PATHS.results / "eda"               # EDA outputs used for reporting + later decisions

    # configs for later
    n_examples = 10
    thresholds = [10, 20, 30, 40, 50]

    # 1) Basic schema / missingness / sample preview sanity checks
    integrity = basic_integrity_report(train_df, test_df)

    # 2) Identity imbalance overview (core issue for re-ID training/splitting)
    counts, class_summary = class_distribution(train_df)
    plot_identity_distribution(
        counts,
        save_path=(save_dir / "identity_distribution.png")
    )

    # 3) "What if we require >= x images per identity?" summary for split design decisions
    summary_df = identity_filter_summary(
        identity_counts=counts,
        thresholds=thresholds,
        out_dir=save_dir,
    )

    # 4) Image-level technical properties (resolution, aspect, color mode)
    img_stats_df = analyze_images(train_df, PATHS.data_train)

    if not img_stats_df.empty:
        plot_image_dimensions(
            img_stats_df,
            save_path=(save_dir / "image_dimensions.png")
        )

    # 5) CSV ↔ filesystem consistency (catch missing/extra files and naming issues early)
    check_filename_and_folder_consistency(train_df, data_path=PATHS.data_train)

    # 6) Sharpness distribution from upstream dedup/preprocessing artifacts
    sharpness_histogramm(artifacts_dir, save_dir)

    # 7) Merge image stats + sharpness/features to analyze resolution/sharpness jointly
    df_sharp = pd.read_parquet(artifacts_dir / "meta_img_features.parquet").copy()
    merged = merge_sharpness_with_image_stats(
        df_sharp=df_sharp,
        img_stats_df=img_stats_df,
        key="filename",
        validate="one_to_one",
    )

    # 8) Resolution distribution + extreme examples for qualitative inspection
    plot_resolution_histogram(
        values=merged["resolution_px"],
        title="Image Resolution Histogram (log x-scale, log-spaced bins)",
        xlabel="Resolution (pixels, width × height)",
        save_path=save_dir / "resolution_histogram_log.png",
        bins_n=50,
        add_quantile_lines=False,
    )

    top_res, low_res = get_top_bottom_by_column(merged, col="resolution_px", n=n_examples)

    show_image_gallery(
        top_res,
        image_root=PATHS.data_train,
        title=f"Top {n_examples} Highest Resolution Images",
        save_path=save_dir / f"top{n_examples}_highest_resolution_gallery.png",
    )

    show_image_gallery(
        low_res,
        image_root=PATHS.data_train,
        title=f"Top {n_examples} Lowest Resolution Images",
        save_path=save_dir / f"top{n_examples}_lowest_resolution_gallery.png",
    )


def build_from_csv_labels(
    dataset_name: str,
    train_dir: Path,
    csv_path: Path,
    overwrite_db: bool = True,
) -> FODataset:
    """
    Build a FiftyOne dataset from the raw training CSV.

    Project role:
    - creates a labeled visual dataset for inspection/EDA in FiftyOne
    - stores train split tag + filename metadata on each sample
    - acts as a bridge from Kaggle-style CSV labels to project-internal dataset tooling
    """
    df = pd.read_csv(csv_path)
    
    print(train_dir, csv_path)

    # basic validation
    assert {"filename", "ground_truth"}.issubset(df.columns), f"CSV columns are {list(df.columns)}"
    assert df["filename"].nunique() == len(df), "Duplicate filenames in CSV"

    fo_wrapper = FODataset(dataset_name=dataset_name, overwrite=overwrite_db)

    samples = []
    missing = 0

    for _, r in df.iterrows():
        p = train_dir / str(r["filename"])
        if not p.exists():
            missing += 1
            continue
        
        label = str(r["ground_truth"])
        s = fo_wrapper.create_sample(filepath=p, label=label, tags=["train"])
        print(s)
        s["split"] = "train"
        s["filename"] = p.name
        samples.append(s)

    if not samples:
        raise RuntimeError("No samples created. Check train_dir and csv filenames.")

    fo_wrapper.add_samples(samples)
    print(f"Built FO dataset with {len(samples)} samples. Missing files: {missing}")
    return fo_wrapper


def main():
    manifest_dir = PATHS.data_export / "init"
    csv_file = PATHS.data / "raw/jaguar-re-id/train.csv"
    test_csv_file = PATHS.data / "raw/jaguar-re-id/test.csv"
    
    
    ### add labels to fiftyOne
    def build_fn():
        return build_from_csv_labels(
            dataset_name=FO_DATASET_NAME,
            train_dir=PATHS.data_train,
            csv_path=csv_file,
            overwrite_db=True,
        )
    '''
    fo_wrapper = get_or_create_manifest_dataset(
        dataset_name=FO_DATASET_NAME,
        manifest_dir=manifest_dir,
        build_fn=build_fn,
        overwrite_load=False,
    )
    '''
    run_eda(PATHS.data_train, train_file=csv_file, test_file=test_csv_file)

    

if __name__ == "__main__":
    main()