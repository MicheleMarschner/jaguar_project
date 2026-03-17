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
from jaguar.utils.utils_eda import (
    alpha_background_table_scene,
    analyze_images, 
    basic_integrity_report, 
    check_filename_and_folder_consistency, 
    class_distribution, 
    identity_filter_summary,
    merge_sharpness_with_image_stats,
    get_top_bottom_by_column,
    plot_bg_label_counts,
    plot_identity_distribution, 
    plot_image_dimensions, 
    sharpness_histogramm, 
    plot_resolution_histogram, 
    show_image_gallery
)

# ----------------------------
# EDA Analysis
# ----------------------------

def run_eda(train_file: Path, test_file: Path, save_dir: Path, artifacts_dir: Path) -> None:
    # Load raw metadata tables (train labels + benchmark test pairing file)
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # configs for later
    n_examples = 10
    thresholds = [10, 20, 30, 40, 50]

    # Basic schema / missingness / sample preview sanity checks
    integrity = basic_integrity_report(train_df, test_df)

    # Identity imbalance overview (core issue for re-ID training/splitting)
    counts, class_summary = class_distribution(train_df)
    plot_identity_distribution(
        counts,
        save_path=(save_dir / "identity_distribution.png")
    )

    # "What if we require >= x images per identity?" summary for split design decisions
    summary_df = identity_filter_summary(
        identity_counts=counts,
        thresholds=thresholds,
        out_dir=save_dir,
    )

    # Image-level technical properties (resolution, aspect, color mode)
    img_stats_df = analyze_images(train_df, PATHS.data_train)

    if not img_stats_df.empty:
        plot_image_dimensions(
            img_stats_df,
            save_path=(save_dir / "image_dimensions.png")
        )

    alpha_df = alpha_background_table_scene(train_df, img_dir=PATHS.data_train, filename_col="filename")
    alpha_df.to_csv(save_dir / "alpha_background_table.csv", index=False)
    print(alpha_df["label"].value_counts())

    alpha_df = alpha_background_table_scene(train_df, img_dir=PATHS.data_train, filename_col="filename")
    plot_bg_label_counts(alpha_df, save_dir / "cutout_background_counts.png")

    # CSV ↔ filesystem consistency (catch missing/extra files and naming issues early)
    check_filename_and_folder_consistency(train_df, data_path=PATHS.data_train)

    # Sharpness distribution from upstream dedup/preprocessing artifacts
    sharpness_histogramm(artifacts_dir, save_dir)

    # Merge image stats + sharpness/features to analyze resolution/sharpness jointly
    df_sharp = pd.read_parquet(artifacts_dir / "meta_img_features.parquet").copy()
    merged = merge_sharpness_with_image_stats(
        df_sharp=df_sharp,
        img_stats_df=img_stats_df,
        key="filename",
        validate="one_to_one",
    )

    # Resolution distribution + extreme examples for qualitative inspection
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