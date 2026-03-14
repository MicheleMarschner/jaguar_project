


from jaguar.analysis.E02_kaggle_deduplication.burst_analysis import run_burst_analysis
from jaguar.analysis.E02_kaggle_deduplication.split_analysis import run_split_diagnostics
from jaguar.analysis.E02_kaggle_deduplication.training_impact_analysis import run_duplicate_impact_report
from jaguar.config import PATHS


if __name__ == "__main__":
    split_artifacts_dir = PATHS.runs / "closed_set__dupFalse__kTrain3__kVal3__p4"
    burst_artifacts_dir = PATHS.runs / "burst_groups__within500__cross10000__ph11"
    training_curated_dir = PATHS.runs / "kaggle_deduplication" / "closed_curated_traink_3_valk_3_p4"
    training_full_dir = PATHS.runs / "kaggle_deduplication" / "closed_keep_all"
    manifest_dir = PATHS.data_export / "splits_curated"
    save_dir = PATHS.results
    img_root = PATHS.data_train


    run_burst_analysis(burst_artifacts_dir, save_dir, img_root)
    run_split_diagnostics(split_artifacts_dir, save_dir, img_root, manifest_dir, dataset_name="jaguar_curated")
    outputs = run_duplicate_impact_report(
        full_run_dir=training_full_dir,
        curated_run_dir=training_curated_dir,
        save_dir=save_dir,
        make_overlay_plot=False,
    )

    for name, path in outputs.items():
        print(f"{name}: {path}")