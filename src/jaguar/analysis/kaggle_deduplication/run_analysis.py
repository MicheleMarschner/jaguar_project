


from pathlib import Path

from jaguar.analysis.kaggle_deduplication.burst_analysis import run_burst_analysis
from jaguar.analysis.kaggle_deduplication.split_analysis import run_split_diagnostics
from jaguar.analysis.kaggle_deduplication.training_impact_analysis import run_duplicate_impact_report
from jaguar.config import PATHS


def run(
    config: dict, 
    save_dir: Path, 
    root_dir: Path | None = None, 
    run_dir: Path | None = None, 
    **kwargs
) -> None:
    training_curated_dir = run_dir
    training_full_dir = root_dir / "closed_keep_all"

    curated_split_path = config["data"]["split_data_path"]
    curated_artifacts_dir = PATHS.runs / curated_split_path

    burst_artifacts_dir = PATHS.runs / "bursts/burst_groups__within500__cross10000__ph11"
    manifest_dir = PATHS.data_export / "splits_curated"
    img_root = PATHS.data_train
    use_fiftyone = config["data"].get("use_fiftyone", False)


    run_burst_analysis(burst_artifacts_dir, save_dir, img_root)
    run_split_diagnostics(
        curated_file_path=curated_artifacts_dir, 
        save_dir=save_dir, 
        img_root=img_root, 
        manifest_dir=manifest_dir, 
        dataset_name="jaguar_curated",
        use_fiftyone=use_fiftyone
    )
        
    outputs = run_duplicate_impact_report(
        full_run_dir=training_full_dir,
        curated_run_dir=training_curated_dir,
        save_dir=save_dir,
        make_overlay_plot=False,
    )

    for name, path in outputs.items():
        print(f"{name}: {path}")