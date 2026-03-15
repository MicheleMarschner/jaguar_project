from pathlib import Path

from jaguar.analysis.baseline_and_eda.baseline import run_baseline_analysis
from jaguar.analysis.baseline_and_eda.eda import run_eda
from jaguar.config import PATHS


def run(
    config: dict, 
    save_dir: Path,
    root_dir: Path | None = None, 
    run_dir: Path | None = None, 
    **kwargs
) -> None:
    train_file = PATHS.data / "jaguar-re-id/train.csv"
    test_file = PATHS.data / "jaguar-re-id/test.csv"
    burst_dir = PATHS.runs / "bursts"
    
    model_label = config["model"]["backbone_name"]
    train_dir = root_dir / "baseline_init"

    #run_eda(train_file, test_file, save_dir, artifacts_dir=burst_dir)
    run_baseline_analysis(train_dir, save_dir, model_label)


    