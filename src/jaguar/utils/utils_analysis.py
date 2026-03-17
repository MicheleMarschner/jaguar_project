from __future__ import annotations

from pathlib import Path

from jaguar.config import PATHS
from jaguar.analysis.baseline_and_eda import run_analysis as baseline_and_eda_analysis
from jaguar.analysis.eda_background_intervention import (
    background_intervention_analysis,
)
from jaguar.analysis.eda_foreground_contribution import (
    foreground_contribution_analysis,
)
from jaguar.analysis.eda_xai_class_attribution import (
    xai_class_attribution_analysis,
)
from jaguar.analysis.eda_xai_similarity import xai_similarity_analysis
from jaguar.analysis.kaggle_deduplication import run_analysis as kaggle_deduplication_analysis
from jaguar.analysis.kaggle_ensemble import ensemble_analysis
from jaguar.utils.utils import read_json_if_exists


REGISTRY = {
    "baseline": baseline_and_eda_analysis.run,
    "kaggle_deduplication": kaggle_deduplication_analysis.run,          
    "kaggle_ensemble": ensemble_analysis.run,
    "eda_background_intervention": background_intervention_analysis.run,
    "eda_foreground_contribution": foreground_contribution_analysis.run,
    "eda_xai_class_attribution": xai_class_attribution_analysis.run,
    "eda_xai_similarity": xai_similarity_analysis.run,
}


def load_run_config(run_dir: Path, config_name: str = "experiment_config.json") -> dict:
    """Load a stored run config from disk."""
    config_path = run_dir / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    return read_json_if_exists(config_path)


def find_run_dirs(root_dir: Path, config_name: str) -> list[Path]:
    """Find run directories containing the given config file."""
    return sorted(
        config_path.parent
        for config_path in root_dir.rglob(config_name)
        if config_path.is_file()
    )


def resolve_experiment_group(config: dict) -> str:
    """Validate and return the configured experiment group."""
    experiment_group = config.get("output", {}).get("experiment_group")
    if not experiment_group:
        raise KeyError("Missing output.experiment_group in stored config.json")
    if experiment_group not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Unknown output.experiment_group='{experiment_group}'. Known groups: {known}"
        )
    return experiment_group


def build_results_out_dir(experiment_group: str, run_name: str | None = None) -> Path:
    """Create and return the results output directory."""
    if run_name is not None:
        out_dir = PATHS.results / experiment_group / run_name
    else:
        out_dir = PATHS.results / experiment_group
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir