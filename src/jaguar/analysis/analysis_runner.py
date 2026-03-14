from __future__ import annotations

import argparse
import json
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

"""
"kaggle_backbone":,
"kaggle_augmentation": ,
"kaggle_losses": ,
"kaggle_optim_and_sched":,
"kaggle_resizing":,
"kaggle_stat_stability":,
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_group",
        type=str,
        required=True,
        help="Experiment group / folder name under PATHS.runs",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name. If omitted, the whole experiment group is analyzed.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="experiment_config.json",
        help="Stored final config filename inside a run directory.",
    )
    return parser.parse_args()


def load_run_config(run_dir: Path, config_name: str = "experiment_config.json") -> dict:
    config_path = run_dir / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    return read_json_if_exists(config_path)


def find_run_dirs(root_dir: Path, config_name: str) -> list[Path]:
    return sorted(
        [
            p for p in root_dir.iterdir()
            if p.is_dir() and (p / config_name).exists()
        ]
    )


def resolve_experiment_group(config: dict) -> str:
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
    if run_name is not None:
        out_dir = PATHS.results / experiment_group / run_name
    else:
        out_dir = PATHS.results / experiment_group
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    args = parse_args()

    root_dir = PATHS.runs / args.experiment_group
    if not root_dir.exists():
        raise FileNotFoundError(f"Experiment group directory not found: {root_dir}")

    # Single-run mode
    if args.run_name is not None:
        run_dir = root_dir / args.run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        config = load_run_config(run_dir, args.config_name)
        experiment_group = resolve_experiment_group(config)
        save_dir = build_results_out_dir(experiment_group, args.run_name)
        REGISTRY[experiment_group](config=config, run_dir=run_dir, save_dir=save_dir)
        return

   # Group / aggregate mode
    candidate_run_dirs = find_run_dirs(root_dir, args.config_name)
    if not candidate_run_dirs:
        raise FileNotFoundError(
            f"No run directories with {args.config_name} found in: {root_dir}"
        )

    run_dir = candidate_run_dirs[0]
    config = load_run_config(run_dir, args.config_name)

    experiment_group = resolve_experiment_group(config)
    save_dir = build_results_out_dir(experiment_group)

    #!TODO wie kann man dass auslagern??
    if experiment_group == "kaggle_deduplication":
        run_dir = root_dir / "closed_curated_traink_3_valk_3_p4"

    REGISTRY[experiment_group](
        config=config,
        root_dir=root_dir,
        run_dir=run_dir,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()