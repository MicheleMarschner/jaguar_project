from __future__ import annotations
import argparse

from jaguar.config import PATHS
from jaguar.utils.utils_analysis import REGISTRY, build_results_out_dir, find_run_dirs, load_run_config, resolve_experiment_group


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