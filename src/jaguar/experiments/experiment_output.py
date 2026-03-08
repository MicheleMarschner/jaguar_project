import argparse

from jaguar.utils.utils import write_json
from jaguar.utils.utils_output import aggregate_experiment_outputs


OUTPUT_PROFILES = {
    "base": {
        "per_run": ["experiment_config", "metrics"],
        "aggregate": ["base_summary"],
    },
    "augmentation": {
        "per_run": ["experiment_config", "metrics"],
        "aggregate": ["augmentation_summary"],
    },
    "backbone": {
        "per_run": ["experiment_config", "metrics", "train_history", "params_flops"],
        "aggregate": ["backbone_summary"],
    },
    "ensemble": {
        "per_run": ["experiment_config", "metrics", "fusion_components", "error_overlap"],
        "aggregate": ["ensemble_summary"],
    },
    "loss": {
        "per_run": ["experiment_config", "metrics", "train_history"],
        "aggregate": ["loss_summary"],
    },
    "optim_sched": {
        "per_run": ["experiment_config", "metrics", "train_history", "timing"],
        "aggregate": ["optim_sched_summary"],
    },
    "resizing": {
        "per_run": ["experiment_config", "metrics", "train_history"],
        "aggregate": ["resizing_summary"],
    },
    "background": {
        "per_run": ["experiment_config", "metrics"],
        "aggregate": ["background_summary"],
    },
    "deduplication": {
        "per_run": [],
        "aggregate": ["deduplication_summary"],
    },
    "stat_stability": {
        "per_run": ["experiment_config", "metrics", "train_history"],
        "aggregate": ["stat_stability_summary"],
    },
    "viewpoint": {
        "per_run": [],
        "aggregate": ["viewpoint_summary"],
    },
    "xai": {
        "per_run": ["experiment_config"],
        "aggregate": [],
    },
    
}


def save_experiment_config(*, run_dir, config, **kwargs):
    write_json(config, run_dir / "experiment_config.json")

def save_metrics(*, run_dir, final_results, **kwargs):
    write_json(final_results, run_dir / "metrics.json")

def save_train_history(*, run_dir, train_history, **kwargs):
    write_json(train_history, run_dir / "train_history.json")

def save_params_flops(*, run_dir, backbone_stats, **kwargs):
    write_json(backbone_stats, run_dir / "params_flops.json")

def save_fusion_components(*, run_dir, ensemble_stats, **kwargs):
    write_json(ensemble_stats, run_dir / "fusion_components.json")

def save_error_overlap(*, run_dir, ensemble_stats, **kwargs):
    write_json(ensemble_stats, run_dir / "error_overlap.json")

def save_timing(*, run_dir, timing_stats, **kwargs):
    write_json(timing_stats, run_dir / "timing.json")



OUTPUT_WRITERS = {
    "experiment_config": save_experiment_config,
    "metrics": save_metrics,
    "train_history": save_train_history,
    "params_flops": save_params_flops,
    "fusion_components": save_fusion_components,
    "error_overlap": save_error_overlap,
    "timing": save_timing
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate experiment outputs into summary tables")
    parser.add_argument("--experiment_group", type=str, required=True)
    parser.add_argument("--output_profile", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    out_path = aggregate_experiment_outputs(
        experiment_group=args.experiment_group,
        output_profile=args.output_profile,
    )
    print(f"[AGGREGATION] Saved summary to: {out_path}")


if __name__ == "__main__":
    main()