from typing import Any
from jaguar.utils.utils import write_json


OUTPUT_PROFILES = {
    "base": {
        "per_run": ["experiment_config", "metrics", "train_history"],
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
    "deduplication": {
        "per_run": ["experiment_config", "metrics", "train_history"],
        "aggregate": ["deduplication_summary"],
    },
    "stat_stability": {
        "per_run": ["experiment_config", "metrics", "train_history"],
        "aggregate": ["stat_stability_summary"],
    },
    "xai": {
        "per_run": ["experiment_config"],
        "aggregate": [],
    },
    
}

def save_experiment_config(*, run_dir, config, **kwargs):
    """Write the resolved experiment configuration to the run directory."""
    write_json(config, run_dir / "experiment_config.json")

def save_metrics(*, run_dir, final_results, **kwargs):
    """Write final evaluation metrics for a run to a JSON file."""
    write_json(final_results, run_dir / "metrics.json")

def save_train_history(*, run_dir, train_history, **kwargs):
    """Write the recorded training history for a run to a JSON file."""
    write_json(train_history, run_dir / "train_history.json")

def save_params_flops(*, run_dir, backbone_stats, **kwargs):
    """Write backbone parameter and FLOP statistics to the run directory."""
    write_json(backbone_stats, run_dir / "params_flops.json")

def save_fusion_components(*, run_dir, ensemble_stats, **kwargs):
    """Write ensemble fusion-component statistics to the run directory."""
    payload = ensemble_stats.get("fusion_components", ensemble_stats)
    write_json(payload, run_dir / "fusion_components.json")

def save_error_overlap(*, run_dir, ensemble_stats, **kwargs):
    """Write ensemble error-overlap statistics to the run directory."""
    payload = ensemble_stats.get("error_overlap", ensemble_stats)
    write_json(payload, run_dir / "error_overlap.json")

def save_timing(*, run_dir, timing_stats, **kwargs):
    """Write runtime timing statistics for a run to a JSON file."""
    write_json(timing_stats, run_dir / "timing.json")

def save_requested_outputs(config: dict, artifacts: dict[str, Any]) -> None:
    """Save all per-run artifacts requested by the configured output profile."""
    output_profile = config.get("output", {}).get("profile")
    requested_outputs = OUTPUT_PROFILES.get(output_profile, {}).get("per_run", [])

    for output_name in requested_outputs:
        writer = OUTPUT_WRITERS.get(output_name)
        if writer is None:
            raise ValueError(f"Unknown output writer: {output_name}")
        writer(**artifacts)


OUTPUT_WRITERS = {
    "experiment_config": save_experiment_config,
    "metrics": save_metrics,
    "train_history": save_train_history,
    "params_flops": save_params_flops,
    "fusion_components": save_fusion_components,
    "error_overlap": save_error_overlap,
    "timing": save_timing
}