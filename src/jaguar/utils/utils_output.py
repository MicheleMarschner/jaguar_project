from pathlib import Path
from typing import Any, Optional
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import json


from jaguar.config import PATHS
from jaguar.utils.utils import ensure_dir, read_json_if_exists


def build_backbone_stats(model, config: dict) -> dict[str, Any]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "backbone_name": config["model"]["backbone_name"],
        "head_type": config["model"]["head_type"],
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "input_size": int(model.backbone_wrapper.input_size),
        "flops": None,
    }


def build_timing_stats(epoch_times: list[float]) -> dict[str, Any]:
    total_train_time_sec = float(sum(epoch_times))
    avg_epoch_time_sec = float(total_train_time_sec / len(epoch_times)) if epoch_times else 0.0

    return {
        "epoch_times_sec": [float(x) for x in epoch_times],
        "total_train_time_sec": total_train_time_sec,
        "avg_epoch_time_sec": avg_epoch_time_sec,
        "num_epochs_recorded": len(epoch_times),
    }


def build_output_artifacts(
    *,
    run_dir: Path,
    config: dict,
    final_results: dict,
    train_history: list[dict],
    model=None,
    epoch_times: list[float] | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "run_dir": run_dir,
        "config": config,
        "final_results": final_results,
        "train_history": train_history,
    }

    if model is not None:
        artifacts["backbone_stats"] = build_backbone_stats(model, config)

    if epoch_times is not None:
        artifacts["timing_stats"] = build_timing_stats(epoch_times)

    return artifacts


def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "experiment_config": read_json_if_exists(run_dir / "experiment_config.json"),
        "metrics": read_json_if_exists(run_dir / "metrics.json"),
        "train_history": read_json_if_exists(run_dir / "train_history.json"),
        "params_flops": read_json_if_exists(run_dir / "params_flops.json"),
        "timing": read_json_if_exists(run_dir / "timing.json"),
    }


def _safe_get(d: Optional[dict], *keys, default=None):
    cur = d
    for k in keys:
        if cur is None or not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_augmentation_summary_row(artifacts: dict[str, Any]) -> dict[str, Any]:
    cfg = artifacts["experiment_config"] or {}
    metrics = artifacts["metrics"] or {}

    return {
        "experiment_name": _safe_get(cfg, "training", "experiment_name"),
        "mAP": _safe_get(metrics, "metrics", "mAP"),
        "rank1": _safe_get(metrics, "metrics", "rank1"),
        "best_epoch": _safe_get(metrics, "best_epoch"),
        "apply_augmentations": _safe_get(cfg, "augmentation", "apply_augmentations"),
        "horizontal_flip": _safe_get(cfg, "augmentation", "horizontal_flip"),
        "affine_degrees": _safe_get(cfg, "augmentation", "affine_degrees"),
        "random_erasing_p": _safe_get(cfg, "augmentation", "random_erasing_p"),
        "color_jitter_brightness": _safe_get(cfg, "augmentation", "color_jitter_brightness"),
        "color_jitter_contrast": _safe_get(cfg, "augmentation", "color_jitter_contrast"),
    }


def build_loss_summary_row(artifacts: dict[str, Any]) -> dict[str, Any]:
    cfg = artifacts["experiment_config"] or {}
    metrics = artifacts["metrics"] or {}

    return {
        "experiment_name": _safe_get(cfg, "training", "experiment_name"),
        "head_type": _safe_get(cfg, "model", "head_type"),
        "s": _safe_get(cfg, "model", "s"),
        "m": _safe_get(cfg, "model", "m"),
        "mAP": _safe_get(metrics, "metrics", "mAP"),
        "rank1": _safe_get(metrics, "metrics", "rank1"),
        "best_epoch": _safe_get(metrics, "best_epoch"),
        "best_score": _safe_get(metrics, "best_score"),
    }


def build_backbone_summary_row(artifacts: dict[str, Any]) -> dict[str, Any]:
    cfg = artifacts["experiment_config"] or {}
    metrics = artifacts["metrics"] or {}
    pf = artifacts["params_flops"] or {}

    return {
        "experiment_name": _safe_get(cfg, "training", "experiment_name"),
        "backbone_name": _safe_get(cfg, "model", "backbone_name"),
        "head_type": _safe_get(cfg, "model", "head_type"),
        "mAP": _safe_get(metrics, "metrics", "mAP"),
        "rank1": _safe_get(metrics, "metrics", "rank1"),
        "best_epoch": _safe_get(metrics, "best_epoch"),
        "best_score": _safe_get(metrics, "best_score"),
        "total_params": pf.get("total_params"),
        "trainable_params": pf.get("trainable_params"),
        "input_size": pf.get("input_size"),
        "flops": pf.get("flops"),
    }


def build_optim_sched_summary_row(artifacts: dict[str, Any]) -> dict[str, Any]:
    cfg = artifacts["experiment_config"] or {}
    metrics = artifacts["metrics"] or {}
    timing = artifacts["timing"] or {}

    return {
        "experiment_name": _safe_get(cfg, "training", "experiment_name"),
        "optimizer_type": _safe_get(cfg, "optimizer", "type"),
        "scheduler_type": _safe_get(cfg, "scheduler", "type"),
        "optimizer_lr": _safe_get(cfg, "optimizer", "lr"),
        "mAP": _safe_get(metrics, "metrics", "mAP"),
        "rank1": _safe_get(metrics, "metrics", "rank1"),
        "best_epoch": _safe_get(metrics, "best_epoch"),
        "best_score": _safe_get(metrics, "best_score"),
        "total_train_time_sec": timing.get("total_train_time_sec"),
        "avg_epoch_time_sec": timing.get("avg_epoch_time_sec"),
        "num_epochs_recorded": timing.get("num_epochs_recorded"),
    }


SUMMARY_BUILDERS = {
    "augmentation": build_augmentation_summary_row,
    "loss": build_loss_summary_row,
    "backbone": build_backbone_summary_row,
    "optim_sched": build_optim_sched_summary_row,
}


SUMMARY_FILENAMES = {
    "augmentation": "augmentation_summary.csv",
    "loss": "loss_summary.csv",
    "backbone": "backbone_summary.csv",
    "optim_sched": "optim_sched_summary.csv",
}

def aggregate_experiment_outputs(experiment_group: str, output_profile: str) -> Path:
    builder = SUMMARY_BUILDERS.get(output_profile)
    if builder is None:
        raise ValueError(f"No summary builder for output_profile={output_profile}")

    run_root = PATHS.runs / experiment_group
    summary_root = PATHS.results / experiment_group
    ensure_dir(summary_root)

    rows: list[dict[str, Any]] = []

    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        artifacts = load_run_artifacts(run_dir)
        if artifacts["metrics"] is None:
            continue
        rows.append(builder(artifacts))

    df = pd.DataFrame(rows)
    out_path = summary_root / SUMMARY_FILENAMES[output_profile]
    df.to_csv(out_path, index=False)
    return out_path