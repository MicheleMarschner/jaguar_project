from pathlib import Path
from typing import Any


def build_backbone_stats(model, config: dict) -> dict[str, Any]:
    """
    Collect core backbone and head statistics for reporting and comparison across runs.
    """
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
    """
    Summarize per-epoch training times into total, average, and count statistics.
    """
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
    """
    Assemble the standard output artifact bundle for one training run.
    """
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