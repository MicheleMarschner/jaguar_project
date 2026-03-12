from pathlib import Path
from typing import Any
import pandas as pd
from matplotlib.figure import Figure

import wandb
from wandb.sdk.wandb_run import Run


def _build_wandb_tags(config: dict[str, Any], experiment_group: str | None) -> list[str]:
    """Build a small set of tags for filtering runs in W&B."""
    tags: list[str] = ["train"]

    output_profile = config.get("output", {}).get("profile")
    backbone_name = config.get("model", {}).get("backbone_name")
    split_strategy = config.get("split", {}).get("strategy")

    if experiment_group:
        tags.append(str(experiment_group))
    if output_profile:
        tags.append(str(output_profile))
    if backbone_name:
        tags.append(str(backbone_name))
    if split_strategy:
        tags.append(str(split_strategy))

    return tags


def is_wandb_enabled(config: dict[str, Any]) -> bool:
    """Return whether W&B logging is enabled in config."""
    return bool(config.get("logging", {}).get("enabled", False))


def init_wandb_run(
    config: dict[str, Any],
    run_dir: Path,
    exp_name: str,
    experiment_group: str | None = None,
    job_type: str | None = "train",
) -> Run | None:
    """Initialize a W&B run for one training experiment."""
    if not is_wandb_enabled(config):
        return None

    logging_cfg = config.get("logging", {})
    project = logging_cfg.get("project", "jaguar-reid")

    tags = _build_wandb_tags(config, experiment_group)

    run = wandb.init(
        entity="michele-marschner-university-of-potsdam",
        project=project,
        group=experiment_group,
        job_type=job_type,
        tags=tags,
        name=exp_name,
        config=config,
        dir=str(run_dir),
    )

    run.config.update(
        {
            "experiment_group": experiment_group,
            "output_profile": config.get("output", {}).get("profile"),
            "backbone_name": config.get("model", {}).get("backbone_name"),
            "head_type": config.get("model", {}).get("head_type"),
            "split_strategy": config.get("split", {}).get("strategy"),
            "include_duplicates": config.get("split", {}).get("include_duplicates"),
            "train_k": config.get("curation", {}).get("train_k"),
            "val_k": config.get("curation", {}).get("val_k"),
            "phash_threshold": config.get("curation", {}).get("phash_threshold"),
            "seed": config.get("training", {}).get("seed"),
        },
        allow_val_change=True,
    )

    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("timing/*", step_metric="epoch")
    run.define_metric("meta/*", step_metric="epoch")

    return run


def log_wandb_dataset_info(
    run: Run | None,
    run_dir: Path,
    parquet_root: Path,
    train_size: int,
    val_size: int,
    num_classes: int,
    device: Any,
) -> None:
    """Log resolved dataset and runtime metadata once per run."""
    if run is None:
        return

    run.config.update(
        {
            "run_dir": str(run_dir),
            "resolved_split_data_path": str(parquet_root),
            "n_train_samples": int(train_size),
            "n_val_samples": int(val_size),
            "num_classes": int(num_classes),
            "device": str(device),
        },
        allow_val_change=True,
    )


def log_wandb_model_info(
    run: Run | None,
    model: Any,
) -> None:
    """Log static model information once per run."""
    if run is None:
        return

    num_parameters = sum(p.numel() for p in model.parameters())
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    run.config.update(
        {
            "num_parameters": int(num_parameters),
            "num_trainable_parameters": int(num_trainable_parameters),
        },
        allow_val_change=True,
    )


def log_wandb_epoch_metrics(
    run: Run | None,
    epoch: int,
    avg_loss: float,
    metrics: dict[str, Any],
    current_lr: float,
    epoch_time_sec: float,
    input_size: int | tuple[int, int] | list[int],
) -> None:
    """Log one epoch of train/validation metrics."""
    if run is None:
        return

    run.log(
        {
            "epoch": int(epoch),
            "train/loss": float(avg_loss),
            "val/mAP": float(metrics["mAP"]),
            "val/pairwise_AP": float(metrics["pairwise_AP"]),
            "val/rank1": float(metrics["rank1"]),
            "val/sim_gap": float(metrics["sim_gap"]),
            "meta/lr": float(current_lr),
            "timing/epoch_time_sec": float(epoch_time_sec),
            "meta/input_size": input_size,
        }
    )


def finish_wandb_run(
    run: Run | None,
    best_epoch: int | None,
    best_score: float,
    monitor_metric: str,
    best_metrics: dict[str, Any] | None,
    epochs_completed: int,
    total_train_time_sec: float,
) -> None:
    """Write final summary fields and finish the W&B run."""
    if run is None:
        return

    run.summary["best_epoch"] = best_epoch
    run.summary["best_score"] = float(best_score)
    run.summary["monitor_metric"] = monitor_metric
    run.summary["epochs_completed"] = int(epochs_completed)
    run.summary["total_train_time_sec"] = float(total_train_time_sec)

    if best_metrics is not None:
        run.summary["best_mAP"] = float(best_metrics["mAP"])
        run.summary["best_pairwise_AP"] = float(best_metrics["pairwise_AP"])
        run.summary["best_rank1"] = float(best_metrics["rank1"])
        run.summary["best_sim_gap"] = float(best_metrics["sim_gap"])

    run.finish()


def log_wandb_output_artifact(
    run: Run | None,
    run_dir: Path,
    artifact_name: str,
) -> None:
    """Log selected run output files as a W&B artifact."""
    if run is None:
        return

    artifact = wandb.Artifact(name=artifact_name, type="run_output")

    for filename in [
        "experiment_config.json",
        "metrics.json",
        "train_history.json",
    ]:
        file_path = run_dir / filename
        if file_path.exists():
            artifact.add_file(local_path=str(file_path), name=filename)

    run.log_artifact(artifact)


def log_wandb_checkpoint_artifact(
    run: Run | None,
    checkpoint_path: Path,
    artifact_name: str,
    metadata: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
) -> None:
    """Log one checkpoint file as a W&B model artifact."""
    if run is None or not checkpoint_path.exists():
        return

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=metadata or {},
    )
    artifact.add_file(local_path=str(checkpoint_path), name=checkpoint_path.name)
    run.log_artifact(artifact, aliases=aliases or ["best"])


def log_wandb_table(
    run: Run | None,
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    """Log a pandas DataFrame as a W&B table."""
    if run is None:
        return

    run.log({name: wandb.Table(dataframe=dataframe)})


def log_wandb_image(
    run: Run | None,
    name: str,
    image_path: Path,
    caption: str | None = None,
) -> None:
    """Log an image file to W&B."""
    if run is None or not image_path.exists():
        return

    run.log({name: wandb.Image(str(image_path), caption=caption)})


def log_wandb_matplotlib_figure(
    run: Run | None,
    name: str,
    fig: Figure,
    close: bool = True,
) -> None:
    """Log a matplotlib figure to W&B."""
    if run is None:
        return

    run.log({name: wandb.Image(fig)})

    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)