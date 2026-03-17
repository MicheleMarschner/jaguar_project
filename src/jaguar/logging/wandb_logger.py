import os
from pathlib import Path
from typing import Any
import pandas as pd
from matplotlib.figure import Figure
from dotenv import load_dotenv
import wandb
from wandb.sdk.wandb_run import Run

from jaguar.config import ROUND

load_dotenv()

def _build_wandb_tags(
    config: dict[str, Any],
    experiment_group: str | None,
    job_type: str | None = None,
) -> list[str]:
    """Build a compact list of W&B tags from the experiment configuration."""
    tags: list[str] = [str(job_type or "run")]

    output_profile = (
        config.get("output", {}).get("profile")
        or config.get("experiment", {}).get("output_profile")
    )
    backbone_name = config.get("model", {}).get("backbone_name")
    split_strategy = config.get("split", {}).get("strategy")

    eval_type = config.get("experiment", {}).get("eval_type")
    explain_type = config.get("experiment", {}).get("explain_type")
    explain_eval_type = config.get("experiment", {}).get("explain_eval_type")

    if eval_type:
        tags.append(str(eval_type))
    if explain_type:
        tags.append(str(explain_type))
    if explain_eval_type:
        tags.append(str(explain_eval_type))

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
    """Return whether W&B logging is enabled in the configuration."""
    return bool(config.get("logging", {}).get("enabled", False))


def init_wandb_run(
    config: dict[str, Any],
    run_dir: Path,
    exp_name: str,
    experiment_group: str | None = None,
    job_type: str | None = "train",
) -> Run | None:
    """Initialize a W&B run for a single experiment."""
    if not is_wandb_enabled(config):
        return None
    
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    if WANDB_API_KEY is None:
        raise ValueError(
            "W&B API key is not set. Define WANDB_API_KEY in your .env file."
        )
    
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")

    project = (WANDB_PROJECT or "jaguar-reid")
    entity = WANDB_ENTITY

    tags = _build_wandb_tags(config, experiment_group, job_type=job_type)

    run = wandb.init(
        entity=entity,
        project=project,  
        group=experiment_group,
        job_type=job_type,
        tags=tags,
        name=f"{exp_name}_{ROUND}",
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
    run.define_metric("inference/*", step_metric="epoch")
    run.define_metric("timing/*", step_metric="epoch")
    run.define_metric("meta/*", step_metric="epoch")
    run.define_metric("val_rare/*", step_metric="epoch")

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
    """Log dataset and runtime metadata for a run."""
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
    """Log static model parameter counts to W&B."""
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
    rare_metrics: dict[str, Any] = None,
) -> None:
    """Log one epoch of training and validation metrics."""
    if run is None:
        return

    log_dict = {
            "epoch": int(epoch),
            "train/loss": float(avg_loss),
            "val/loss": float(metrics["val_loss"]),
            "val/mAP": float(metrics["mAP"]),
            "val/pairwise_AP": float(metrics["pairwise_AP"]),
            "val/rank1": float(metrics["rank1"]),
            "val/sim_gap": float(metrics["sim_gap"]),
            "meta/lr": float(current_lr),
            "timing/epoch_time_sec": float(epoch_time_sec),
            "meta/input_size": input_size,
            "val/silhouette": float(metrics["silhouette"]) if "silhouette" in metrics else None,
        }

    if "silhouette" in metrics:
        log_dict["val/silhouette"] = float(metrics["silhouette"])
    
    if rare_metrics is not None:
        log_dict["val_rare/mAP"] = float(rare_metrics["mAP"])
        log_dict["val_rare/rank1"] = float(rare_metrics["rank1"])
        log_dict["val_rare/pairwise_AP"] = float(rare_metrics["pairwise_AP"])
        
    run.log(log_dict)


def log_wandb_ensemble_config(
    run: Run | None,
    config: dict[str, Any],
) -> None:
    """Log ensemble-specific configuration fields to W&B."""
    if run is None:
        return

    run.config.update(
        {
            "fusion_method": config.get("fusion", {}).get("method", "score"),
            "fusion_weights": config.get("fusion", {}).get("weights"),
            "normalize_mode": config.get("fusion", {}).get("normalize_mode"),
            "square_before_fusion": config.get("fusion", {}).get("square_before_fusion"),
            "n_members": len(config.get("members", [])),
            "member_names": [m.get("name") for m in config.get("members", [])],
        },
        allow_val_change=True,
    )


def log_wandb_ensemble_results(
    run: Run | None,
    config: dict[str, Any],
    exp_name: str,
    score_metrics: dict[str, Any],
    emb_metrics: dict[str, Any],
    oracle_summary: dict[str, Any],
    oracle_df: pd.DataFrame,
    score_query_df: pd.DataFrame,
    emb_query_df: pd.DataFrame,
    per_model_query_dfs: dict[str, pd.DataFrame],
) -> None:
    """Log ensemble metrics, summaries, and per-query tables."""
    if run is None:
        return

    run.log(
        {
            "ensemble/score_mAP": float(score_metrics["mAP"]),
            "ensemble/score_rank1": float(score_metrics["rank1"]),
            "ensemble/emb_mAP": float(emb_metrics["mAP"]),
            "ensemble/emb_rank1": float(emb_metrics["rank1"]),
            "ensemble/oracle_mAP": float(oracle_summary["oracle_mAP"]),
            "ensemble/oracle_rank1": float(oracle_summary["oracle_rank1"]),
        }
    )

    run.summary["experiment_name"] = exp_name
    run.summary["n_members"] = len(config.get("members", []))
    run.summary["member_names"] = [m.get("name") for m in config.get("members", [])]
    run.summary["weights"] = config.get("fusion", {}).get("weights")
    run.summary["normalize_mode"] = config.get("fusion", {}).get("normalize_mode", "global_minmax")
    run.summary["square_before_fusion"] = config.get("fusion", {}).get("square_before_fusion", True)
    run.summary["score_mAP"] = float(score_metrics["mAP"])
    run.summary["score_rank1"] = float(score_metrics["rank1"])
    run.summary["emb_mAP"] = float(emb_metrics["mAP"])
    run.summary["emb_rank1"] = float(emb_metrics["rank1"])
    run.summary["oracle_mAP"] = float(oracle_summary["oracle_mAP"])
    run.summary["oracle_rank1"] = float(oracle_summary["oracle_rank1"])

    log_wandb_table(run, "oracle/per_query", oracle_df)
    log_wandb_table(run, "score_fusion/per_query", score_query_df)
    log_wandb_table(run, "embedding_fusion/per_query", emb_query_df)

    for name, query_df in per_model_query_dfs.items():
        log_wandb_table(run, f"{name}/per_query", query_df)


def finish_wandb_run(
    run: Run | None,
    best_epoch: int | None,
    best_score: float,
    monitor_metric: str,
    best_metrics: dict[str, Any] | None,
    epochs_completed: int,
    total_train_time_sec: float,
    best_rare_epoch: int | None = None,
    best_rare_metrics: dict[str, Any] | None = None,
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
    
    if best_rare_metrics is not None:
        run.summary["best_rare_mAP"] = float(best_rare_metrics["mAP"])
        run.summary["best_rare_epoch"] = best_rare_epoch
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
    """Log a checkpoint file as a W&B model artifact."""

    if run is None or not checkpoint_path.exists():
        return

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=metadata or {},
    )
    artifact.add_file(local_path=str(checkpoint_path), name=checkpoint_path.name)
    run.log_artifact(artifact, aliases=aliases or ["best"])


def log_wandb_background_intervention_results(
    run: Run | None,
    result: dict,
) -> None:
    """Log summary tables and metrics for background intervention experiments."""
    if run is None:
        return

    summary_all = result["summary_all"]

    run.log({
        "background/original_mAP": float(summary_all.loc[summary_all["setting"] == "original", "mAP"].iloc[0]),
        "background/original_rank1": float(summary_all.loc[summary_all["setting"] == "original", "rank1"].iloc[0]),
    })

    for _, row in summary_all.iterrows():
        setting = str(row["setting"])
        run.log(
            {
                f"background/{setting}/mAP": float(row["mAP"]),
                f"background/{setting}/rank1": float(row["rank1"]),
                f"background/{setting}/delta_mAP_vs_original": float(row["delta_mAP_vs_original"]),
                f"background/{setting}/delta_rank1_vs_original": float(row["delta_rank1_vs_original"]),
            }
        )

    log_wandb_table(run, "background/summary_all", summary_all)
    log_wandb_table(run, "background/per_query_delta", result["per_query_delta"])


def log_wandb_background_sensitivity_results(
    run: Run | None,
    retrieval_summary: dict,
    similarity_summary: dict,
    analysis_df: pd.DataFrame,
) -> None:
    """Log background sensitivity summaries and analysis tables."""
    if run is None:
        return

    run.log({
        "bg_sensitivity/share_bg_better_rank": float(retrieval_summary["all"]["share_bg_better_rank"]),
        "bg_sensitivity/share_bg_better_rank1": float(retrieval_summary["all"]["share_bg_better_rank1"]),
        "bg_sensitivity/share_bg_more_stable": float(similarity_summary["all"]["share_bg_more_stable"]),
    })

    log_wandb_table(run, "bg_sensitivity/analysis_df", analysis_df)


def log_wandb_xai_similarity_results(
    run: Run | None,
    ref_df: pd.DataFrame,
    explainer_names: list[str],
    pair_types: tuple[str, ...],
) -> None:
    """Log reference-pair counts and metadata for XAI similarity experiments."""
    if run is None:
        return

    counts = ref_df["pair_type"].value_counts().to_dict()
    payload = {f"xai_refs/{k}": int(v) for k, v in counts.items()}
    run.log(payload)

    run.summary["explainer_names"] = list(explainer_names)
    run.summary["pair_types"] = list(pair_types)

    log_wandb_table(run, "xai/reference_pairs", ref_df)


def log_wandb_xai_metrics_results(run, summary_df: pd.DataFrame) -> None:
    """Log XAI metric summary rows to W&B."""
    if run is None or summary_df is None or summary_df.empty:
        return

    slice_col = "pair_type" if "pair_type" in summary_df.columns else "group"

    metric_cols = [
        "sanity_mean", "sanity_median", "sanity_std",
        "faith_topk_mean", "faith_topk_median", "faith_topk_std",
        "faith_random_mean", "faith_random_median", "faith_random_std",
        "faith_gap_mean", "faith_gap_median", "faith_gap_std",
        "complexity_mean", "complexity_median", "complexity_std",
    ]

    for _, row in summary_df.iterrows():
        expl = row["explainer"]
        slice_value = row[slice_col]

        payload = {}
        for col in metric_cols:
            if col in row.index and pd.notna(row[col]):
                payload[f"xai_metrics/{expl}/{slice_value}/{col}"] = float(row[col])

        if payload:
            run.log(payload)


def log_wandb_xai_class_attribution_results(
    run: Run | None,
    manifest: pd.DataFrame,
    artifacts_saved: list[dict[str, Any]],
    groups: tuple[str, ...],
    explainer_names: tuple[str, ...],
) -> None:
    """Log compact summaries for Stage-2 class attribution generation."""
    if run is None:
        return

    run.summary["groups"] = list(groups)
    run.summary["explainer_names"] = list(explainer_names)
    run.summary["n_manifest_rows"] = int(len(manifest))
    run.summary["n_saved_artifacts"] = int(len(artifacts_saved))

    if not manifest.empty:
        run.log({"xai_class/manifest_rows": int(len(manifest))})
        log_wandb_table(run, "xai_class/source_manifest", manifest)

    if artifacts_saved:
        artifacts_df = pd.DataFrame(artifacts_saved)

        for _, row in artifacts_df.iterrows():
            run.log(
                {
                    f"xai_class/{row['group']}/{row['explainer']}/n_samples": int(row["n_samples"])
                }
            )

        log_wandb_table(run, "xai_class/artifacts_saved", artifacts_df)

        

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