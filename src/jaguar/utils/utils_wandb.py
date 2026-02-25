'''
how to use:
study_id = "dedup_ablation__2026-02-25__splitv03"
group = "02_dedup_ablation"

for ph_t in [8, 10, 12]:
    for seed in [42, 51]:
        run_name = f"phash{ph_t}_s{seed}"

        overrides = {
            "repro": {"seed": seed},
            "data": {
                "dataset_version": "jaguar_stage0_v2",
                "image_size": 224,
                "background_mode": "original",
            },
            "split": {
                "split_version": "split_v03_grouped_burstsafe",
                "split_strategy": "stratified_grouped",
                "group_key": "burst_group_id",
            },
            "dedup": {
                "enabled": True,
                "dedup_version": "dedup_v2",
                "method": "phash+faiss",
                "hash_type": "phash",
                "phash_threshold": ph_t,
                "faiss_knn_k": 50,
                "cluster_method": "connected_components",
                "representative_rule": "sharpest",
                "max_within_pairs_per_identity": 200,
                "max_cross_pairs_total": 5000,
            },
            "logging": {
                "log_artifacts": True,
                "log_images": False,
            },
        }

        run, cfg = init_jaguar_wandb(
            PATHS=PATHS,
            project="jaguar-reid",
            group=group,
            experiment_family="dedup_ablation",
            study_id=study_id,
            job_type="dedup",
            run_name=run_name,
            overrides=overrides,
            tags=["dedup", "phash", "faiss", f"seed{seed}"],
            run_purpose="ablation",
            seed=seed,
            script_name="run_dedup_ablation.py",
        )

        try:
            # ... run dedup pipeline ...
            # Log dedup metrics (example)
            wandb.log({
                "dedup/n_images_total": 12345,
                "dedup/n_duplicate_images": 2345,
                "dedup/n_kept_images": 10000,
                "dedup/n_burst_groups": 890,
                "dedup/duplicate_rate": 2345 / 12345,
                "dedup/threshold_phash": ph_t,
                "dedup/cross_identity_collision_rate": 0.0032,
            })
        finally:
            run.finish()

'''


from __future__ import annotations

import os
import sys
import platform
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from jaguar.config import IN_COLAB
import wandb


# ----------------------------
# Controlled vocabulary
# ----------------------------
JOB_TYPES = {
    "train",
    "eval",
    "embeddings",
    "dedup",
    "xai",
    "eda",
    "split",
    "export",
}


# ----------------------------
# Deep merge
# ----------------------------
def deep_update(dst: dict, src: Mapping[str, Any]) -> dict:
    """Recursively update nested dictionaries."""
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst



def _stringify_paths_inplace(cfg: dict) -> None:
    """Convert pathlib Paths recursively so W&B config serialization is safe."""
    def convert(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        if isinstance(x, tuple):
            return [convert(v) for v in x]  # W&B config likes JSON-ish types
        return x

    new_cfg = convert(cfg)
    cfg.clear()
    cfg.update(new_cfg)


# ----------------------------
# Base + job templates
# ----------------------------
def make_base_config(
    *,
    paths: Any,
    group: str,
    experiment_family: str,
    study_id: str,
    job_type: str,
    run_purpose: str = "baseline",
    seed: int = 42,
    project_name: str = "jaguar-reid",
    script_name: str | None = None,
) -> dict[str, Any]:
    if job_type not in JOB_TYPES:
        raise ValueError(f"Unknown job_type='{job_type}'. Allowed: {sorted(JOB_TYPES)}")

    # Try to extract a useful root path from your PATHS object
    data_root = getattr(paths, "data_export", None)
    runs_root = getattr(paths, "runs", None)

    cfg = {
        "meta": {
            "project_name": project_name,
            "group": group,
            "experiment_family": experiment_family,   # semantic label
            "study_id": study_id,                     # exact campaign instance
            "job_type": job_type,
            "run_purpose": run_purpose,               # baseline / ablation / debug / final / paper
        },
        "repro": {
            "seed": seed,
            "deterministic": True,
            "cudnn_benchmark": False,
            "num_workers": None,
        },
        "data": {
            "dataset_name": "jaguar_stage0",
            "dataset_version": None,
            "data_root": data_root,
            "image_size": None,
            "color_mode": "rgb",
            "background_mode": None,                  # original / gray / removed / transparent
            "augmentation_profile": None,             # usually train-only
        },
        "split": {
            "split_version": None,
            "split_strategy": None,
            "identity_key": "identity_id",
            "group_key": None,                        # e.g. burst_group_id
            "train_csv": None,
            "val_csv": None,
            "test_csv": None,
            "fold": None,
        },
        "dedup": {
            "enabled": None,
            "dedup_version": None,
            "method": None,                           # none / phash / phash+faiss
            "hash_type": None,                        # phash / dhash / both
            "phash_threshold": None,
            "dhash_threshold": None,
            "faiss_knn_k": None,
            "cluster_method": None,
            "representative_rule": None,
            "cross_identity_collision_rate": None,    # metric may be copied into config for traceability
        },
        "runtime": {
            "script": script_name,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "runs_root": runs_root,
        },
        "logging": {
            "wandb_mode": os.environ.get("WANDB_MODE", "online"),
            "log_artifacts": True,
            "log_images": False,
            "log_model_checkpoint": False,
        },
    }

    _stringify_paths_inplace(cfg)
    return cfg


def make_job_config(job_type: str) -> dict[str, Any]:
    """Return job-specific config skeleton."""
    if job_type == "train":
        return {
            "model": {
                "backbone": None,
                "pretrained": True,
                "embedding_dim": None,
                "pooling": None,
                "bnneck": None,
                "head_type": None,  # arcface / linear / none
                "checkpoint_ref": None,  # optional resume / finetune source
            },
            "loss": {
                "primary": None,
                "secondary": [],
                "weights": {},
                "triplet_margin": None,
                "miner": None,
            },
            "train": {
                "epochs": None,
                "batch_size": None,
                "optimizer": None,
                "lr": None,
                "weight_decay": None,
                "scheduler": None,
                "warmup_epochs": None,
                "amp": True,
                "grad_clip_norm": None,
            },
            "sampler": {
                "type": None,
                "identities_per_batch": None,
                "images_per_identity": None,
            },
            "augment": {
                "profile": None,
            },
            "checkpointing": {
                "monitor": "val/cmc1",
                "mode": "max",
                "save_best": True,
                "save_last": True,
            },
        }

    if job_type == "eval":
        return {
            "model": {
                "backbone": None,
                "checkpoint_ref": None,
                "embedding_dim": None,
            },
            "eval": {
                "eval_type": "retrieval",   # retrieval / classification / robustness
                "split_used": None,
                "distance_metric": "cosine",
                "normalize_embeddings": True,
                "k_values": [1, 5, 10],
                "use_faiss": True,
                "tta": False,
            },
            "analysis": {
                "save_per_identity_stats": True,
                "save_failure_cases": False,
                "top_failure_examples_n": None,
            },
        }

    if job_type == "embeddings":
        return {
            "model": {
                "backbone": None,
                "checkpoint_ref": None,
                "embedding_dim": None,
            },
            "embeddings": {
                "split_used": None,
                "batch_size": None,
                "normalize_embeddings": True,
                "save_format": "npy",
                "include_metadata": True,
                "include_labels": True,
            },
        }

    if job_type == "dedup":
        return {
            "dedup": {
                "enabled": True,
                "method": "phash",
                "hash_type": "phash",
                "phash_threshold": None,
                "dhash_threshold": None,
                "faiss_knn_k": None,
                "cluster_method": "connected_components",
                "representative_rule": "sharpest",
                "burst_min_size": 2,
                "max_within_pairs_per_identity": None,
                "max_cross_pairs_total": None,
                "use_embedding_refine": None,
                "embedding_model_for_refine": None,
            },
            "outputs": {
                "burst_assignments_parquet": None,
                "dedup_stats_csv": None,
                "dedup_plots_dir": None,
            },
        }

    if job_type == "split":
        return {
            "split": {
                "split_version": None,
                "split_strategy": None,
                "identity_key": "identity_id",
                "group_key": None,
                "label_min_count_filter": None,
                "train_ratio": None,
                "val_ratio": None,
                "test_ratio": None,
                "stratify": True,
                "grouped_no_leakage": True,
            },
            "inputs": {
                "source_metadata_csv": None,
                "dedup_assignments": None,
            },
            "outputs": {
                "train_csv": None,
                "val_csv": None,
                "test_csv": None,
            },
        }

    if job_type == "xai":
        return {
            "model": {
                "backbone": None,
                "checkpoint_ref": None,
            },
            "xai": {
                "task_type": None,          # reid_pair_similarity / classification
                "method": None,             # gradcam / ig / lrp / rollout
                "target_type": "predicted", # predicted / true / pair_similarity
                "layer": None,
                "abs_map": True,
                "save_signed_maps": False,
            },
            "xai_sampling": {
                "subset_name": None,
                "n_pairs": None,
                "pair_mining": None,
                "seed": None,
            },
            "ig": {
                "steps": None,
                "baseline": None,
                "internal_batch_size": None,
            },
            "quantus": {
                "enabled": False,
                "metrics": [],
                "n_samples": None,
                "batch_size": None,
                "use_abs_saliency": True,
                "normalization": None,
            },
            "outputs": {
                "saliency_dir": None,
                "quantus_results_csv": None,
                "visual_examples_dir": None,
            },
        }

    if job_type == "eda":
        return {
            "eda": {
                "analysis_name": None,
                "save_plots": True,
                "max_examples_per_identity": None,
            },
            "inputs": {
                "metadata_csv": None,
                "burst_assignments": None,
            },
            "outputs": {
                "plots_dir": None,
                "summary_csv": None,
            },
        }

    if job_type == "export":
        return {
            "export": {
                "kind": None,               # submission / report_table / embeddings_package / checkpoint_bundle
                "format": None,
                "source_refs": [],
            }
        }

    raise ValueError(f"Unsupported job_type='{job_type}'")


# ----------------------------
# Validation
# ----------------------------
def _get_nested(cfg: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


REQUIRED_BY_JOB_TYPE = {
    "dedup": [
        "meta.study_id",
        "meta.experiment_family",
        "dedup.method",
        # usually one of phash_threshold / dhash_threshold depending on method
    ],
    "split": [
        "meta.study_id",
        "split.split_version",
        "split.split_strategy",
    ],
    "train": [
        "meta.study_id",
        "model.backbone",
        "model.head_type",
        "loss.primary",
        "train.epochs",
        "train.batch_size",
        "train.optimizer",
        "train.lr",
    ],
    "embeddings": [
        "meta.study_id",
        "model.backbone",
        "model.checkpoint_ref",
        "embeddings.split_used",
        "embeddings.batch_size",
    ],
    "eval": [
        "meta.study_id",
        "model.backbone",
        "model.checkpoint_ref",
        "eval.eval_type",
        "eval.split_used",
    ],
    "xai": [
        "meta.study_id",
        "model.backbone",
        "model.checkpoint_ref",
        "xai.method",
        "xai.task_type",
    ],
    "eda": [
        "meta.study_id",
        "eda.analysis_name",
    ],
    "export": [
        "meta.study_id",
        "export.kind",
        "export.format",
    ],
}


def validate_config(cfg: Mapping[str, Any]) -> None:
    job_type = _get_nested(cfg, "meta.job_type")
    if job_type not in JOB_TYPES:
        raise ValueError(f"Invalid meta.job_type='{job_type}'")

    missing = []
    for key in REQUIRED_BY_JOB_TYPE.get(job_type, []):
        val = _get_nested(cfg, key)
        if val is None:
            missing.append(key)

    # small conditional checks
    if job_type == "dedup":
        method = _get_nested(cfg, "dedup.method")
        ph_t = _get_nested(cfg, "dedup.phash_threshold")
        dh_t = _get_nested(cfg, "dedup.dhash_threshold")

        if method in {"phash", "phash+faiss"} and ph_t is None:
            missing.append("dedup.phash_threshold")
        if method == "dhash" and dh_t is None:
            missing.append("dedup.dhash_threshold")
        if method == "both" and ph_t is None and dh_t is None:
            missing.append("dedup.phash_threshold or dedup.dhash_threshold")

    if missing:
        msg = (
            f"W&B config validation failed for job_type='{job_type}'. "
            f"Missing required fields: {missing}"
        )
        raise ValueError(msg)


# ----------------------------
# Main helper
# ----------------------------
def init_jaguar_wandb(
    *,
    PATHS: Any,
    project: str,
    group: str,
    experiment_family: str,
    study_id: str,
    job_type: str,
    run_name: str,
    overrides: Mapping[str, Any] | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    run_purpose: str = "baseline",
    seed: int = 42,
    script_name: str | None = None,
    validate: bool = True,
) -> tuple[wandb.sdk.wandb_run.Run, dict[str, Any]]:
    """
    Build a resolved, validated config and initialize a W&B run.

    Returns:
        (wandb_run, resolved_config)
    """
    wandb.login(key=os.environ["WANDB_API_KEY"])

    if IN_COLAB:
        mode = "online"
    else: 
        mode = "offline"

    cfg = make_base_config(
        paths=PATHS,
        group=group,
        experiment_family=experiment_family,
        study_id=study_id,
        job_type=job_type,
        run_purpose=run_purpose,
        seed=seed,
        project_name=project,
        script_name=script_name,
    )

    cfg = deep_update(cfg, make_job_config(job_type))

    if overrides:
        cfg = deep_update(cfg, deepcopy(dict(overrides)))

    _stringify_paths_inplace(cfg)

    if validate:
        validate_config(cfg)

    run = wandb.init(
        project=project,
        group=group,
        job_type=job_type,
        name=run_name,
        tags=tags or [],
        notes=notes,
        config=cfg,
        mode=mode,
    )
    return run, cfg