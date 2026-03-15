# src/jaguar/retrieval/retrieval_experiments.py

import tomli_w
from pathlib import Path
from torch.utils.data import DataLoader

from jaguar.config import PATHS, EXPERIMENTS_STORE
from jaguar.utils.utils import resolve_path
from jaguar.utils.utils_datasets import (
    build_processing_fn,
    get_transforms,
    load_split_jaguar_from_FO_export,
)

from jaguar.retrieval.retrieval_utils import run_retrieval_sweep


def build_val_loader(config, model):

    train_processing_fn = build_processing_fn(config, split="train")
    val_processing_fn = build_processing_fn(config, split="val")

    parquet_root = resolve_path(
        config["data"]["split_data_path"],
        EXPERIMENTS_STORE
    )

    _, _, val_ds = load_split_jaguar_from_FO_export(
        PATHS.data_export / "splits_curated",
        overwrite_db=False,
        parquet_path=parquet_root,
        train_processing_fn=train_processing_fn,
        val_processing_fn=val_processing_fn,
        include_duplicates=config["split"]["include_duplicates"],
    )

    val_ds.transform = get_transforms(
        config,
        model.backbone_wrapper,
        is_training=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4, #config["data"]["num_workers"],
        pin_memory=True
    )

    return val_loader, val_ds.labels_idx


def build_retrieval_override(run_cfg, experiment_meta):

    override = {}

    field_to_section = {

        "apply_tta": ("evaluation", "apply_tta"),
        "tta_modality": ("evaluation", "tta_modality"),

        "apply_qe": ("evaluation", "apply_qe"),
        "top_k_expansion": ("evaluation", "top_k_expansion"),

        "apply_rerank": ("evaluation", "apply_rerank"),
        "k1": ("evaluation", "k1"),
        "k2": ("evaluation", "k2"),
        "lambda_value": ("evaluation", "lambda_value"),
    }

    for key, value in run_cfg.items():

        if key == "experiment_name":
            continue

        if key not in field_to_section:
            continue

        section, target = field_to_section[key]

        override.setdefault(section, {})
        override[section][target] = value

    override.setdefault("evaluation", {})
    override["evaluation"]["experiment_group"] = experiment_meta["name"]

    return override


def generate_retrieval_experiments(experiment_cfg, output_root):

    meta = experiment_cfg["experiment"]

    experiment_name = meta["name"]

    runs = meta["runs"]

    output_dir = output_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []

    for run in runs:
        run_name = run["experiment_name"]
        override = build_retrieval_override(run, meta)
        out_path = output_dir / f"{run_name}.toml"

        with open(out_path, "wb") as f:
            tomli_w.dump(override, f)

        generated_paths.append(out_path)
        print(f"Generated {out_path}")
    return generated_paths


def run_retrieval_experiment(
    model,
    config,
    run_cfg,
    checkpoint_dir
):

    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Building validation loader")

    val_loader, labels = build_val_loader(config, model)

    print("Starting retrieval sweep")

    run_retrieval_sweep(
        model=model,
        val_loader=val_loader,
        labels=labels,
        run_cfg=run_cfg,
        output_dir=checkpoint_dir
    )
