import argparse
import subprocess
import tempfile
import tomllib
from pathlib import Path

from jaguar.config import PATHS
from jaguar.experiments.experiment_setup import build_split_relpath
from jaguar.utils.utils import ensure_dir

################################################
# Argument Parsing
################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        "--base_config",
        type=str,
        required=True,
        help="Path to the base config TOML file, relative to PATHS.configs and without .toml",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        required=True,
        help="Path to the experiment TOML file, relative to PATHS.configs and without .toml",
    )
    parser.add_argument(
        "--main_script",
        type=str,
        default="src/jaguar/main.py",
        help="Path to the main script - conducts a single run",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print generated configs without executing them",
    )
    parser.add_argument(
        "--submit_slurm",
        action="store_true",
        help="Submit experiments as SLURM array job"
    )
    return parser.parse_args()


################################################
# TOML files loading and overwriting logic
################################################
def load_toml_config(config_name: str) -> dict:
    with open(PATHS.configs / f"{config_name}.toml", "rb") as f:
        return tomllib.load(f)


def to_toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(to_toml_value(v) for v in value) + "]"
    raise TypeError(f"Unsupported type: {type(value)}")


def dict_to_toml(data: dict) -> str:
    lines = []
    for section, values in data.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {to_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)

################################################
# Helper to pick values from config dictionaries
################################################
def _pick_value(run_cfg: dict, experiment_meta: dict, base_config: dict, *path, default=None):
    key = path[-1]

    if key in run_cfg:
        return run_cfg[key]
    if key in experiment_meta:
        return experiment_meta[key]

    cur = base_config
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


################################################
# Build per-run overrides
################################################
def build_experiment_override(
    run_cfg: dict, 
    experiment_meta: dict,
    base_config: dict
) -> dict:
    override = {
        "training": {
            "experiment_name": run_cfg["experiment_name"],
        }
    }

    field_to_section = {
        "train_background": ("preprocessing", "train_background"),
        "val_background": ("preprocessing", "val_background"),

        "backbone_name": ("model", "backbone_name"),
        "emb_dim": ("model", "emb_dim"),
        "head_type": ("model", "head_type"),
        "s": ("model", "s"),
        "m": ("model", "m"),
        "use_projection": ("model", "use_projection"),
        "use_forward_features": ("model", "use_forward_features"),
        "mining_type": ("model", "mining_type"),
        
        "ema": ("training", "ema"),
        "ema_decay": ("training", "ema_decay"),
        "monitor_metric": ("training", "monitor_metric"),
        "epochs": ("training", "epochs"),
        "unfreeze_epoch": ("training", "unfreeze_epoch"),
        "unfreeze_blocks": ("training", "unfreeze_blocks"),
        "early_stopping" : ("training", "early_stopping"),
        "early_stopping_patience" : ("training", "early_stopping_patience"),
        "samples_per_class" : ("training", "samples_per_class"),

        "optimizer_type": ("optimizer", "type"),
        "optimizer_lr": ("optimizer", "lr"),
        "optimizer_lr_muon": ("optimizer", "lr_muon"),
        "optimizer_backbone_lr": ("optimizer", "backbone_lr"),
        "optimizer_weight_decay": ("optimizer", "weight_decay"),
        "optimizer_momentum": ("optimizer", "momentum"),
        "optimizer_betas": ("optimizer", "betas"),
        "optimizer_factor": ("optimizer", "optimizer_factor"),
        "optimizerr_patience": ("optimizer", "optimizer_patience"),

        "scheduler_type": ("scheduler", "type"),
        "scheduler_T_max": ("scheduler", "T_max"),
        "scheduler_lr_start": ("scheduler", "lr_start"),
        "scheduler_lr_min": ("scheduler", "lr_min"),
        "scheduler_lr_max": ("scheduler", "lr_max"),
        "scheduler_lr_ramp_ep": ("scheduler", "lr_ramp_ep"),
        "scheduler_lr_sus_ep": ("scheduler", "lr_sus_ep"),
        "scheduler_lr_decay": ("scheduler", "lr_decay"),

        "apply_augmentations": ("augmentation", "apply_augmentations"),
        "horizontal_flip": ("augmentation", "horizontal_flip"),
        "random_resized_crop": ("augmentation", "random_resized_crop"),
        "gaussian_blur": ("augmentation", "gaussian_blur"),
        "affine_degrees": ("augmentation", "affine_degrees"),
        "affine_translate": ("augmentation", "affine_translate"),
        "affine_scale": ("augmentation", "affine_scale"),
        "color_jitter_brightness": ("augmentation", "color_jitter_brightness"),
        "color_jitter_contrast": ("augmentation", "color_jitter_contrast"),
        "random_erasing_p": ("augmentation", "random_erasing_p"),
        
        "silhouette_freq": ("mining_analysis", "silhouette_freq"),
        "force_silhouette": ("mining_analysis", "force_silhouette"),
        
        "enabled": ("rare_identity_eval", "enabled"),
        "threshold": ("rare_idenrare_identity_evaltity_val", "threshold"),
        
        "pr_enabled": ("progressive_resizing", "enabled"),
        "pr_sizes": ("progressive_resizing", "sizes"),
        "pr_stage_epochs": ("progressive_resizing", "stage_epochs"),

        "train_k": ("curation", "train_k"),
        "val_k": ("curation", "val_k"),
        "phash_threshold": ("curation", "phash_threshold"),
        "split_strategy": ("split", "strategy"),
        "val_split_size": ("split", "val_split_size"),
        "include_duplicates": ("split", "include_duplicates"),

        "seed": ("training", "seed"),

        "output_profile": ("output", "profile"),
    }

    for key, value in run_cfg.items():
        if key == "experiment_name":
            continue

        if key not in field_to_section:
            continue

        section, target_key = field_to_section[key]
        override.setdefault(section, {})
        override[section][target_key] = value
    
    output_profile = experiment_meta.get("output_profile")
    experiment_group = experiment_meta.get("name")

    if output_profile is not None:
        override.setdefault("output", {})
        override["output"]["profile"] = output_profile
        override["output"]["experiment_group"] = experiment_group

    existing_split_path = _pick_value(run_cfg, experiment_meta, base_config, "data", "split_data_path")

    has_split_override = any(
        key in run_cfg or key in experiment_meta
        for key in ["split_strategy", "include_duplicates", "train_k", "val_k", "phash_threshold"]
    )

    if existing_split_path is not None and not has_split_override:
        override.setdefault("data", {})
        override["data"]["split_data_path"] = existing_split_path
    elif has_split_override:
        split_strategy = _pick_value(run_cfg, experiment_meta, base_config, "split", "strategy")
        include_duplicates = _pick_value(run_cfg, experiment_meta, base_config, "split", "include_duplicates")
        train_k = _pick_value(run_cfg, experiment_meta, base_config, "curation", "train_k")
        val_k = _pick_value(run_cfg, experiment_meta, base_config, "curation", "val_k")
        phash_threshold = _pick_value(run_cfg, experiment_meta, base_config, "curation", "phash_threshold")

        if (
            split_strategy is not None
            and include_duplicates is not None
            and train_k is not None
            and val_k is not None
            and phash_threshold is not None
        ):
            split_relpath = build_split_relpath(
                split_strategy=split_strategy,
                include_duplicates=include_duplicates,
                train_k_per_dedup=train_k,
                val_k_per_dedup=val_k,
                phash_thresh_dedup=phash_threshold,
            )
            override.setdefault("data", {})
            override["data"]["split_data_path"] = split_relpath

    return override

################################################
# Main launcher
################################################
def run_experiments():
    args = parse_args()
    experiment_config = load_toml_config(args.experiment_config)
    base_config = load_toml_config(args.base_config)

    experiment_meta = experiment_config.get("experiment", {})
    runs = experiment_meta.get("runs", [])
    # runs = runs[:2]                     ## !TODO for dry test!
    if not runs:
        raise ValueError("No runs found under [[experiment.runs]]")

    experiment_name = experiment_config.get("experiment", {}).get("name", "experiment")
    setup_name = experiment_meta.get("setup_name")
    generated_dir = PATHS.configs / "_generated" / experiment_name
    ensure_dir(generated_dir)

    print(f"Found {len(runs)} runs.")
 
    # all_cmds = []
    # store config paths
    config_paths = []

    # Overrides
    for i, run_cfg in enumerate(runs, start=1):
        run_name = run_cfg["experiment_name"]
        seeds = run_cfg.get("seed", [run_cfg.get("seed", 42)])  # support multiple seeds
        for seed in seeds:
            override = build_experiment_override(run_cfg, experiment_meta=experiment_meta, base_config=base_config)
            override.setdefault("training", {})["seed"] = seed
            override["training"]["multiple_seeds"] = len(seeds) > 1 or "stability" in experiment_meta['name'].lower()

            override_text = dict_to_toml(override)
            
            generated_name = f"{run_name}_seed_{seed}.toml" if "optim" in experiment_meta['name'].lower() else f"{run_name}.toml"
            override_path = generated_dir / generated_name
            override_path.write_text(override_text, encoding="utf-8")
            rel_path = override_path.relative_to(PATHS.configs).with_suffix("")
            config_paths.append(str(rel_path))
            
            print(f"[{i}/{len(runs)}] Generated override: {override_path.name} | seed={seed}")
    
    # Generate SLURM array script
    slurm_lines = [
        "#!/bin/bash",
        "#SBATCH --gres=gpu:h100:1",
        "#SBATCH --cpus-per-gpu=32",
        "#SBATCH --time=0-6:00:00",
        "#SBATCH --mem=200GB",
        "#SBATCH --account=kainmueller",
        f"#SBATCH --array=0-{len(config_paths)-1}",
        "#SBATCH --nodelist=maxg[09,10,20]",
        # "#SBATCH --partition=h100",
        "#SBATCH --output=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.out",
        "#SBATCH --error=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.err",
        "#SBATCH -pkainmueller",
        "",
        "CONFIGS=(",
    ]
    
    for c in config_paths:
        slurm_lines.append(f'"{c}"')
    slurm_lines += [
        ")",
        "",
        "CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}",
        "echo \"Running config: $CONFIG\"",
        "",
        f"python {args.main_script} \\",
        f"  --base_config {args.base_config} \\",
        "  --experiment_config \"$CONFIG\"",
    ]
    slurm_path = generated_dir / "run_slurm.sh"
    slurm_path.write_text("\n".join(slurm_lines))
    print(f"Generated SLURM script: {slurm_path}")

    # Optionally submit
    if args.submit_slurm and not args.dry_run:
        subprocess.run(["sbatch", str(slurm_path)])
        print("Submitted SLURM array job.")

if __name__ == "__main__":
    run_experiments()