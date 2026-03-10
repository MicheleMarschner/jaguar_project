import argparse
import subprocess
import tempfile
import tomllib
from pathlib import Path

from jaguar.config import PATHS
from jaguar.experiments.experiment_setup import build_split_relpath
from jaguar.utils.utils import ensure_dir


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
    return parser.parse_args()


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


def build_ensemble_override(run_cfg: dict, experiment_meta: dict, base_config: dict) -> dict:
    override = {
        "ensemble": {
            "name": run_cfg["experiment_name"],
        }
    }

    field_to_section = {
        "weights": ("fusion", "weights"),
        "normalize_per_model": ("fusion", "normalize_per_model"),
        "square_before_fusion": ("fusion", "square_before_fusion"),
        "use_tta": ("inference", "use_tta"),
        "use_qe": ("inference", "use_qe"),
        "use_rerank": ("inference", "use_rerank"),
        "batch_size": ("inference", "batch_size"),
        "split_data_path": ("data", "split_data_path"),
        "num_workers": ("data", "num_workers"),
    }

    for key, value in run_cfg.items():
        if key == "experiment_name":
            continue
        if key not in field_to_section:
            continue

        section, target_key = field_to_section[key]
        override.setdefault(section, {})
        override[section][target_key] = value

    return override
    

def build_training_override(
    run_cfg: dict, 
    experiment_meta: dict,
    base_config: dict
) -> dict:
    override = { 
        "training": { "experiment_name": run_cfg["experiment_name"], },
    }

    field_to_section = {
        "train_background": ("preprocessing", "train_background"),
        "val_background": ("preprocessing", "val_background"),

        "backbone_name": ("model", "backbone_name"),
        "head_type": ("model", "head_type"),
        "s": ("model", "s"),
        "m": ("model", "m"),

        "optimizer_type": ("optimizer", "type"),
        "optimizer_lr": ("optimizer", "lr"),
        "optimizer_weight_decay": ("optimizer", "weight_decay"),
        "optimizer_momentum": ("optimizer", "momentum"),
        "optimizer_betas": ("optimizer", "betas"),

        "scheduler_type": ("scheduler", "type"),
        "scheduler_T_max": ("scheduler", "T_max"),
        "scheduler_lr_start": ("scheduler", "lr_start"),
        "scheduler_lr_min": ("scheduler", "lr_min"),
        "scheduler_lr_max": ("scheduler", "lr_max"),
        "scheduler_lr_ramp_ep": ("scheduler", "lr_ramp_ep"),
        "scheduler_lr_sus_ep": ("scheduler", "lr_sus_ep"),
        "scheduler_lr_decay": ("scheduler", "lr_decay"),
        "scheduler_factor": ("scheduler", "factor"),
        "scheduler_patience": ("scheduler", "patience"),
        "scheduler_epochs": ("scheduler", "epochs"),
        "scheduler_total_steps": ("scheduler", "total_steps"),

        "apply_augmentations": ("augmentation", "apply_augmentations"),
        "horizontal_flip": ("augmentation", "horizontal_flip"),
        "affine_degrees": ("augmentation", "affine_degrees"),
        "affine_translate": ("augmentation", "affine_translate"),
        "affine_scale": ("augmentation", "affine_scale"),
        "color_jitter_brightness": ("augmentation", "color_jitter_brightness"),
        "color_jitter_contrast": ("augmentation", "color_jitter_contrast"),
        "random_erasing_p": ("augmentation", "random_erasing_p"),

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

        "model_name": ("xai", "model_name"),
        "n_samples": ("xai", "n_samples"),
        "split_name": ("xai", "split_name"),
        "pair_types": ("xai", "pair_types"),
        "explainer_names": ("xai", "explainer_names"),
        "ig_steps": ("xai", "ig_steps"),
        "ig_internal_bs": ("xai", "ig_internal_bs"),
        "ig_batch_size": ("xai", "ig_batch_size"),

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


def run_experiments():
    args = parse_args()
    experiment_config = load_toml_config(args.experiment_config)
    base_config = load_toml_config(args.base_config)

    experiment_meta = experiment_config.get("experiment", {})
    runs = experiment_meta.get("runs", [])
    if not runs:
        raise ValueError("No runs found under [[experiment.runs]]")

    experiment_name = experiment_config.get("experiment", {}).get("name", "experiment")
    setup_name = experiment_meta.get("setup_name")
    generated_dir = PATHS.configs / "_generated" / experiment_name
    ensure_dir(generated_dir)

    mode = experiment_meta.get("mode", "scientific")

    print(f"Found {len(runs)} runs.")

    all_cmds = []

    for i, run_cfg in enumerate(runs, start=1):
        experiment_name = run_cfg["experiment_name"]
        
        if mode == "ensemble":
            override = build_ensemble_override(
                run_cfg=run_cfg,
                experiment_meta=experiment_meta,
                base_config=base_config,
            )
        else:
            override = build_training_override(
                run_cfg=run_cfg,
                experiment_meta=experiment_meta,
                base_config=base_config,
            )
        override_text = dict_to_toml(override)

        print(f"\n[{i}/{len(runs)}] {experiment_name}")

        override_path = generated_dir / f"{experiment_name}.toml"
        override_path.write_text(override_text, encoding="utf-8")
        rel_path = override_path.relative_to(PATHS.configs).with_suffix("")

        if mode == "xai":
            # !TODO change according to file_path and function_name
            target_script = "src/jaguar/run_xai_experiment.py"
        elif mode == "ensemble":
            # !TODO change according to file_path and function_name
            target_script = "src/jaguar/run_ensemble.py"
        else:
            target_script = args.main_script

        cmd = [
            "python",
            target_script,
            "--base_config",
            args.base_config,
            "--experiment_config",
            str(rel_path),
        ]

        if mode not in {"xai"}:
            cmd.extend([
                "--experiment_name",
                experiment_name,
            ])

        print("Generated override config:")
        print(override_text)
        print("Command:")
        print(" ".join(cmd))

        if setup_name:
            setup_cmd = [
                "python",
                "src/jaguar/experiments/experiment_setup.py",
                "--setup_name",
                setup_name,
                "--base_config",
                args.base_config,
                "--experiment_config",
                str(rel_path),
            ]
            print("Running setup:", " ".join(setup_cmd))
            setup_result = subprocess.run(setup_cmd)

            if setup_result.returncode != 0:
                raise RuntimeError(f"Setup failed: {experiment_name}")

        print("Running:", " ".join(cmd))
        #result = subprocess.run(cmd)

        #if result.returncode != 0:
        #    raise RuntimeError(f"Run failed: {experiment_name}")

        run_lines = []
        if setup_name:
            run_lines.append(" ".join(setup_cmd))
        run_lines.append(" ".join(cmd))
        all_cmds.extend(run_lines)
        all_cmds.append("")

        print("Generated override config:")
        print(override_text)
        print("Command:")
        print(" ".join(cmd))


    run_script_path = generated_dir / "run_all.sh"

    script_lines = [
        "#!/usr/bin/env bash",
        "set -e",
        "",
    ]

    script_lines.extend(all_cmds)
    script_lines.append("")

    run_script_path.write_text("\n".join(script_lines), encoding="utf-8")

    print(f"\nSaved run script: {run_script_path}")
    print("\nAll runs finished.")


if __name__ == "__main__":
    run_experiments()