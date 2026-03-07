import argparse
import subprocess
import tempfile
import tomllib
from pathlib import Path

from jaguar.config import PATHS
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


def build_experiment_override(run_cfg: dict) -> dict:
    override = {
        "training": {
            "experiment_name": run_cfg["experiment_name"],
        }
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

        "seed": ("training", "seed"),
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


def run_experiments():
    args = parse_args()
    experiment_config = load_toml_config(args.experiment_config)

    exp_meta = experiment_config.get("experiment", {})
    runs = exp_meta.get("runs", [])
    if not runs:
        raise ValueError("No runs found under [[experiment.runs]]")

    experiment_name = experiment_config.get("experiment", {}).get("name", "experiment")
    setup_name = exp_meta.get("setup_name")
    generated_dir = PATHS.configs / "_generated" / experiment_name
    ensure_dir(generated_dir)

    print(f"Found {len(runs)} runs.")

    all_cmds = []

    for i, run_cfg in enumerate(runs, start=1):
        experiment_name = run_cfg["experiment_name"]
        override = build_experiment_override(run_cfg)
        override_text = dict_to_toml(override)

        print(f"\n[{i}/{len(runs)}] {experiment_name}")

        override_path = generated_dir / f"{experiment_name}.toml"
        override_path.write_text(override_text, encoding="utf-8")
        rel_path = override_path.relative_to(PATHS.configs).with_suffix("")

        cmd = [
            "python",
            args.main_script,
            "--base_config",
            args.base_config,
            "--experiment_config",
            str(rel_path),
            "--experiment_name",
            experiment_name,
        ]

        all_cmds.append(" ".join(cmd))

        print("Generated override config:")
        print(override_text)
        print("Command:")
        print(" ".join(cmd))

        print("Running:", " ".join(cmd))
        #result = subprocess.run(cmd)

        #if result.returncode != 0:
        #    raise RuntimeError(f"Run failed: {experiment_name}")

    run_script_path = generated_dir / "run_all.sh"

    script_lines = [
        "#!/usr/bin/env bash",
        "set -e",
        "",
    ]

    if setup_name:
        script_lines.append(
            f"python src/jaguar/setup_experiment.py --setup_name {setup_name}"
        )
        script_lines.append("")

    script_lines.extend(all_cmds)
    script_lines.append("")

    run_script_path.write_text("\n".join(script_lines), encoding="utf-8")

    print(f"\nSaved run script: {run_script_path}")
    print("\nAll runs finished.")


if __name__ == "__main__":
    run_experiments()