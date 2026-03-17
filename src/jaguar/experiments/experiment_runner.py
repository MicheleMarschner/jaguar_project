import argparse
import subprocess

from jaguar.config import PATHS
from jaguar.utils.utils import ensure_dir
from jaguar.utils.utils_experiments import build_ensemble_override, build_standard_override, build_xai_override, dict_to_toml, load_toml_config


def parse_args():
    """Parse CLI arguments for batch experiment execution."""
    parser = argparse.ArgumentParser(description="Run experiments")
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
    return parser.parse_args()


def resolve_target_script(mode: str, experiment_meta: dict, main_script: str) -> str:
    """Resolve which experiment entry script to run for the given mode and experiment metadata."""
    if mode == "train":
        return main_script

    if mode == "ensemble":
        return "src/jaguar/experiments/run_ensemble.py"

    if mode == "eval":
        eval_type = experiment_meta.get("eval_type")
        if eval_type == "background_intervention":
            return "src/jaguar/experiments/run_background_intervention.py"
        elif eval_type == "foreground_contribution":
            return "src/jaguar/experiments/run_foreground_contribution.py"
        raise ValueError(f"Unknown eval_type: {eval_type}")

    if mode == "explain":
        explain_type = experiment_meta.get("explain_type")
        if explain_type == "pair_similarity":
            return "src/jaguar/experiments/run_xai_similarity.py"
        if explain_type == "class_attribution":
            return "src/jaguar/experiments/run_class_attribution_generation.py"
        if explain_type == "pair_similarity_metrics":
            return "src/jaguar/experiments/run_xai_metrics.py"
        if explain_type == "background_sensitivity":
            return "src/jaguar/experiments/run_background_sensitivity.py"
        raise ValueError(f"Unknown explain_type: {explain_type}")

    raise ValueError(f"Unknown mode: {mode}")


def expand_run_variants(run_cfg: dict) -> list[dict]:
    """Expand one run config into one or more concrete run variants."""
    if "seed" not in run_cfg:
        return [run_cfg]

    raw_seed = run_cfg["seed"]

    if not isinstance(raw_seed, list):
        return [run_cfg]

    experiment_name = run_cfg["experiment_name"]
    variants = []

    for seed in raw_seed:
        variant = dict(run_cfg)
        variant["seed"] = seed
        variant["multiple_seeds"] = True
        variant["experiment_name"] = f"{experiment_name}_seed_{seed}"
        variants.append(variant)

    return variants


def run_experiments():
    """Generate per-run override configs, optionally run setup, and write a shell script for all commands."""
    args = parse_args()
    experiment_config = load_toml_config(args.experiment_config)
    base_config = load_toml_config(args.base_config)

    experiment_meta = experiment_config.get("experiment", {})
    runs = experiment_meta.get("runs", [])
    if not runs:
        raise ValueError("No runs found under [[experiment.runs]]")

    experiment_group = experiment_config.get("experiment", {}).get("name", "experiment")
    setup_name = experiment_meta.get("setup_name")
    generated_dir = PATHS.configs / "_generated" / experiment_group
    ensure_dir(generated_dir)

    mode = experiment_meta.get("mode", "train")

    expanded_runs = []
    for run_cfg in runs:
        expanded_runs.extend(expand_run_variants(run_cfg))

    print(f"Found {len(expanded_runs)} concrete runs.")
    print(f"mode = {mode}")

    all_cmds = []

    for i, run_cfg in enumerate(expanded_runs, start=1):
        experiment_name = run_cfg["experiment_name"]

        if mode == "ensemble":
            override = build_ensemble_override(
                run_cfg=run_cfg,
                experiment_meta=experiment_meta,
                base_config=base_config,
            )
        elif mode == "explain" or mode == "eval":
            override = build_xai_override(
                run_cfg=run_cfg,
                experiment_meta=experiment_meta,
                base_config=base_config,
            )
        else:
            override = build_standard_override(
                run_cfg=run_cfg,
                experiment_meta=experiment_meta,
                base_config=base_config,
            )

        if "seed" in run_cfg:
            override.setdefault("training", {})["seed"] = run_cfg["seed"]

        if run_cfg.get("multiple_seeds"):
            override.setdefault("training", {})["multiple_seeds"] = True
        
        override_text = dict_to_toml(override)

        print(f"\n[{i}/{len(expanded_runs)}] {experiment_name}")

        override_path = generated_dir / f"{experiment_name}.toml"
        override_path.write_text(override_text, encoding="utf-8")
        rel_path = override_path.relative_to(PATHS.configs).with_suffix("")

        target_script = resolve_target_script(
            mode=mode,
            experiment_meta=experiment_meta,
            main_script=args.main_script,
        )

        cmd = [
            "python",
            target_script,
            "--base_config",
            args.base_config,
            "--experiment_config",
            str(rel_path),
        ]

        if mode in {"train", "eval", "explain", "ensemble"}:
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
        result = subprocess.run(cmd)

        if result.returncode != 0:
            raise RuntimeError(f"Run failed: {experiment_name}")

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