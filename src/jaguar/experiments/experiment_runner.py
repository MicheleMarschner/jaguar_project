import argparse
import subprocess

from jaguar.config import PATHS
from jaguar.utils.utils import ensure_dir
from jaguar.utils.utils_experiments import build_ensemble_override, build_training_override, dict_to_toml, load_toml_config


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
            target_script = "src/jaguar/experiments/run_ensemble.py"
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