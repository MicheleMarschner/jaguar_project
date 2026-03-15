import argparse
import subprocess

from jaguar.config import PATHS
from jaguar.utils.utils import ensure_dir
from jaguar.utils.utils_experiments import build_override_for_mode, dict_to_toml, load_toml_config
from jaguar.experiments.experiment_setup import run_setup

def parse_args():
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
        if explain_type == "pair_similarity_metrics":
            return "src/jaguar/experiments/run_xai_metrics.py"
        if explain_type == "background_sensitivity":
            return "src/jaguar/experiments/run_background_sensitivity.py"
        raise ValueError(f"Unknown explain_type: {explain_type}")

    raise ValueError(f"Unknown mode: {mode}")


def run_experiments():
    args = parse_args()
    
    experiment_config = load_toml_config(args.experiment_config)
    base_config = load_toml_config(args.base_config)

    experiment_meta = experiment_config.get("experiment", {})
    runs = experiment_meta.get("runs", [])
    if not runs:
        raise ValueError("No runs found under [[experiment.runs]]")

    experiment_group = experiment_meta.get("name", "experiment")
    generated_dir = PATHS.configs / "_generated" / experiment_group
    ensure_dir(generated_dir)

    mode = experiment_meta.get("mode", "train")

    print(f"Found {len(runs)} runs.")
    print(f"mode = {mode}")

    all_cmds = []

    for i, run_cfg in enumerate(runs, start=1):
        experiment_name = run_cfg["experiment_name"]
        
        override = build_override_for_mode(
            mode=mode,
            run_cfg=run_cfg,
            experiment_meta=experiment_meta,
            base_config=base_config,
        )
        override_text = dict_to_toml(override)

        print(f"\n[{i}/{len(runs)}] {experiment_name}")

        override_path = generated_dir / f"{experiment_name}.toml"
        override_path.write_text(override_text, encoding="utf-8")
        rel_path = override_path.relative_to(PATHS.configs).with_suffix("")

        run_setup(args.base_config, rel_path)
    
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

        print("Running:", " ".join(cmd))
        #result = subprocess.run(cmd)

        #if result.returncode != 0:
        #    raise RuntimeError(f"Run failed: {experiment_name}")

        
        all_cmds.append(" ".join(cmd))
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