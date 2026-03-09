import argparse
from pathlib import Path

from jaguar.analysis.aggregate import aggregate_experiment_outputs
from jaguar.analysis.experiment_analysis import run_experiment_analysis
from jaguar.analysis.split_analysis import run_split_diagnostics
from jaguar.analysis.burst_analysis import run_burst_analysis
from jaguar.analysis.xai_metrics_analysis import run_xai_metrics_analysis
from jaguar.analysis.xai_similarity_analysis import run_xai_similarity_analysis
from jaguar.analysis.xai_background_analysis import run_xai_background_sensitivity
from jaguar.analysis.aggregate import (
    aggregate_experiment_outputs,
    detect_output_profile,
    discover_experiment_groups,
)
from jaguar.config import PATHS


ANALYSIS_TASKS = [
    "aggregate_all",
    "split_diagnostics",
    "burst_analysis",
    "xai_metrics_analysis",
    "xai_similarity_analysis",
    "xai_background_sensitivity",
]

def run_xai_metrics_analysis_task(args) -> None:
    if not args.run_root:
        raise ValueError("--run_root is required for xai_metrics_analysis")

    run_root = Path(args.run_root)

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = PATHS.results / "xai" / "similarity"

    outputs = run_xai_metrics_analysis(
        run_root=run_root,
        save_dir=save_dir,
    )
    print(f"[ANALYSIS] XAI metrics analysis saved: {outputs}")


def run_xai_similarity_analysis_task(args) -> None:
    if not args.run_root:
        raise ValueError("--run_root is required for xai_similarity_analysis")

    run_root = Path(args.run_root)

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = PATHS.results / "xai" / "similarity"

    if args.manifest_dir:
        manifest_dir = Path(args.manifest_dir)
    else:
        manifest_dir = PATHS.data_export / "splits_curated"

    outputs = run_xai_similarity_analysis(
        run_root=run_root,
        save_dir=save_dir,
        manifest_dir=manifest_dir,
        dataset_name=args.dataset_name,
        overlay_model_name=args.overlay_model_name,
        overlay_explainer=args.overlay_explainer,
        n_per_model=args.n_per_model,
    )
    print(f"[ANALYSIS] XAI similarity analysis saved: {outputs}")


def run_xai_background_analysis_task(args) -> None:
    if args.experiments_dir:
        experiments_dir = Path(args.experiments_dir)
    else:
        experiments_dir = PATHS.runs / "xai/background_sensitivity"

    outputs = run_xai_background_sensitivity(
        experiments_dir=experiments_dir,
        #experiments_dir=PATHS.runs / "xai/background_sensitivity",
        # runs = load_runs(PATHS.runs / "xai/background_sensitivity")
    )
    print(f"[ANALYSIS] XAI background analysis saved: {outputs}")



def run_burst_analysis_task(args) -> None:
    if not args.artifacts_dir:
        raise ValueError("--artifacts_dir is required for burst_analysis")

    artifacts_dir = Path(args.artifacts_dir)
    # artifacts_dir = resolve_path( "bursts/burst_groups__within500__cross10000__ph13", EXPERIMENTS_STORE)

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = PATHS.results / "burst_analysis" / artifacts_dir.name

    outputs = run_burst_analysis(
        artifacts_dir=artifacts_dir,
        save_dir=save_dir,
        img_root=PATHS.data_train,
    )
    print(f"[ANALYSIS] Burst analysis saved: {outputs}")


def run_split_diagnostics_task(args) -> None:
    if not args.artifacts_dir:
        raise ValueError("--artifacts_dir is required for split_diagnostics")

    artifacts_dir = Path(args.artifacts_dir)
    # artifacts_dir = resolve_path( 

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = PATHS.results / "split_diagnostics" / artifacts_dir.name

    outputs = run_split_diagnostics(
        artifacts_dir=artifacts_dir,
        save_dir=save_dir,
        img_root=PATHS.data_train,
        manifest_dir=PATHS.data_export / "splits_curated",
        dataset_name=args.dataset_name,
    )
    print(f"[ANALYSIS] Split diagnostics saved: {outputs}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run analysis tasks")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=ANALYSIS_TASKS,
    )
    parser.add_argument("--artifacts_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--dataset_name", type=str, default="jaguar_curated")
    parser.add_argument("--run_root", type=str)
    parser.add_argument("--experiments_dir", type=str)
    parser.add_argument("--manifest_dir", type=str)
    parser.add_argument("--overlay_model_name", type=str)
    parser.add_argument("--overlay_explainer", type=str, default="IG")
    parser.add_argument("--n_per_model", type=int, default=10)
    return parser.parse_args()


SPECIAL_ANALYSIS_TASKS = {
    "split_diagnostics": run_split_diagnostics_task,
    "burst_analysis": run_burst_analysis_task,
    "xai_metrics_analysis": run_xai_metrics_analysis_task,
    "xai_similarity_analysis": run_xai_similarity_analysis_task,
    "xai_background_analysis": run_xai_background_analysis_task,
}

def main():
    args = parse_args()

    if args.task == "aggregate_all":
        experiment_groups = discover_experiment_groups()

        aggregated_count = 0
        analyzed_count = 0
        skipped_count = 0

        if not experiment_groups:
            print("[ANALYSIS] No experiment groups found.")
            return

        for experiment_group in experiment_groups:
            output_profile = detect_output_profile(experiment_group)

            if output_profile is None:
                print(f"[ANALYSIS][WARN] No output.profile found for: {experiment_group} -> skip")
                skipped_count += 1
                continue

            try:
                out_path = aggregate_experiment_outputs(
                    experiment_group=experiment_group,
                    output_profile=output_profile,
                )
                print(f"[ANALYSIS] Saved summary for {experiment_group}: {out_path}")
                aggregated_count += 1

                run_experiment_analysis(
                    experiment_group=experiment_group,
                    output_profile=output_profile,
                    summary_path=out_path,
                )
                analyzed_count += 1
            except Exception as e:
                skipped_count += 1
                print(f"[ANALYSIS][WARN] Failed to aggregate {experiment_group}: {e}")
        return
    
    print(
        f"[ANALYSIS] Done. "
        f"aggregated={aggregated_count} | "
        f"analyzed={analyzed_count} | "
        f"skipped={skipped_count}"
    )
    
    handler = SPECIAL_ANALYSIS_TASKS.get(args.task)
    if handler is not None:
        handler(args)
        return

    raise ValueError(f"Unknown analysis task: {args.task}")
    

if __name__ == "__main__":
    main()