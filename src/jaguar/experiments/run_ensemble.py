import argparse
from pathlib import Path

import pandas as pd

from jaguar.config import PATHS
from jaguar.utils.utils import ensure_dir
from jaguar.utils.utils_experiments import deep_update, load_toml_config
from jaguar.utils.utils_evaluation import (
    build_query_gallery_retrieval_state_from_sim,
    evaluate_query_gallery_retrieval,
)
from jaguar.logging.wandb_logger import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_ensemble_config,
    log_wandb_ensemble_results,
)
from jaguar.models.ensemble import create_simple_ensemble
from jaguar.models.fusion_suite import build_fusion_suite_results
from jaguar.analysis.kaggle_ensemble.ensemble_analysis import (
    build_qualitative_review_df,
    build_topk_candidates_df,
)
from jaguar.utils.utils_ensemble import (
    build_compute_summary_df,
    build_ensemble_results_long_df,
    build_per_identity_gain_df,
    build_per_query_comparison_df,
    build_protocol_comparison_df,
    compute_oracle_from_query_dfs,
    rank1_overlap_from_query_dfs,
)


def parse_args():
    """Parse command-line arguments for the ensemble runner."""
    parser = argparse.ArgumentParser(description="Run ensemble experiment")
    parser.add_argument(
        "--base_config",
        type=str,
        help="Path to the base config TOML file, relative to PATHS.configs and without .toml",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to the experiment override TOML file, relative to PATHS.configs and without .toml",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Optional name of the experiment (used for logging/checkpoints)",
    )
    return parser.parse_args()


def evaluate_method_from_sim(
    sim_matrix,
    query_global_indices,
    gallery_global_indices,
    query_labels,
    gallery_labels,
    split_df,
):
    """Evaluate one method from a query-gallery similarity matrix."""
    retrieval = build_query_gallery_retrieval_state_from_sim(
        sim_matrix=sim_matrix,
        query_global_indices=query_global_indices,
        gallery_global_indices=gallery_global_indices,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        split_df=split_df,
    )
    query_df, metrics = evaluate_query_gallery_retrieval(retrieval)
    topk_df = build_topk_candidates_df(retrieval, top_k=3)
    return retrieval, query_df, metrics, topk_df


def save_fusion_artifacts(
    run_dir: Path,
    gallery_protocol: str,
    fusion_name: str,
    artifacts: dict,
) -> None:
    """Save tabular fusion artifacts to disk."""
    for artifact_name, artifact_value in artifacts.items():
        if isinstance(artifact_value, pd.DataFrame):
            artifact_value.to_csv(
                run_dir / f"{artifact_name}__{fusion_name}__{gallery_protocol}.csv",
                index=False,
            )


def run_one_ensemble_protocol(
    config: dict,
    exp_name: str,
    experiment_group: str | None,
    gallery_protocol: str,
    run_dir: Path,
) -> dict:
    """Run one ensemble evaluation protocol and return all summary artifacts."""
    protocol_config = deep_update(config, {})
    protocol_config.setdefault("ensemble", {})
    protocol_config["ensemble"]["gallery_protocol"] = gallery_protocol

    print(f"\n=== Running protocol: {gallery_protocol} ===")

    out = create_simple_ensemble(protocol_config, save_dir=None)

    query_labels = out["query_labels"]
    gallery_labels = out["gallery_labels"]
    query_global_indices = out["query_global_indices"]
    gallery_global_indices = out["gallery_global_indices"]
    split_df = out["split_df"]

    per_model_query_dfs = {}
    per_model_metrics = {}
    per_model_topk_dfs = {}

    for name, member_out in out["member_outputs"].items():
        _, query_df, metrics, topk_df = evaluate_method_from_sim(
            sim_matrix=member_out["sim_matrix"],
            query_global_indices=query_global_indices,
            gallery_global_indices=gallery_global_indices,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
            split_df=split_df,
        )

        per_model_query_dfs[name] = query_df
        per_model_metrics[name] = metrics
        per_model_topk_dfs[name] = topk_df

        print(f"{name:20s} mAP={metrics['mAP']:.4f} rank1={metrics['rank1']:.4f}")

    oracle_df, oracle_summary = compute_oracle_from_query_dfs(per_model_query_dfs)

    print(
        f"{'oracle':20s} "
        f"mAP={oracle_summary['oracle_mAP']:.4f} "
        f"rank1={oracle_summary['oracle_rank1']:.4f}"
    )

    fusion_results = build_fusion_suite_results(out, protocol_config)

    fusion_query_dfs = {}
    fusion_metrics = {}
    fusion_topk_dfs = {}
    fusion_artifacts = {}
    fusion_summary_rows = []

    for fusion_res in fusion_results:
        fusion_name = fusion_res["name"]

        _, query_df, metrics, topk_df = evaluate_method_from_sim(
            sim_matrix=fusion_res["sim_matrix"],
            query_global_indices=query_global_indices,
            gallery_global_indices=gallery_global_indices,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
            split_df=split_df,
        )

        fusion_query_dfs[fusion_name] = query_df
        fusion_metrics[fusion_name] = metrics
        fusion_topk_dfs[fusion_name] = topk_df
        fusion_artifacts[fusion_name] = fusion_res.get("artifacts", {})

        fusion_summary_rows.append({
            "fusion_name": fusion_name,
            "mAP": float(metrics["mAP"]),
            "rank1": float(metrics["rank1"]),
            **fusion_res["meta"],
        })

        print(f"{fusion_name:20s} mAP={metrics['mAP']:.4f} rank1={metrics['rank1']:.4f}")

    fusion_summary_df = pd.DataFrame(fusion_summary_rows)
    fusion_summary_df.to_csv(
        run_dir / f"fusion_summary__{gallery_protocol}.csv",
        index=False,
    )

    method_topk_dfs = {
        **per_model_topk_dfs,
        **fusion_topk_dfs,
    }
    for method_name, topk_df in method_topk_dfs.items():
        topk_df.to_csv(
            run_dir / f"topk_candidates__{method_name}__{gallery_protocol}.csv",
            index=False,
        )

    for fusion_name, artifacts in fusion_artifacts.items():
        save_fusion_artifacts(
            run_dir=run_dir,
            gallery_protocol=gallery_protocol,
            fusion_name=fusion_name,
            artifacts=artifacts,
        )

    per_query_comparison_df = build_per_query_comparison_df(
        per_model_query_dfs=per_model_query_dfs,
        fusion_query_dfs=fusion_query_dfs,
    )
    per_query_comparison_df.to_csv(
        run_dir / f"per_query_comparison__{gallery_protocol}.csv",
        index=False,
    )

    best_single_name = max(per_model_metrics.items(), key=lambda x: x[1]["mAP"])[0]
    best_single_query_df = per_model_query_dfs[best_single_name]

    for fusion_name, fusion_query_df in fusion_query_dfs.items():
        per_identity_gain_df = build_per_identity_gain_df(
            query_df_base=best_single_query_df,
            query_df_target=fusion_query_df,
            identity_col="query_label",
            base_name=best_single_name,
            target_name=fusion_name,
        )
        per_identity_gain_df.to_csv(
            run_dir / f"per_identity_gain__{fusion_name}__vs__{best_single_name}__{gallery_protocol}.csv",
            index=False,
        )

    method_query_dfs = {
        **per_model_query_dfs,
        **fusion_query_dfs,
    }

    if "score_fusion" in fusion_query_dfs:
        review_df = build_qualitative_review_df(
            per_query_comparison_df=per_query_comparison_df,
            split_df=split_df,
            method_query_dfs=method_query_dfs,
            method_topk_dfs=method_topk_dfs,
            target_method="score_fusion",
            worst_n=10,
        )
        review_df.to_csv(
            run_dir / f"qualitative_review_worst_score_fusion__{gallery_protocol}.csv",
            index=False,
        )

        print("\nScore fusion vs best single:")
        print(
            per_query_comparison_df[
                [
                    "score_fusion_beats_best_model",
                    "score_fusion_matches_best_model",
                ]
            ].mean(numeric_only=True)
        )

    results_long_df = build_ensemble_results_long_df(
        exp_name=exp_name,
        experiment_group=experiment_group,
        gallery_protocol=gallery_protocol,
        per_model_metrics=per_model_metrics,
        fusion_metrics=fusion_metrics,
        oracle_summary=oracle_summary,
    )
    results_long_df.to_csv(run_dir / f"results_long__{gallery_protocol}.csv", index=False)

    compute_summary_df = build_compute_summary_df(
        out=out,
        fusion_results=fusion_results,
        members_cfg=config["members"],
    )
    compute_summary_df.to_csv(
        run_dir / f"compute_summary__{gallery_protocol}.csv",
        index=False,
    )

    return {
        "gallery_protocol": gallery_protocol,
        "compute_summary_df": compute_summary_df,
        "per_model_query_dfs": per_model_query_dfs,
        "per_model_metrics": per_model_metrics,
        "fusion_query_dfs": fusion_query_dfs,
        "fusion_metrics": fusion_metrics,
        "oracle_df": oracle_df,
        "oracle_summary": oracle_summary,
        "per_query_comparison_df": per_query_comparison_df,
        "fusion_summary_df": fusion_summary_df,
        "results_long_df": results_long_df,
    }


def main():
    """Run one or more ensemble gallery protocols and save evaluation outputs."""
    args = parse_args()

    base_config = load_toml_config(args.base_config)
    experiment_config = load_toml_config(args.experiment_config)
    config = deep_update(base_config, experiment_config)

    if args.experiment_name is not None:
        config.setdefault("ensemble", {})
        config["ensemble"]["name"] = args.experiment_name

    exp_name = config.get("ensemble", {}).get("name")
    if not exp_name:
        raise ValueError("Missing ensemble.name in config.")

    gallery_protocol = config.get("ensemble", {}).get("gallery_protocol")
    if not gallery_protocol:
        raise ValueError("Missing ensemble.gallery_protocol in config.")

    if gallery_protocol == "both":
        gallery_protocols = ["trainval_gallery", "valonly_gallery"]
    elif gallery_protocol in {"trainval_gallery", "valonly_gallery"}:
        gallery_protocols = [gallery_protocol]
    else:
        raise ValueError(
            "ensemble.gallery_protocol must be one of "
            "{'trainval_gallery', 'valonly_gallery', 'both'}"
        )

    experiment_group = config.get("output", {}).get("experiment_group")

    if experiment_group:
        run_dir = PATHS.runs / experiment_group / exp_name
    else:
        run_dir = PATHS.runs / exp_name
    ensure_dir(run_dir)

    use_wandb = len(gallery_protocols) == 1
    wandb_run = None

    if use_wandb:
        wandb_run = init_wandb_run(
            config=config,
            run_dir=run_dir,
            exp_name=exp_name,
            experiment_group=experiment_group,
            job_type="ensemble",
        )
        log_wandb_ensemble_config(wandb_run, config)

    all_protocol_results = []

    for protocol in gallery_protocols:
        protocol_result = run_one_ensemble_protocol(
            config=config,
            exp_name=exp_name,
            experiment_group=experiment_group,
            gallery_protocol=protocol,
            run_dir=run_dir,
        )
        all_protocol_results.append(protocol_result)

    all_results_long_df = pd.concat(
        [res["results_long_df"] for res in all_protocol_results],
        ignore_index=True,
    )
    all_results_long_df.to_csv(run_dir / "results_long_all_protocols.csv", index=False)

    print("\nAll protocol results:")
    print(all_results_long_df.to_string(index=False))

    comparison_df = build_protocol_comparison_df(all_results_long_df)
    comparison_df.to_csv(run_dir / "protocol_comparison.csv", index=False)

    print("\nProtocol comparison:")
    print(comparison_df.to_string(index=False))

    if use_wandb:
        protocol_result = all_protocol_results[0]

        fusion_metrics = protocol_result["fusion_metrics"]
        oracle_df = protocol_result["oracle_df"]
        oracle_summary = protocol_result["oracle_summary"]
        per_model_query_dfs = protocol_result["per_model_query_dfs"]
        fusion_query_dfs = protocol_result["fusion_query_dfs"]

        if "score_fusion" not in fusion_query_dfs or "embedding_concat" not in fusion_query_dfs:
            raise KeyError("Expected score_fusion and embedding_concat in fusion_query_dfs for W&B logging.")

        score_query_df = fusion_query_dfs["score_fusion"]
        score_metrics = fusion_metrics["score_fusion"]

        emb_query_df = fusion_query_dfs["embedding_concat"]
        emb_metrics = fusion_metrics["embedding_concat"]

        member_names = list(per_model_query_dfs.keys())
        if len(member_names) == 2:
            overlap_summary = rank1_overlap_from_query_dfs(
                per_model_query_dfs[member_names[0]],
                per_model_query_dfs[member_names[1]],
                name_a=member_names[0],
                name_b=member_names[1],
            )
            overlap_summary.to_csv(
                run_dir / f"rank1_overlap__{member_names[0]}__vs__{member_names[1]}__{gallery_protocols[0]}.csv",
                index=False,
            )

        print(
            f"{exp_name:24s} "
            f"{gallery_protocols[0]:18s} "
            f"score_mAP={score_metrics['mAP']:.4f} "
            f"score_rank1={score_metrics['rank1']:.4f} "
            f"oracle_mAP={oracle_summary['oracle_mAP']:.4f}"
        )

        log_wandb_ensemble_results(
            run=wandb_run,
            config=config,
            exp_name=exp_name,
            score_metrics=score_metrics,
            emb_metrics=emb_metrics,
            oracle_summary=oracle_summary,
            oracle_df=oracle_df,
            score_query_df=score_query_df,
            emb_query_df=emb_query_df,
            per_model_query_dfs=per_model_query_dfs,
        )

        finish_wandb_run(
            run=wandb_run,
            best_epoch=None,
            best_score=float(score_metrics["mAP"]),
            monitor_metric="ensemble/score_mAP",
            best_metrics=None,
            epochs_completed=1,
            total_train_time_sec=0.0,
        )


if __name__ == "__main__":
    main()