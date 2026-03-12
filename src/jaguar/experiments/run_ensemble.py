import argparse
import pandas as pd

from jaguar.config import PATHS
from jaguar.utils.utils_experiments import deep_update, load_toml_config
from jaguar.logging.wandb_logger import finish_wandb_run, init_wandb_run, log_wandb_ensemble_config, log_wandb_ensemble_results
from jaguar.models.ensemble import create_simple_ensemble
from jaguar.utils.utils import ensure_dir
from jaguar.utils.utils_evaluation import (
    build_query_gallery_retrieval_state_from_sim,
    evaluate_query_gallery_retrieval,
)


def compute_oracle_from_query_dfs(
    per_model_query_dfs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    model_names = list(per_model_query_dfs.keys())

    base = per_model_query_dfs[model_names[0]][["query_idx", "query_label"]].copy()

    for name, df in per_model_query_dfs.items():
        base = base.merge(
            df[["query_idx", "ap", "rank1_correct"]].rename(
                columns={
                    "ap": f"ap__{name}",
                    "rank1_correct": f"rank1__{name}",
                }
            ),
            on="query_idx",
            how="left",
        )

    ap_cols = [f"ap__{name}" for name in model_names]
    rank1_cols = [f"rank1__{name}" for name in model_names]

    base["oracle_ap"] = base[ap_cols].max(axis=1, skipna=True)
    base["oracle_rank1"] = base[rank1_cols].any(axis=1)

    summary = {
        "oracle_mAP": float(base["oracle_ap"].mean()),
        "oracle_rank1": float(base["oracle_rank1"].mean()),
    }

    return base, summary


def rank1_overlap_from_query_dfs(
    query_df_a: pd.DataFrame,
    query_df_b: pd.DataFrame,
    name_a: str,
    name_b: str,
) -> pd.DataFrame:
    df = query_df_a[["query_idx", "rank1_correct"]].rename(
        columns={"rank1_correct": f"{name_a}_correct"}
    )
    df = df.merge(
        query_df_b[["query_idx", "rank1_correct"]].rename(
            columns={"rank1_correct": f"{name_b}_correct"}
        ),
        on="query_idx",
        how="inner",
    )

    a = df[f"{name_a}_correct"]
    b = df[f"{name_b}_correct"]

    return pd.DataFrame([
        {"case": "both_correct", "count": int((a & b).sum()), "fraction": float((a & b).mean())},
        {"case": f"only_{name_a}", "count": int((a & ~b).sum()), "fraction": float((a & ~b).mean())},
        {"case": f"only_{name_b}", "count": int((~a & b).sum()), "fraction": float((~a & b).mean())},
        {"case": "both_wrong", "count": int((~a & ~b).sum()), "fraction": float((~a & ~b).mean())},
    ])


def parse_args():
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


def main():
    args = parse_args()

    base_config = load_toml_config(args.base_config)
    experiment_config = load_toml_config(args.experiment_config)
    config = deep_update(base_config, experiment_config)

    # Optionally override experiment name inside config
    if args.experiment_name is not None:
        config.setdefault("experiment", {})
        config["experiment"]["name"] = args.experiment_name

    exp_name = args.experiment_name or config["experiment"]["name"]
    experiment_group = config.get("experiment", {}).get("name")

    if experiment_group:
        run_dir = PATHS.runs / experiment_group / exp_name
    else:
        run_dir = PATHS.runs / exp_name

    ensure_dir(run_dir)


    wandb_run = init_wandb_run(
        config=config,
        run_dir=run_dir,
        exp_name=exp_name,
        experiment_group=experiment_group,
        job_type="ensemble",
    )

    log_wandb_ensemble_config(wandb_run, config)

    out = create_simple_ensemble(config, save_dir=None)

    query_labels = out["query_labels"]
    gallery_labels = out["gallery_labels"]
    query_global_indices = out["query_global_indices"]
    gallery_global_indices = out["gallery_global_indices"]
    split_df = out["split_df"]

    per_model_query_dfs = {}

    for name, member_out in out["member_outputs"].items():
        retrieval = build_query_gallery_retrieval_state_from_sim(
            sim_matrix=member_out["sim_matrix"],
            query_global_indices=query_global_indices,
            gallery_global_indices=gallery_global_indices,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
            split_df=split_df,
        )
        query_df, metrics = evaluate_query_gallery_retrieval(retrieval)
        per_model_query_dfs[name] = query_df

        print(
            f"{name:12s} "
            f"mAP={metrics['mAP']:.4f} "
            f"rank1={metrics['rank1']:.4f}"
        )

    oracle_df, oracle_summary = compute_oracle_from_query_dfs(per_model_query_dfs)

    print(
        f"{'oracle':12s} "
        f"mAP={oracle_summary['oracle_mAP']:.4f} "
        f"rank1={oracle_summary['oracle_rank1']:.4f}"
    )

    score_retrieval = build_query_gallery_retrieval_state_from_sim(
        sim_matrix=out["fused_sim_matrix"],
        query_global_indices=query_global_indices,
        gallery_global_indices=gallery_global_indices,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        split_df=split_df,
    )
    score_query_df, score_metrics = evaluate_query_gallery_retrieval(score_retrieval)

    print(
        f"{'score_fusion':12s} "
        f"mAP={score_metrics['mAP']:.4f} "
        f"rank1={score_metrics['rank1']:.4f}"
    )

    emb_retrieval = build_query_gallery_retrieval_state_from_sim(
        sim_matrix=out["fused_embedding_sim_matrix"],
        query_global_indices=query_global_indices,
        gallery_global_indices=gallery_global_indices,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        split_df=split_df,
    )
    emb_query_df, emb_metrics = evaluate_query_gallery_retrieval(emb_retrieval)

    print(
        f"{'emb_fusion':12s} "
        f"mAP={emb_metrics['mAP']:.4f} "
        f"rank1={emb_metrics['rank1']:.4f}"
    )

    member_names = list(per_model_query_dfs.keys())
    overlap_summary = None
    if len(member_names) == 2:
        overlap_summary = rank1_overlap_from_query_dfs(
            per_model_query_dfs[member_names[0]],
            per_model_query_dfs[member_names[1]],
            name_a=member_names[0],
            name_b=member_names[1],
        )
        print("\nRank1 overlap:")
        print(overlap_summary.to_string(index=False))

    print(
        f"{exp_name:24s} "
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