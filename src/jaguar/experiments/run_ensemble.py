import argparse
from pathlib import Path
import pandas as pd
from jaguar.utils.utils_ensemble import build_ensemble_results_long_df, build_protocol_comparison_df
import numpy as np

from jaguar.config import PATHS
from jaguar.utils.utils_experiments import deep_update, load_toml_config
from jaguar.logging.wandb_logger import finish_wandb_run, init_wandb_run, log_wandb_ensemble_config, log_wandb_ensemble_results
from jaguar.models.ensemble import create_simple_ensemble
from jaguar.utils.utils import ensure_dir
from jaguar.utils.utils_evaluation import (
    build_query_gallery_retrieval_state_from_sim,
    evaluate_query_gallery_retrieval,
)
from jaguar.models.fusion_suite import build_fusion_suite_results
from jaguar.analysis.kaggle_ensemble.ensemble_analysis import build_topk_candidates_df, build_qualitative_review_df


def build_per_query_comparison_df(
    per_model_query_dfs: dict[str, pd.DataFrame],
    fusion_query_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge per-query results from single models and fusion methods into one comparison table."""
    model_names = list(per_model_query_dfs.keys())
    fusion_names = list(fusion_query_dfs.keys())

    if not model_names:
        raise ValueError("per_model_query_dfs must not be empty")

    base = per_model_query_dfs[model_names[0]][["query_idx", "query_label"]].copy()

    for name, df in per_model_query_dfs.items():
        base = base.merge(
            df[["query_idx", "ap", "rank1_correct", "first_pos_rank", "top1_idx", "top1_label", "top1_sim"]].rename(
                columns={
                    "ap": f"ap__{name}",
                    "rank1_correct": f"rank1__{name}",
                    "first_pos_rank": f"first_pos_rank__{name}",
                    "top1_idx": f"top1_idx__{name}",
                    "top1_label": f"top1_label__{name}",
                    "top1_sim": f"top1_sim__{name}",
                }
            ),
            on="query_idx",
            how="left",
        )

    for name, df in fusion_query_dfs.items():
        base = base.merge(
            df[["query_idx", "ap", "rank1_correct", "first_pos_rank", "top1_idx", "top1_label", "top1_sim"]].rename(
                columns={
                    "ap": f"ap__{name}",
                    "rank1_correct": f"rank1__{name}",
                    "first_pos_rank": f"first_pos_rank__{name}",
                    "top1_idx": f"top1_idx__{name}",
                    "top1_label": f"top1_label__{name}",
                    "top1_sim": f"top1_sim__{name}",
                }
            ),
            on="query_idx",
            how="left",
        )

    model_ap_cols = [f"ap__{name}" for name in model_names]
    model_rank1_cols = [f"rank1__{name}" for name in model_names]

    base["oracle_ap"] = base[model_ap_cols].max(axis=1, skipna=True)
    base["oracle_rank1"] = base[model_rank1_cols].any(axis=1)

    base["best_model"] = base[model_ap_cols].idxmax(axis=1).str.replace("ap__", "", regex=False)
    base["best_model_ap"] = base[model_ap_cols].max(axis=1, skipna=True)

    if "score_fusion" in fusion_query_dfs:
        base["score_fusion_delta_vs_best_model"] = base["ap__score_fusion"] - base["best_model_ap"]
        base["score_fusion_delta_vs_oracle"] = base["ap__score_fusion"] - base["oracle_ap"]
        base["score_fusion_beats_best_model"] = base["ap__score_fusion"] > base["best_model_ap"]
        base["score_fusion_matches_best_model"] = np.isclose(
            base["ap__score_fusion"], base["best_model_ap"], atol=1e-12
        )

    return base


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

def build_per_identity_gain_df(
    query_df_base: pd.DataFrame,
    query_df_target: pd.DataFrame,
    identity_col: str = "query_label",
    base_name: str = "best_single",
    target_name: str = "ensemble",
) -> pd.DataFrame:
    """Aggregate per-query gains to per-identity comparison."""
    base = query_df_base[[identity_col, "ap", "rank1_correct"]].rename(
        columns={
            "ap": f"ap__{base_name}",
            "rank1_correct": f"rank1__{base_name}",
        }
    )
    target = query_df_target[[identity_col, "ap", "rank1_correct"]].rename(
        columns={
            "ap": f"ap__{target_name}",
            "rank1_correct": f"rank1__{target_name}",
        }
    )

    df = pd.concat([base, target.drop(columns=[identity_col])], axis=1)
    grouped = df.groupby(identity_col, dropna=False).agg(
        queries=(identity_col, "size"),
        base_ap_mean=(f"ap__{base_name}", "mean"),
        target_ap_mean=(f"ap__{target_name}", "mean"),
        base_rank1_mean=(f"rank1__{base_name}", "mean"),
        target_rank1_mean=(f"rank1__{target_name}", "mean"),
    ).reset_index()

    grouped["delta_ap"] = grouped["target_ap_mean"] - grouped["base_ap_mean"]
    grouped["delta_rank1"] = grouped["target_rank1_mean"] - grouped["base_rank1_mean"]
    return grouped.sort_values("delta_ap", ascending=False)


def build_compute_summary_df(
    out: dict,
    fusion_results: list[dict[str, any]],
    members_cfg: list[dict],
) -> pd.DataFrame:
    """Save simple compute/latency proxy information for later reporting."""
    member_rows = []
    for member_cfg in members_cfg:
        name = member_cfg["name"]
        member_out = out["member_outputs"][name]
        member_rows.append(
            {
                "kind": "member",
                "method_name": name,
                "family": "single_model",
                "n_members_used": 1,
                "embedding_dim": int(member_out["query_embeddings"].shape[1]),
                "query_count": int(member_out["query_embeddings"].shape[0]),
                "gallery_count": int(member_out["gallery_embeddings"].shape[0]),
            }
        )

    fusion_rows = []
    for res in fusion_results:
        fusion_rows.append(
            {
                "kind": "fusion",
                "method_name": res["name"],
                "family": res["meta"].get("family"),
                "n_members_used": int(res["meta"].get("n_members_used", len(out["member_outputs"]))),
                "embedding_dim": res["meta"].get("embedding_dim"),
                "query_count": int(out["query_labels"].shape[0]),
                "gallery_count": int(out["gallery_labels"].shape[0]),
            }
        )

    return pd.DataFrame(member_rows + fusion_rows)


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
        per_model_metrics[name] = metrics
        per_model_topk_dfs[name] = build_topk_candidates_df(retrieval, top_k=3)

        print(
            f"{name:20s} "
            f"mAP={metrics['mAP']:.4f} "
            f"rank1={metrics['rank1']:.4f}"
        )

    oracle_df, oracle_summary = compute_oracle_from_query_dfs(per_model_query_dfs)

    print(
        f"{'oracle':20s} "
        f"mAP={oracle_summary['oracle_mAP']:.4f} "
        f"rank1={oracle_summary['oracle_rank1']:.4f}"
    )

    fusion_results = build_fusion_suite_results(out, protocol_config)

    fusion_query_dfs = {}
    fusion_metrics = {}
    fusion_artifacts = {}
    fusion_summary_rows = []
    fusion_topk_dfs = {}

    for fusion_res in fusion_results:
        fusion_name = fusion_res["name"]

        retrieval = build_query_gallery_retrieval_state_from_sim(
            sim_matrix=fusion_res["sim_matrix"],
            query_global_indices=query_global_indices,
            gallery_global_indices=gallery_global_indices,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
            split_df=split_df,
        )
        query_df, metrics = evaluate_query_gallery_retrieval(retrieval)

        fusion_query_dfs[fusion_name] = query_df
        fusion_metrics[fusion_name] = metrics
        fusion_topk_dfs[fusion_name] = build_topk_candidates_df(retrieval, top_k=3)
        fusion_artifacts[fusion_name] = fusion_res.get("artifacts", {})

        fusion_summary_rows.append({
            "fusion_name": fusion_name,
            "mAP": float(metrics["mAP"]),
            "rank1": float(metrics["rank1"]),
            **fusion_res["meta"],
        })

        print(
            f"{fusion_name:20s} "
            f"mAP={metrics['mAP']:.4f} "
            f"rank1={metrics['rank1']:.4f}"
        )

    fusion_summary_df = pd.DataFrame(fusion_summary_rows)
    
    fusion_summary_df.to_csv(
        run_dir / f"fusion_summary__{gallery_protocol}.csv",
        index=False,
    )

    for method_name, topk_df in method_topk_dfs.items():
        topk_df.to_csv(
            run_dir / f"topk_candidates__{method_name}__{gallery_protocol}.csv",
            index=False,
        )

    for fusion_name, artifacts in fusion_artifacts.items():
        for artifact_name, artifact_value in artifacts.items():
            if isinstance(artifact_value, pd.DataFrame):
                artifact_value.to_csv(
                    run_dir / f"{artifact_name}__{fusion_name}__{gallery_protocol}.csv",
                    index=False,
                )

    per_query_comparison_df = build_per_query_comparison_df(
        per_model_query_dfs=per_model_query_dfs,
        fusion_query_dfs=fusion_query_dfs,
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
    method_topk_dfs = {
        **per_model_topk_dfs,
        **fusion_topk_dfs,
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

        print("\nWorst score-fusion losses vs best single:")
        cols = [
            "query_idx",
            "query_label",
            "best_model",
            "best_model_ap",
            "ap__score_fusion",
            "score_fusion_delta_vs_best_model",
            "score_fusion_delta_vs_oracle",
        ]
        print(
            per_query_comparison_df.sort_values("score_fusion_delta_vs_best_model")[cols]
            .head(10)
            .to_string(index=False)
        )

    per_query_comparison_df.to_csv(
        run_dir / f"per_query_comparison__{gallery_protocol}.csv",
        index=False,
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
        "out": out,
        "compute_summary_df": compute_summary_df,
        "per_model_query_dfs": per_model_query_dfs,
        "per_model_metrics": per_model_metrics,
        "fusion_query_dfs": fusion_query_dfs,
        "fusion_metrics": fusion_metrics,
        "fusion_artifacts": fusion_artifacts,
        "oracle_df": oracle_df,
        "oracle_summary": oracle_summary,
        "per_query_comparison_df": per_query_comparison_df,
        "fusion_summary_df": fusion_summary_df,
        "results_long_df": results_long_df,
    }




def main():
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

    print(exp_name)
    print(experiment_group)
    print(gallery_protocols)

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


    """
    # Jaju query ids from your analysis
    target_query_ids = [555, 556]

    # per_model_query_dfs should already exist in run_ensemble.py
    # e.g. {"EVA-02": df1, "MiewID": df2, "ConvNeXt-V2": df3}
    rows = []

    for model_name, df in per_model_query_dfs.items():
        sub = df[df["query_idx"].isin(target_query_ids)].copy()
        sub = sub[["query_idx", "query_label", "rank1_correct", "first_pos_rank", "ap", "top1_label", "top1_idx"]]
        sub["model"] = model_name
        rows.append(sub)

    jaju_rank_df = pd.concat(rows, ignore_index=True)
    jaju_rank_df = jaju_rank_df.sort_values(["query_idx", "first_pos_rank", "model"])

    print(jaju_rank_df.to_string(index=False))

    rows = []

    for method_name, df in {**per_model_query_dfs, **fusion_query_dfs}.items():
        sub = df[df["query_idx"].isin(target_query_ids)].copy()
        sub = sub[["query_idx", "query_label", "rank1_correct", "first_pos_rank", "ap", "top1_label", "top1_idx"]]
        sub["method"] = method_name
        rows.append(sub)

    jaju_rank_df = pd.concat(rows, ignore_index=True)
    jaju_rank_df = jaju_rank_df.sort_values(["query_idx", "first_pos_rank", "method"])

    print(jaju_rank_df.to_string(index=False))
    
    pivot = jaju_rank_df.pivot_table(
        index=["query_idx", "query_label"],
        columns="method",
        values="first_pos_rank",
        aggfunc="first",
    )

    print(pivot)


    target_query_ids = [1144, 1213, 1217, 1221, 1215, 1214]

    rows = []

    for method_name, df in {**per_model_query_dfs, **fusion_query_dfs}.items():
        sub = df[df["query_idx"].isin(target_query_ids)].copy()
        sub = sub[
            [
                "query_idx",
                "query_label",
                "rank1_correct",
                "first_pos_rank",
                "ap",
                "top1_label",
                "top1_idx",
            ]
        ]
        sub["method"] = method_name
        rows.append(sub)

    medrosa_rank_df = pd.concat(rows, ignore_index=True)
    medrosa_rank_df = medrosa_rank_df.sort_values(["query_idx", "first_pos_rank", "method"])

    print(medrosa_rank_df.to_string(index=False))

    pivot_first_pos = medrosa_rank_df.pivot_table(
        index=["query_idx", "query_label"],
        columns="method",
        values="first_pos_rank",
        aggfunc="first",
    )

    print(pivot_first_pos)

    pivot_ap = medrosa_rank_df.pivot_table(
        index=["query_idx", "query_label"],
        columns="method",
        values="ap",
        aggfunc="first",
    )

    print(pivot_ap)
    """

if __name__ == "__main__":
    main()