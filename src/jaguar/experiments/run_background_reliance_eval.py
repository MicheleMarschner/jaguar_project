import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import pandas as pd
import argparse

from jaguar.config import PATHS
from jaguar.utils.utils_experiments import load_toml_config, deep_update, load_toml_from_path
from jaguar.utils.utils import ensure_dir, resolve_path
from jaguar.utils.utils_evaluation import build_original_gallery_base, build_query_for_setting, build_query_gallery_retrieval_state, evaluate_query_gallery_retrieval
from jaguar.logging.wandb_logger import init_wandb_run, log_wandb_background_reliance_results

def save_retrieval_results(
    save_dir: Path,
    setting: str,
    query_df: pd.DataFrame,
    summary: dict,
) -> dict:
    """
    Saves per-query and summary retrieval outputs for one background setting.
    """
    query_df = query_df.copy()
    query_df["setting"] = setting

    summary = dict(summary)
    summary["setting"] = setting

    setting_dir = save_dir / setting
    ensure_dir(setting_dir)

    query_path = setting_dir / "retrieval_per_query.parquet"
    summary_path = setting_dir / "retrieval_summary.parquet"

    query_df.to_parquet(query_path, index=False)
    pd.DataFrame([summary]).to_parquet(summary_path, index=False)

    return {
        "query_df": query_df, 
        "summary": summary, 
        "query_path": query_path, 
        "summary_path": summary_path,
    }


def aggregate_and_save_background_results(
    all_query_dfs: list[pd.DataFrame],
    all_summaries: list[dict],
    save_dir: Path,
) -> dict:
    """
    Aggregates all background-setting results, computes deltas against the original
    setting, and saves the combined outputs.
    """
    per_query_all = pd.concat(all_query_dfs, ignore_index=True)
    summary_all = pd.DataFrame(all_summaries)

    orig_row = summary_all.loc[summary_all["setting"] == "original"].iloc[0]
    summary_all["delta_mAP_vs_original"] = summary_all["mAP"] - float(orig_row["mAP"])
    summary_all["delta_rank1_vs_original"] = summary_all["rank1"] - float(orig_row["rank1"])

    orig_per_query = (
        per_query_all[per_query_all["setting"] == "original"][
            ["query_idx", "ap", "rank1_correct", "first_pos_rank", "top1_idx", "top1_label", "top1_sim"]
        ]
        .rename(
            columns={
                "ap": "ap_original",
                "rank1_correct": "rank1_correct_original",
                "first_pos_rank": "first_pos_rank_original",
                "top1_idx": "top1_idx_original",
                "top1_label": "top1_label_original",
                "top1_sim": "top1_sim_original",
            }
        )
        .copy()
    )

    per_query_delta = per_query_all.merge(orig_per_query, on="query_idx", how="left")

    per_query_delta["delta_ap_vs_original"] = (
        per_query_delta["ap"] - per_query_delta["ap_original"]
    )

    per_query_delta["rank1_flip_vs_original"] = (
        per_query_delta["rank1_correct"] != per_query_delta["rank1_correct_original"]
    )

    per_query_delta["delta_first_pos_rank_vs_original"] = (
        per_query_delta["first_pos_rank"] - per_query_delta["first_pos_rank_original"]
    )

    per_query_path = save_dir / "background_per_query_all.parquet"
    summary_path = save_dir / "background_summary_all.parquet"
    per_query_delta_path = save_dir / "background_per_query_delta_vs_original.parquet"

    per_query_all.to_parquet(per_query_path, index=False)
    summary_all.to_parquet(summary_path, index=False)
    per_query_delta.to_parquet(per_query_delta_path, index=False)

    return {
        "per_query_all": per_query_all,
        "per_query_delta": per_query_delta,
        "summary_all": summary_all,
        "summary_path": summary_path,
        "per_query_path": per_query_path,
        "per_query_delta_path": per_query_delta_path,
    }


def run_background_reliance_eval(config, save_dir):
    """
    Runs the full background-reliance evaluation by comparing several query background
    settings against the same original gallery.
    """
    checkpoint_dir = PATHS.checkpoints / config["evaluation"]["checkpoint_dir"]
    train_config = load_toml_from_path(checkpoint_dir / "config_leaderboard_exp.toml")
    ensure_dir(save_dir)

    base = build_original_gallery_base(config=config, train_config=train_config, checkpoint_dir=checkpoint_dir)

    ctx_orig = base["ctx_orig"]
    gallery_embeddings_orig = base["gallery_embeddings_orig"]
    gallery_labels_orig = base["gallery_labels_orig"]
    gallery_global_indices_orig = base["gallery_global_indices_orig"]

    settings = ["original", "gray_bg", "blur_bg", "black_bg"]

    if "original" not in settings:
        raise ValueError(
            "'original' must be included in settings because all summary and per-query deltas are computed against it."
        )

    all_query_dfs = []
    all_summaries = []

    for setting in settings:
        query = build_query_for_setting(
            config=config,
            ctx_orig=ctx_orig,
            setting=setting,
        )

        retrieval = build_query_gallery_retrieval_state(
                query_embeddings=query["query_embeddings"],
                gallery_embeddings=gallery_embeddings_orig,
                query_global_indices=query["query_global_indices"],
                gallery_global_indices=gallery_global_indices_orig,
                query_labels=query["query_labels"],
                gallery_labels=gallery_labels_orig,
                split_df=ctx_orig.split_df,
            )
        
        query_df, summary = evaluate_query_gallery_retrieval(retrieval)
        result = save_retrieval_results(save_dir, setting, query_df, summary)
        
        all_query_dfs.append(result["query_df"])
        all_summaries.append(result["summary"])

    result = aggregate_and_save_background_results(
        all_query_dfs=all_query_dfs,
        all_summaries=all_summaries,
        save_dir=save_dir,
    )

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run background reliance evaluation")
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    base_config = load_toml_config(args.base_config)
    experiment_config = load_toml_config(args.experiment_config)
    config = deep_update(base_config, experiment_config)

    config.setdefault("evaluation", {})
    config["evaluation"]["experiment_name"] = args.experiment_name

    experiment_group = config.get("output", {}).get("experiment_group")
    if experiment_group:
        run_dir = PATHS.runs / experiment_group / args.experiment_name
    else:
        run_dir = PATHS.runs / args.experiment_name
    ensure_dir(run_dir)

    run = init_wandb_run(
        config=config,
        run_dir=run_dir,
        exp_name=args.experiment_name,
        experiment_group=experiment_group,
        job_type="eval",
    )

    result = run_background_reliance_eval(config=config, save_dir=run_dir)
    log_wandb_background_reliance_results(run, result)
    if run is not None:
        run.finish()

    print(f"[Done] background reliance eval: {args.experiment_name}")
    print(f"[Saved] {run_dir}")
    print(f"[Summary] {result['summary_path']}")



if __name__ == "__main__":
    main()  