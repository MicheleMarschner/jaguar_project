import argparse
from pathlib import Path
from jaguar.experiments.experiment_output import save_requested_outputs
import torch

from jaguar.config import DATA_STORE, PATHS
from jaguar.logging.wandb_logger import init_wandb_run, log_wandb_background_sensitivity_results
from jaguar.utils.utils import resolve_path
from jaguar.utils.utils_experiments import load_toml_config, deep_update, load_toml_from_path
from jaguar.utils.utils_xai import build_bg_sensitivity_summaries
from jaguar.xai.foreground_contribution import (
    run_foreground_contribution_analysis,
    run_bg_vs_jaguar_stress_analysis,
    save_bg_sensitivity_outputs,
)
from jaguar.datasets.JaguarDataset import MaskAwareJaguarDataset
from jaguar.utils.utils_evaluation import (
    build_original_gallery_base,
    select_val_samples_from_emb_rows,
)
from jaguar.utils.utils_xai import get_val_query_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Run background sensitivity analysis")
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    base_config = load_toml_config(args.base_config)
    experiment_config = load_toml_config(args.experiment_config)
    config = deep_update(base_config, experiment_config)
    exp_name = config["evaluation"]["experiment_name"]
    experiment_group = config.get("output", {}).get("experiment_group")

    checkpoint_dir = PATHS.checkpoints / config["evaluation"]["checkpoint_dir"]
    train_config = load_toml_from_path(checkpoint_dir / "config_leaderboard_exp.toml")

    config.setdefault("evaluation", {})
    config["evaluation"]["experiment_name"] = args.experiment_name

    manifest_dir = resolve_path("fiftyone/splits_curated", DATA_STORE)
    run_name = args.experiment_name

    # run artifact directory
    if experiment_group:
        suffix = Path(experiment_group) / exp_name
        
    else:
        suffix =  exp_name
    
    run_dir = PATHS.runs / suffix
    #config["training"]["save_dir"] = str(checkpoint_dir / suffix)
    results_path = PATHS.results / suffix
    #save_path = resolve_path(f"xai/background_sensitivity/{run_name}", EXPERIMENTS_STORE)
    #results_path = resolve_path(f"xai/background_sensitivity/{run_name}", RESULTS_STORE)
    #ensure_dir(save_path)
    #ensure_dir(results_path)

    run = init_wandb_run(
        config=config,
        run_dir=run_dir,
        exp_name=config["evaluation"]["experiment_name"],
        experiment_group=config.get("output", {}).get("experiment_group"),
        job_type="eval",
    )

    base = build_original_gallery_base(
        config=config,
        train_config = train_config,
        checkpoint_dir=checkpoint_dir,
    )

    ctx_orig = base["ctx_orig"]
    gallery_emb = base["gallery_embeddings_orig"]
    gallery_ids = list(base["gallery_labels_orig"])
    gallery_files = list(base["gallery_files_orig"])

    query_emb_rows = get_val_query_indices(
        split_df=ctx_orig.split_df,
        out_root=run_dir,
        n_samples=config["xai"]["n_samples"],
        seed=config["xai"]["seed"],
    )

    samples_list = select_val_samples_from_emb_rows(
        ctx_orig=ctx_orig,
        query_emb_rows=query_emb_rows,
    )

    """
    # load model weights
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tmp_state = ckpt["model_state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in tmp_state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("full model missing:", len(missing), "unexpected:", len(unexpected))
    model = model.to(DEVICE).eval()
    """

    query_ds = MaskAwareJaguarDataset(
        jaguar_model=ctx_orig.model,
        base_root=PATHS.data_train,
        data_root=PATHS.data.parent,
        samples_list=samples_list,
    )
    query_loader = torch.utils.data.DataLoader(
        query_ds,
        batch_size=config["inference"]["batch_size"],
        shuffle=False,
    )

    xai_result = run_foreground_contribution_analysis(
        model=ctx_orig.model,
        query_loader=query_loader,
        results_path=results_path,
    )

    stress_result = run_bg_vs_jaguar_stress_analysis(
        model=ctx_orig.model,
        query_loader=query_loader,
        gallery_emb=gallery_emb,
        gallery_ids=gallery_ids,
        gallery_files=gallery_files,
    )

    logits_res = xai_result["logits_res"]
    similarity_res = xai_result["similarity_res"]
    retrieval_df = stress_result["retrieval_df"]

    summary_result = build_bg_sensitivity_summaries(
        retrieval_df=retrieval_df,
        similarity_res=similarity_res,
    )

    analysis_df = summary_result["analysis_df"]
    retrieval_summary = summary_result["retrieval_summary"]
    similarity_summary = summary_result["similarity_summary"]

    artifacts = {
        "run_dir": run_dir,
        "config": config,
    }
    save_requested_outputs(config, artifacts)

    save_bg_sensitivity_outputs(
        save_path=run_dir,
        results_path=results_path,
        config=config,
        train_config=train_config,
        manifest_dir=manifest_dir,
        ctx_orig=ctx_orig,
        query_ds=query_ds,
        query_emb_rows=query_emb_rows,
        logits_res=logits_res,
        similarity_res=similarity_res,
        retrieval_df=retrieval_df,
        analysis_df=analysis_df,
        retrieval_summary=retrieval_summary,
        similarity_summary=similarity_summary,
    )

    log_wandb_background_sensitivity_results(run, retrieval_summary, similarity_summary, analysis_df)
    if run is not None:
        run.finish()

    print(f"[Done] background sensitivity: {run_name}")
    print(f"[Saved] {run_dir}")


if __name__ == "__main__":
    main()
