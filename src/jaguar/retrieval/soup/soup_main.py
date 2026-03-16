import argparse
import tomllib
import torch
from pathlib import Path

from jaguar.retrieval.soup.soup_utils import generate_soup_experiments
from jaguar.retrieval.soup.soup_runner import run_soup_sensitivity
from jaguar.retrieval.soup.soup_grouping import discover_seed_models
from jaguar.retrieval.retrieval_runner import build_val_loader
from jaguar.retrieval.retrieval_main import load_model, load_checkpoint_config
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.config import DEVICE
from jaguar.logging.wandb_logger import (
    init_wandb_run,
    finish_wandb_run,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Path to trained seeds models folder",
        
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        required=True,
        help="Soup sensitivity experiment config (configs/experiments/...)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    root_dir = Path(args.checkpoints_dir)

    with open(args.experiment_config, "rb") as f:
        soup_cfg = tomllib.load(f)

    # Generate run configs
    generated_paths = generate_soup_experiments(
        soup_cfg,
        output_root=Path("configs/_generated")
    )

    # Discover checkpoints
    models = discover_seed_models(root_dir)

    if len(models) == 0:
        raise RuntimeError("No seed models found")

    # Build loader once
    example_checkpoint = Path(models[0]["path"])
    example_config = load_checkpoint_config(example_checkpoint)
    example_model = load_model(example_checkpoint, example_config)
    val_loader, labels = build_val_loader(example_config, example_model)
    
    # Run experiments
    is_stability_round = "stability" in root_dir.name
    for cfg_path in generated_paths:
        run_name = cfg_path.stem
        if is_stability_round and run_name != "stability_analysis": continue
        if not is_stability_round and run_name == "stability_analysis": continue

        print(f"\nRunning soup experiment: {run_name}")
        with open(cfg_path, "rb") as f:
            run_cfg = tomllib.load(f)
            
        # Setup WandB config
        run_cfg.setdefault("logging", {})
        run_cfg["logging"].update({
            "enabled": True,
            "project": "jaguar-reid-soup",
            "online": True
        })

        output_dir = Path("soup_eval") / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize WandB
        wandb_run = init_wandb_run(
            config=run_cfg,
            run_dir=output_dir,
            exp_name=run_name,
            experiment_group=soup_cfg["experiment"]["name"],
        )

        leaderboard_df = run_soup_sensitivity(
            root_dir=root_dir,
            val_loader=val_loader,
            labels=labels,
            run_cfg=run_cfg["evaluation"],
            output_dir=output_dir,
            wandb_run=wandb_run
        )

        # Finish WandB
        top_map = leaderboard_df.iloc[0]["mAP"] if not leaderboard_df.empty else 0
        finish_wandb_run(
            run=wandb_run,
            best_epoch=None,
            best_score=top_map,
            monitor_metric="mAP",
            best_metrics=leaderboard_df.iloc[0].to_dict() if not leaderboard_df.empty else {},
            epochs_completed=1,
            total_train_time_sec=0
        )
        
if __name__ == "__main__":
    main()