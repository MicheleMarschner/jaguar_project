import argparse
import tomllib
import tomli_w
import torch
from pathlib import Path

from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.config import DEVICE
from jaguar.retrieval.retrieval_runner import (
    run_retrieval_experiment, 
    generate_retrieval_experiments
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to trained model checkpoint folder",
    )

    parser.add_argument(
        "--retrieval_config",
        type=str,
        required=True,
        help="Retrieval experiment config (configs/experiments/...)",
    )

    return parser.parse_args()


# --------------Load training config from checkpoint--------------

def load_checkpoint_config(checkpoint_dir):
    cfg_path = checkpoint_dir / "config_leaderboard_exp.toml"

    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    with open(cfg_path, "rb") as f:
        config = tomllib.load(f)

    return config

# --------------Load trained model--------------

def load_model(checkpoint_dir, config):
    ckpt_path = checkpoint_dir / "best_model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    print(f"Loading checkpoint {ckpt_path}")

    model = JaguarIDModel(
        backbone_name=config["model"]["backbone_name"],
        num_classes=31, # we cna use a fixed number of classes 
        head_type=config["model"]["head_type"],
        device=DEVICE,
        emb_dim=config["model"]["emb_dim"],
        use_projection=config["model"]["use_projection"],
        use_forward_features=config["model"]["use_forward_features"],
    )

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    return model

def main():

    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    # Load checkpoint 
    config = load_checkpoint_config(checkpoint_dir)
     #Load experiment config
    with open(args.retrieval_config, "rb") as f:
        retrieval_cfg = tomllib.load(f)

    experiment_meta = retrieval_cfg["experiment"]
    print(experiment_meta)

    # Generate retreiavl configs
    generated_paths = generate_retrieval_experiments(
        retrieval_cfg,
        output_root=Path("configs/_generated")
    )

    # Load model
    model = load_model(checkpoint_dir, config)

    # Run experiments
    for cfg_path in generated_paths:

        run_name = cfg_path.stem

        print(f"\nRunning retrieval experiment: {run_name}")

        with open(cfg_path, "rb") as f:
            run_cfg = tomllib.load(f)

        output_dir = checkpoint_dir / "retrieval_eval" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save generated config for reproducibility
        with open(output_dir / "retrieval_config_used.toml", "wb") as f:
            tomli_w.dump(run_cfg, f)

        run_retrieval_experiment(
            model=model,
            config=config,
            run_cfg=run_cfg["evaluation"],
            checkpoint_dir=output_dir
        )

if __name__ == "__main__":
    main()
