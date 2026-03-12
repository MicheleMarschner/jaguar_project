import argparse
from jaguar.xai.xai_metrics import run_xai_metrics
from jaguar.xai.xai_similarity import XAIConfig
from jaguar.utils.utils_experiments import load_toml_config, deep_update


def parse_args():
    parser = argparse.ArgumentParser(description="Run XAI metric evaluation")
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--experiment_config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    base_config = load_toml_config(args.base_config)
    experiment_config = load_toml_config(args.experiment_config)
    config = deep_update(base_config, experiment_config)
    config.setdefault("evaluation", {})
    config["evaluation"]["experiment_name"] = args.experiment_name

    cfg = XAIConfig(
        dataset_name=config["xai"]["dataset_name"],
        split_name=config["xai"]["split_name"],
        n_samples=config["xai"]["n_samples"],
        seed=config["xai"]["seed"],
        explainer_names=tuple(config["xai"]["explainer_names"]),
        ig_steps=config["xai"]["ig_steps"],
        ig_internal_bs=config["xai"]["ig_internal_bs"],
        ig_batch_size=config["xai"]["ig_batch_size"],
        pair_types=tuple(config["xai"]["pair_types"]),
    )
    
    summary_df = run_xai_metrics(config, cfg)
        
    print("[Done] XAI metric evaluation")
    print(summary_df)


if __name__ == "__main__":
    main()