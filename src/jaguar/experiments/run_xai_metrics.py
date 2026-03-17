import argparse

from jaguar.xai.xai_similarity import XAIConfig
from jaguar.xai.xai_metrics import run_xai_similarity_metrics
from jaguar.xai.xai_metrics import run_xai_class_metrics
from jaguar.utils.utils_experiments import load_toml_config, deep_update


def parse_args():
    """Parse command-line arguments for the XAI metric evaluation runner."""
    parser = argparse.ArgumentParser(description="Run XAI metric evaluation")
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    return parser.parse_args()


def build_xai_cfg(config: dict) -> XAIConfig:
    """Build the shared XAI configuration object from the experiment config."""
    return XAIConfig(
        dataset_name=config["xai"]["dataset_name"],
        split_name=config["xai"]["split_name"],
        n_samples=config["xai"]["n_samples"],
        seed=config["xai"]["seed"],
        explainer_names=tuple(config["xai"]["explainer_names"]),
        ig_steps=config["xai"]["ig_steps"],
        ig_internal_bs=config["xai"]["ig_internal_bs"],
        ig_batch_size=config["xai"]["ig_batch_size"],
        pair_types=tuple(config["xai"]["pair_types"]),
        groups=tuple(config["xai"].get("groups", [])),
    )


def main():
    """Load configuration, run the requested XAI metric evaluation, and print the summary."""
    args = parse_args()

    base_config = load_toml_config(args.base_config)
    experiment_config = load_toml_config(args.experiment_config)
    config = deep_update(base_config, experiment_config)

    config.setdefault("evaluation", {})
    cfg = build_xai_cfg(config)
    source_type = config["xai_metrics"]["source_type"]

    if source_type == "similarity":
        summary_df = run_xai_similarity_metrics(config, cfg)
    elif source_type == "class_attribution":
        summary_df = run_xai_class_metrics(config, cfg)
    else:
        raise ValueError(f"Unknown xai_metrics.source_type: {source_type}")

    print("[Done] XAI metric evaluation")
    print(summary_df)


if __name__ == "__main__":
    main()