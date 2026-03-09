import argparse
import tomllib

from jaguar.config import PATHS
from jaguar.XAI.run_xai_similarity import XAIConfig, run_xai


def parse_args():
    parser = argparse.ArgumentParser(description="Run XAI experiment")
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--experiment_config", type=str, required=True)
    return parser.parse_args()


def load_toml_config(config_name: str) -> dict:
    with open(PATHS.configs / f"{config_name}.toml", "rb") as f:
        return tomllib.load(f)


def deep_update(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def main():
    args = parse_args()

    base_config = load_toml_config(args.base_config)
    experiment_config = load_toml_config(args.experiment_config)
    config = deep_update(base_config, experiment_config)

    xai_cfg = config["xai"]
    xai_type = xai_cfg["type"]

    if xai_type != "similarity":
        raise ValueError(f"Unknown xai.type: {xai_type}")

    cfg = XAIConfig(
        dataset_name=xai_cfg["dataset_name"],
        split_name=xai_cfg["split_name"],
        n_samples=xai_cfg["n_samples"],
        seed=xai_cfg.get("seed", 51),
        ig_steps=xai_cfg.get("ig_steps", 10),
        ig_internal_bs=xai_cfg.get("ig_internal_bs", 32),
        ig_batch_size=xai_cfg.get("ig_batch_size", 32),
        pair_types=tuple(xai_cfg.get("pair_types", ["easy_pos", "hard_neg"])),
        out_root=PATHS.runs / "xai" / "similarity",
    )

    model_name = xai_cfg["model_name"]
    explainer_names = xai_cfg.get("explainer_names", ["IG", "GradCAM"])

    run_xai(
        cfg,
        model_name=model_name,
        explainer_names=explainer_names,
        split_data_path=config["data"]["split_data_path"],
    )


if __name__ == "__main__":
    main()