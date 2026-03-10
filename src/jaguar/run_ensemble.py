import argparse

from jaguar.models.ensemble import create_simple_ensemble
from jaguar.utils.utils import load_toml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run ensemble experiment")
    parser.add_argument(
        "--base_config",
        type=str,
        required=True,
        help="Path to the base config TOML file, relative to PATHS.configs and without .toml",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        required=False,
        help="Optional ensemble override TOML, relative to PATHS.configs and without .toml",
    )
    return parser.parse_args()


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
    
    if args.experiment_config is not None:
        experiment_config = load_toml_config(args.experiment_config)
        config = deep_update(base_config, experiment_config)
    else:
        config = base_config

    out = create_simple_ensemble(config, save_dir=None)
    print(f"Built ensemble: {config['ensemble']['name']}")
    print(f"Fused sim shape: {out['fused_sim_matrix'].shape}")


if __name__ == "__main__":
    main()