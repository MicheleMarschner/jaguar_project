import argparse
from pathlib import Path

from jaguar.config import DATA_STORE, PATHS, ROUND
from jaguar.utils.utils_experiments import deep_update, load_toml_config
from jaguar.utils.utils_setup import build_habitat_backgrounds, get_burst_paths, get_split_paths
from jaguar.utils.utils import ensure_dirs, read_json_if_exists, resolve_path
from jaguar.utils.utils_setup import init_fiftyone_dataset

SETUP_STEPS = {
    "base": [
        "ensure_output_dirs",
        "ensure_fiftyone_init_dataset",
        "ensure_background_pool",
    ],
    "deduplication": [
        "ensure_output_dirs",
        "ensure_fiftyone_init_dataset",
        "ensure_burst_artifacts",
        "ensure_split_artifacts",
    ],
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run setup steps for an experiment")
    parser.add_argument(
        "--setup_name",
        type=str,
        required=True,
        help="Name of the setup routine",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        required=True,
        help="Base config path/key",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        required=True,
        help="Experiment config path/key",
    )
    return parser.parse_args()


def ensure_output_dirs():
    print("[SETUP] ensure_output_dirs")
    ensure_dirs(PATHS)
    print(f"  -> created all directories")


def ensure_fiftyone_init_dataset(current_setup_config: dict):
    print("[SETUP] ensure_fiftyone_init_dataset")

    use_fiftyone = current_setup_config["data"].get("use_fiftyone", False)
    if not use_fiftyone:
        print("  -> use_fiftyone=False, skip FiftyOne init dataset")
        return

    manifest_dir = resolve_path("fiftyone/init", DATA_STORE)
    samples_path = manifest_dir / "samples.json"
    if samples_path.exists():
        print(f"  -> exists: {samples_path}")
        return

    fo_path = PATHS.data_export / "init" / "metadata.json"
    if fo_path.exists():
        print(f"  -> exists: {fo_path}")
        return

    fo_dataset_name = "jaguar_init"
    train_dir = PATHS.data_train
    csv_file = PATHS.data / "jaguar-re-id/train.csv"

    init_fiftyone_dataset(fo_dataset_name, manifest_dir, csv_file, train_dir)
    print(f"  -> finished building: {manifest_dir}")



def ensure_background_pool():
    print("[SETUP] ensure_background_pool")

    out_dir = resolve_path("backgrounds", DATA_STORE)
    raw_dir = PATHS.data_train
    cutout_dir = PATHS.data_train

    # simple existence criterion: directory exists and contains at least one file
    if out_dir.exists():
        existing_files = [p for p in out_dir.iterdir() if p.is_file()]
        if existing_files:
            print(f"  -> exists with {len(existing_files)} files: {out_dir}")
            return

    print(f"  -> missing or empty, building background pool in: {out_dir}")

    build_habitat_backgrounds(
        raw_dir=raw_dir,
        cutout_dir=cutout_dir,
        out_dir=out_dir,
        n_patches=200,
        patch_size=224,
    )

    print(f"  -> finished building: {out_dir}")


def ensure_burst_artifacts(config):

    from jaguar.preprocessing.burst_discovery import discover_bursts
    bursts_root = get_burst_paths()["write_root"]

    for cfg_path in bursts_root.rglob("config.json"):
        cfg = read_json_if_exists(cfg_path)
        if cfg is not None and cfg.get("round") == ROUND:
            print(f"[SETUP] burst config for {ROUND} already exists -> skip")
            print(f"[SETUP] using: {cfg_path.parent}")
            return cfg_path.parent

    print(f"[SETUP] no burst config for {ROUND} found -> running burst discovery")

    use_fiftyone = config["data"].get("use_fiftyone", False)
    seed = config.get("split", {}).get("seed")

    discover_bursts(
        burst_min_cluster_size=2,
        burst_max_within=500,
        burst_max_cross=10000,
        seed=seed,
        phash_size=8,
        use_fiftyone=use_fiftyone,
    )

    for cfg_path in bursts_root.rglob("config.json"):
        cfg = read_json_if_exists(cfg_path)
        if cfg is not None and cfg.get("round") == ROUND:
            return cfg_path.parent

    raise RuntimeError("Burst discovery ran, but no config.json with matching round was found.")


def ensure_split_artifacts(config) -> Path:
    split_cfg = config.get("split", {})
    curation_cfg = config.get("curation", {})
    use_fiftyone = config["data"].get("use_fiftyone", False)

    split_strategy = split_cfg["strategy"]
    include_duplicates = split_cfg["include_duplicates"]
    val_split_size = split_cfg["val_split_size"]
    seed = split_cfg["seed"]
    train_k = curation_cfg.get("train_k")
    val_k = curation_cfg.get("val_k")
    phash_threshold = curation_cfg.get("phash_threshold")

    
    from jaguar.preprocessing.split_and_curate import create_splits_and_curate
    paths = get_split_paths(
        split_strategy=split_strategy,
        include_duplicates=include_duplicates,
        train_k=train_k,
        val_k=val_k,
        phash_threshold=phash_threshold,
    )

    full_path = paths["full_split_file"]

    print("[SETUP] expected split path:", full_path)
    print("[SETUP] exists:", full_path.exists())
    print("[SETUP] split args:", {
        "split_strategy": split_strategy,
        "include_duplicates": include_duplicates,
        "train_k": train_k,
        "val_k": val_k,
        "phash_threshold": phash_threshold,
        "val_split_size": val_split_size,
        "seed": seed,
    })

    if full_path.exists():
        print("[SETUP] split artifacts already exist -> skip")
        print(f"[SETUP] using: {full_path}")
        return full_path

    print("[SETUP] no matching split artifacts found -> running split_and_curate")
    create_splits_and_curate(
        split_strategy=split_strategy,
        include_duplicates=include_duplicates,
        train_k_per_dedup=train_k,
        val_k_per_dedup=val_k,
        phash_thresh_dedup=phash_threshold,
        val_split_size=val_split_size,
        seed=seed,
        use_fiftyone=use_fiftyone
    )

    if not full_path.exists():
        raise RuntimeError(f"Split creation finished, but file not found: {full_path}")

    return full_path


def run_step(step_name: str, config: dict):
    print(f"[SETUP] {step_name}")

    if step_name == "ensure_output_dirs":
        ensure_output_dirs()
    elif step_name == "ensure_fiftyone_init_dataset":
        ensure_fiftyone_init_dataset(config)
    elif step_name == "ensure_background_pool":
        ensure_background_pool()
    elif step_name == "ensure_burst_artifacts":
        ensure_burst_artifacts(config)
    elif step_name == "ensure_split_artifacts":
        ensure_split_artifacts(config)
    else:
        raise ValueError(f"Unknown setup step: {step_name}")


def main():
    args = parse_args()

    if args.setup_name not in SETUP_STEPS:
        raise ValueError(f"Unknown setup_name: {args.setup_name}")

    steps = SETUP_STEPS[args.setup_name]

    print(f"[SETUP] setup_name = {args.setup_name}")
    print(f"[SETUP] steps = {steps}")

    base_config = load_toml_config(args.base_config)
    experiment_config = load_toml_config(args.experiment_config)
    current_setup_config = deep_update(base_config, experiment_config)

    for step in steps:
        run_step(step, current_setup_config)

    print("[SETUP] done")


if __name__ == "__main__":
    main()