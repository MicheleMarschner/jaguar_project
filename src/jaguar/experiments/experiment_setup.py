import argparse
from pathlib import Path
from typing import Any

from jaguar.config import DATA_STORE, PATHS, ROUND
from jaguar.utils.utils_experiments import deep_update, load_toml_config
from jaguar.utils.utils_setup import build_habitat_backgrounds, get_burst_paths, get_split_paths
from jaguar.utils.utils import ensure_dirs, read_json_if_exists, resolve_path
from jaguar.utils.utils_setup import init_fiftyone_dataset


SETUP_STEPS = [
    "ensure_output_dirs",
    "ensure_fiftyone_init_dataset",
    "ensure_background_pool",
    "ensure_burst_artifacts",
    "ensure_split_artifacts",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run unified setup for an experiment")
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
    print("  -> created all directories")


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
        n_patches=100,
        patch_size=224,
    )

    print(f"  -> finished building: {out_dir}")


def ensure_burst_artifacts(current_setup_config: dict):
    print("[SETUP] ensure_burst_artifacts")

    # Skip unless dedup/split logic is actually needed
    split_cfg = current_setup_config.get("split", {})
    curation_cfg = current_setup_config.get("curation", {})

    needs_bursts = (
        not split_cfg.get("include_duplicates", True)
        or curation_cfg.get("train_k") is not None
        or curation_cfg.get("val_k") is not None
        or curation_cfg.get("phash_threshold") is not None
    )

    if not needs_bursts:
        print("  -> no burst-dependent setup requested, skip")
        return

    from jaguar.preprocessing.burst_discovery import discover_bursts

    use_fiftyone = current_setup_config["data"].get("use_fiftyone", False)
    burst_min_cluster_size = curation_cfg.get("burst_min_cluster_size", 2)
    burst_max_within = curation_cfg.get("burst_max_within", 500)
    burst_max_cross = curation_cfg.get("burst_max_cross", 10000)
    seed = split_cfg.get("seed", 51)
    phash_size = curation_cfg.get("phash_size", 8)

    bursts_root = get_burst_paths()["write_root"]

    for cfg_path in bursts_root.rglob("config.json"):
        cfg = read_json_if_exists(cfg_path)
        if cfg is not None and cfg.get("round") == ROUND:
            print(f"  -> burst config for {ROUND} already exists")
            print(f"  -> using: {cfg_path.parent}")
            return cfg_path.parent

    print(f"  -> no burst config for {ROUND} found, running burst discovery")
    discover_bursts(
        use_fiftyone=use_fiftyone,
        burst_min_cluster_size=burst_min_cluster_size,
        burst_max_within=burst_max_within,
        burst_max_cross=burst_max_cross,
        seed=seed,
        phash_size=phash_size,
    )

    for cfg_path in bursts_root.rglob("config.json"):
        cfg = read_json_if_exists(cfg_path)
        if cfg is not None and cfg.get("round") == ROUND:
            return cfg_path.parent

    raise RuntimeError("Burst discovery ran, but no config.json with matching round was found.")

def ensure_split_artifacts(current_setup_config: dict) -> Path | None:
    print("[SETUP] ensure_split_artifacts")

    split_cfg = current_setup_config.get("split", {})
    curation_cfg = current_setup_config.get("curation", {})
    use_fiftyone = current_setup_config["data"].get("use_fiftyone", False)

    required_keys = [
        "strategy",
        "include_duplicates",
        "val_split_size",
        "seed",
    ]
    if not all(k in split_cfg for k in required_keys):
        print("  -> split config incomplete, skip")
        return None

    from jaguar.preprocessing.split_and_curate import create_splits_and_curate

    split_strategy = split_cfg["strategy"]
    include_duplicates = split_cfg["include_duplicates"]
    val_split_size = split_cfg["val_split_size"]
    seed = split_cfg["seed"]
    train_k = curation_cfg.get("train_k")
    val_k = curation_cfg.get("val_k")
    phash_threshold = curation_cfg.get("phash_threshold")

    if train_k is None or val_k is None or phash_threshold is None:
        print("  -> curation config incomplete, skip")
        return None

    paths = get_split_paths(
        split_strategy=split_strategy,
        include_duplicates=include_duplicates,
        train_k=train_k,
        val_k=val_k,
        phash_threshold=phash_threshold,
    )

    full_path = paths["full_split_file"]
    write_root = paths["write_root"]

    print("[SETUP] expected split path:", full_path)
    print("[SETUP] exists:", full_path.exists())

    if full_path.exists():
        print("[SETUP] split artifacts already exist -> skip")
        print(f"[SETUP] using: {full_path}")
        return full_path

    print("[SETUP] no matching split artifacts found -> running split_and_curate")
    create_splits_and_curate(
        use_fiftyone=use_fiftyone,
        split_strategy=split_strategy,
        include_duplicates=include_duplicates,
        train_k_per_dedup=train_k,
        val_k_per_dedup=val_k,
        phash_thresh_dedup=phash_threshold,
        val_split_size=val_split_size,
        seed=seed,
    )
    if not full_path.exists():
        raise RuntimeError(f"Split creation finished, but file not found: {full_path}")

    return full_path

def run_step(step_name: str, current_setup_config: dict):
    if step_name == "ensure_output_dirs":
        ensure_output_dirs()
    elif step_name == "ensure_fiftyone_init_dataset":
        ensure_fiftyone_init_dataset(current_setup_config)
    elif step_name == "ensure_background_pool":
        ensure_background_pool()
    elif step_name == "ensure_burst_artifacts":
        ensure_burst_artifacts(current_setup_config)
    elif step_name == "ensure_split_artifacts":
        ensure_split_artifacts(current_setup_config)
    else:
        raise ValueError(f"Unknown setup step: {step_name}")


def run_setup(base_config_key: str, experiment_config_key: str) -> dict:
    print(f"[SETUP] steps = {SETUP_STEPS}")

    base_config = load_toml_config(base_config_key)
    experiment_config = load_toml_config(experiment_config_key)
    current_setup_config = deep_update(base_config, experiment_config)

    for step in SETUP_STEPS:
        run_step(step, current_setup_config)

    print("[SETUP] done")
    return current_setup_config


def main():
    args = parse_args()
    run_setup(
        base_config_key=args.base_config,
        experiment_config_key=args.experiment_config,
    )


if __name__ == "__main__":
    main()