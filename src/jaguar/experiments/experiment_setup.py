# src/jaguar/setup_experiment.py

import argparse
import json
from pathlib import Path
from typing import Any

from jaguar.analysis.eda import run_eda
from jaguar.config import DATA_STORE, EXPERIMENTS_STORE, PATHS, ROUND
from jaguar.preprocessing.burst_discovery import discover_bursts
from jaguar.utils.utils_setup import build_habitat_backgrounds
from jaguar.utils.utils import ensure_dir, ensure_dirs, read_json_if_exists, resolve_path, write_json
from jaguar.utils.utils_setup import init_fiftyone_dataset


## burst_discovery

SETUP_STEPS = {
    "initial_eda": [
        "ensure_output_dirs",
        "ensure_fiftyone_init_dataset",
        "run_initial_eda",
    ],
    "scientific_background": [
        "ensure_output_dirs",
        #"ensure_split_manifest",
        "ensure_fiftyone_init_dataset",
        "ensure_background_pool",
    ],
    "kaggle_backbone": [
        "ensure_output_dirs",
       #"ensure_split_manifest",
        "ensure_fiftyone_init_dataset",
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
    return parser.parse_args()


def ensure_output_dirs():
    print("[SETUP] ensure_output_dirs")
    ensure_dirs(PATHS)
    print(f"  -> created all directories")


def ensure_fiftyone_init_dataset():
    print("[SETUP] ensure_fiftyone_init_dataset")

    FO_path = PATHS.data_export / "init" / "metadata.json"

    if FO_path.exists():
        print(f"  -> exists: {FO_path}")
        return
    
    FO_DATASET_NAME = "jaguar_init"
    manifest_dir = resolve_path("fiftyone/init", DATA_STORE)
    train_dir = PATHS.data_train
    csv_file = PATHS.data / "jaguar-re-id/train.csv"
    init_fiftyone_dataset(FO_DATASET_NAME, manifest_dir, csv_file, train_dir)

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


def run_initial_eda():
    print("[SETUP] run_initial_eda")

    ## check if folder exists
    artifacts_dir = resolve_path("bursts", EXPERIMENTS_STORE)   # upstream image-feature artifacts (e.g., sharpness)
    save_dir = PATHS.results / "eda"               # EDA outputs used for reporting + later decisions
    csv_file = PATHS.data / "jaguar-re-id/train.csv"
    test_csv_file = PATHS.data / "jaguar-re-id/test.csv"

    # if EDA output already exists, skip
    if save_dir.exists() and any(save_dir.iterdir()):
        print(f"  -> EDA output already exists, skip: {save_dir}")
        return

    # required base data must exist
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing train csv: {csv_file}")
    if not test_csv_file.exists():
        raise FileNotFoundError(f"Missing test csv: {test_csv_file}")

    # upstream artifacts must already have been prepared elsewhere
    if not artifacts_dir.exists():
        raise FileNotFoundError(
            f"Missing artifacts dir: {artifacts_dir}\n"
            "Run the burst discovery step first."
        )
    
    ensure_dir(save_dir)
    
    run_eda(
        train_file=csv_file, 
        test_file=test_csv_file,
        save_dir=save_dir,
        artifacts_dir=artifacts_dir
    )



def _burst_outputs_exist(out_dir: Path) -> bool:
    required = [
        out_dir / "meta_img_features.parquet",
        out_dir / "candidate_edges_all_pairs.parquet",
    ]
    return all(p.exists() for p in required)


def _build_burst_signature(
    *,
    dataset_name: str,
    burst_min_cluster_size: int,
    burst_max_within: int,
    burst_max_cross: int,
    seed: int,
    phash_size: int,
) -> dict[str, Any]:
    return {
        "round": ROUND,
        "dataset_name": dataset_name,
        "burst_min_cluster_size": burst_min_cluster_size,
        "burst_max_within": burst_max_within,
        "burst_max_cross": burst_max_cross,
        "seed": seed,
        "phash_size": phash_size,
    }


def ensure_burst_artifacts(
    *,
    dataset_name: str = "jaguar_init",
    burst_min_cluster_size: int = 2,
    burst_max_within: int = 500,
    burst_max_cross: int = 10000,
    seed: int = 51,
    phash_size: int = 8,
):
    bursts_root = PATHS.runs / "bursts"

    for cfg_path in bursts_root.rglob("config.json"):
        cfg = read_json_if_exists(cfg_path)
        if cfg is not None and cfg.get("round") == ROUND:
            print(f"[SETUP] burst config for {ROUND} already exists -> skip")
            print(f"[SETUP] using: {cfg_path.parent}")
            return cfg_path.parent

    print(f"[SETUP] no burst config for {ROUND} found -> running burst discovery")
    discover_bursts(
        BURST_MIN_CLUSTER_SIZE=burst_min_cluster_size,
        BURST_MAX_WITHIN=burst_max_within,
        BURST_MAX_CROSS=burst_max_cross,
        SEED=seed,
        phash_size=phash_size,
    )

    for cfg_path in bursts_root.rglob("config.json"):
        cfg = read_json_if_exists(cfg_path)
        if cfg is not None and cfg.get("round") == ROUND:
            return cfg_path.parent

    raise RuntimeError("Burst discovery ran, but no config.json with matching round was found.")


def run_step(step_name: str):
    print(f"[SETUP] {step_name}")

    if step_name == "ensure_output_dirs":
        print("  -> ensure output dirs")
    #elif step_name == "ensure_split_manifest":
    #    print("  -> ensure split manifest")
    elif step_name == "ensure_fiftyone_init_dataset":
        print("  -> ensure fiftyone init dataset")
    elif step_name == "ensure_background_pool":
        print("  -> ensure background pool")
    elif step_name == "run_initial_eda":
        run_initial_eda()
    else:
        raise ValueError(f"Unknown setup step: {step_name}")


def main():
    args = parse_args()

    if args.setup_name not in SETUP_STEPS:
        raise ValueError(f"Unknown setup_name: {args.setup_name}")

    steps = SETUP_STEPS[args.setup_name]

    print(f"[SETUP] setup_name = {args.setup_name}")
    print(f"[SETUP] steps = {steps}")

    for step in steps:
        run_step(step)

    print("[SETUP] done")


if __name__ == "__main__":
    main()