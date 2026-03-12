#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name scientific_base --base_config base/xai_base --experiment_config _generated/background_reliance_eval/eva02_origin_gray
python src/jaguar/experiments/run_background_reliance_eval.py --base_config base/xai_base --experiment_config _generated/background_reliance_eval/eva02_origin_gray --experiment_name eva02_origin_gray

