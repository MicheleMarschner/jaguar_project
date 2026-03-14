#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name eda_base --base_config base/xai_base --experiment_config _generated/eda_background_intervention/eva02_origin_gray
python src/jaguar/experiments/run_background_intervention.py --base_config base/xai_base --experiment_config _generated/eda_background_intervention/eva02_origin_gray --experiment_name eva02_origin_gray

