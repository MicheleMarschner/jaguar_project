#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/baseline/baseline_init
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/baseline/baseline_init --experiment_name baseline_init

