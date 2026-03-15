#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/kaggle_base --experiment_config _generated/baseline/baseline_init_round2
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/baseline/baseline_init_round2 --experiment_name baseline_init_round2

