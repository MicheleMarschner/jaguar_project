#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/train_single_baseline/backbone_eva02
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/train_single_baseline/backbone_eva02 --experiment_name backbone_eva02

