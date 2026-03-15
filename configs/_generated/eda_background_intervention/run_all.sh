#!/usr/bin/env bash
set -e

python src/jaguar/experiments/run_background_intervention.py --base_config base/kaggle_base --experiment_config _generated/eda_background_intervention/eva02 --experiment_name eva02

