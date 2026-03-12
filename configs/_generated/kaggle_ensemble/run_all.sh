#!/usr/bin/env bash
set -e

python src/jaguar/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/eva04_miewid04_conv02 --experiment_name eva04_miewid04_conv02

