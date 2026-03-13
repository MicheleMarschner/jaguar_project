#!/usr/bin/env bash
set -e

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/single_eva02 --experiment_name single_eva02

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/single_convnext --experiment_name single_convnext

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/single_dinov2_base --experiment_name single_dinov2_base

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/single_megadescriptor_l --experiment_name single_megadescriptor_l

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/single_miew_id --experiment_name single_miew_id

