#!/usr/bin/env bash
set -e

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/ens3_equal_global --experiment_name ens3_equal_global

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/ens3_model1_stronger --experiment_name ens3_model1_stronger

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/ens3_model2_stronger --experiment_name ens3_model2_stronger

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/ens3_model3_stronger --experiment_name ens3_model3_stronger

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/ens3_equal_rowminmax --experiment_name ens3_equal_rowminmax

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/ens3_equal_rowzscore --experiment_name ens3_equal_rowzscore

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/ens3_equal_no_square --experiment_name ens3_equal_no_square

python src/jaguar/experiments/run_ensemble.py --base_config base/ensemble_base --experiment_config _generated/kaggle_ensemble/ens3_equal_no_norm --experiment_name ens3_equal_no_norm

