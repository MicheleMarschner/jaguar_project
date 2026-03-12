#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name scientific_base --base_config base/scientific_base --experiment_config _generated/background_reliance_eval/bgrel_test
python src/jaguar/experiments/run_background_reliance_eval.py --base_config base/scientific_base --experiment_config _generated/background_reliance_eval/bgrel_test --experiment_name bgrel_test

