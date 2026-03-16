#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/xai_base --experiment_config _generated/eda_xai_class_attribution/eva02_val_n332_all_groups
python src/jaguar/experiments/run_class_attribution_generation.py --base_config base/xai_base --experiment_config _generated/eda_xai_class_attribution/eva02_val_n332_all_groups --experiment_name eva02_val_n332_all_groups

