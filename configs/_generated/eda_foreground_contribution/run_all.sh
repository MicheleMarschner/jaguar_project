#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/xai_base --experiment_config _generated/eda_foreground_contribution/fg_contribution_eva02_base
python src/jaguar/experiments/run_foreground_contribution.py --base_config base/xai_base --experiment_config _generated/eda_foreground_contribution/fg_contribution_eva02_base --experiment_name fg_contribution_eva02_base

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/xai_base --experiment_config _generated/eda_foreground_contribution/fg_contribution_megadescriptor_base
python src/jaguar/experiments/run_foreground_contribution.py --base_config base/xai_base --experiment_config _generated/eda_foreground_contribution/fg_contribution_megadescriptor_base --experiment_name fg_contribution_megadescriptor_base

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/xai_base --experiment_config _generated/eda_foreground_contribution/fg_contribution_miewid_base
python src/jaguar/experiments/run_foreground_contribution.py --base_config base/xai_base --experiment_config _generated/eda_foreground_contribution/fg_contribution_miewid_base --experiment_name fg_contribution_miewid_base

