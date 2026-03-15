#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_eva02
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_eva02 --experiment_name backbone_eva02

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_megadescriptor_l
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_megadescriptor_l --experiment_name backbone_megadescriptor_l

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_miew_id
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_miew_id --experiment_name backbone_miew_id

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dinov2_base
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dinov2_base --experiment_name backbone_dinov2_base

