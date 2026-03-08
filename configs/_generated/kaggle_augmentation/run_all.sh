#!/usr/bin/env bash
set -e

python src/jaguar/setup_experiment.py --setup_name kaggle_backbone

python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_augmentation/aug_none --experiment_name aug_none
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_augmentation/aug_full --experiment_name aug_full
