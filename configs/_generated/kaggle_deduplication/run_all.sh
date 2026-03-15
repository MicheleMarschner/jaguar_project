#!/usr/bin/env bash
set -e

python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_deduplication/closed_keep_all --experiment_name closed_keep_all

python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_deduplication/closed_curated_traink_1_valk_50_p4 --experiment_name closed_curated_traink_1_valk_50_p4

python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_deduplication/closed_curated_traink_3_valk_1_p4 --experiment_name closed_curated_traink_3_valk_1_p4

python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_deduplication/closed_curated_traink_3_valk_3_p4 --experiment_name closed_curated_traink_3_valk_3_p4

python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_deduplication/closed_curated_traink_1_valk_1_p4 --experiment_name closed_curated_traink_1_valk_1_p4

python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_deduplication/closed_curated_traink_5_valk_1_p4 --experiment_name closed_curated_traink_5_valk_1_p4

python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_deduplication/closed_curated_traink_8_valk_1_p4 --experiment_name closed_curated_traink_8_valk_1_p4

