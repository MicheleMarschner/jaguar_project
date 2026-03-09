#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name scientific_deduplication --base_config base/scientific_base --experiment_config _generated/scientific_deduplication/closed_keep_all
python src/jaguar/main.py --base_config base/scientific_base --experiment_config _generated/scientific_deduplication/closed_keep_all --experiment_name closed_keep_all

python src/jaguar/experiments/experiment_setup.py --setup_name scientific_deduplication --base_config base/scientific_base --experiment_config _generated/scientific_deduplication/closed_curated_traink_1_valk_50_p4
python src/jaguar/main.py --base_config base/scientific_base --experiment_config _generated/scientific_deduplication/closed_curated_traink_1_valk_50_p4 --experiment_name closed_curated_traink_1_valk_50_p4

