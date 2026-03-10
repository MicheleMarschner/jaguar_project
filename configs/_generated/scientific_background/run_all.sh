#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name scientific_background --base_config base/scientific_base --experiment_config _generated/scientific_background/bg_train_original
python src/jaguar/main.py --base_config base/scientific_base --experiment_config _generated/scientific_background/bg_train_original --experiment_name bg_train_original

python src/jaguar/experiments/experiment_setup.py --setup_name scientific_background --base_config base/scientific_base --experiment_config _generated/scientific_background/bg_train_random_bg
python src/jaguar/main.py --base_config base/scientific_base --experiment_config _generated/scientific_background/bg_train_random_bg --experiment_name bg_train_random_bg

python src/jaguar/experiments/experiment_setup.py --setup_name scientific_background --base_config base/scientific_base --experiment_config _generated/scientific_background/bg_train_mixed_original_random_bg
python src/jaguar/main.py --base_config base/scientific_base --experiment_config _generated/scientific_background/bg_train_mixed_original_random_bg --experiment_name bg_train_mixed_original_random_bg

python src/jaguar/experiments/experiment_setup.py --setup_name scientific_background --base_config base/scientific_base --experiment_config _generated/scientific_background/bg_train_blur_bg
python src/jaguar/main.py --base_config base/scientific_base --experiment_config _generated/scientific_background/bg_train_blur_bg --experiment_name bg_train_blur_bg

python src/jaguar/experiments/experiment_setup.py --setup_name scientific_background --base_config base/scientific_base --experiment_config _generated/scientific_background/bg_train_gray_bg
python src/jaguar/main.py --base_config base/scientific_base --experiment_config _generated/scientific_background/bg_train_gray_bg --experiment_name bg_train_gray_bg

