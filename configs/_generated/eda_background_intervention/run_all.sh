#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/xai_base --experiment_config _generated/eda_background_intervention/eva02_triplet_bg_orig_gray_black_rdn_blur_mixed
python src/jaguar/experiments/run_background_intervention.py --base_config base/xai_base --experiment_config _generated/eda_background_intervention/eva02_triplet_bg_orig_gray_black_rdn_blur_mixed --experiment_name eva02_triplet_bg_orig_gray_black_rdn_blur_mixed

