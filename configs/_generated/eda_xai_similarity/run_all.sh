#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/xai_base --experiment_config _generated/eda_xai_similarity/eva02_triplet_poseasy_hard_neghard
python src/jaguar/experiments/run_xai_similarity.py --base_config base/xai_base --experiment_config _generated/eda_xai_similarity/eva02_triplet_poseasy_hard_neghard --experiment_name eva02_triplet_poseasy_hard_neghard

