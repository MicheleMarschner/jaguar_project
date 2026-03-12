#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name xai_similarity --base_config base/xai_base --experiment_config _generated/xai_similarity/eva02_easy_hard
python src/jaguar/experiments/run_xai_similarity.py --base_config base/xai_base --experiment_config _generated/xai_similarity/eva02_easy_hard --experiment_name eva02_easy_hard

