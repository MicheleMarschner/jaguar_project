#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name scientific_base --base_config base/xai_base --experiment_config _generated/xai_metrics/eva02_faith_complex_
python src/jaguar/experiments/run_xai_metrics.py --base_config base/xai_base --experiment_config _generated/xai_metrics/eva02_faith_complex_ --experiment_name eva02_faith_complex_

