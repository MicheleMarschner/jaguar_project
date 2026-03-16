#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name base --base_config base/xai_base --experiment_config _generated/eda_xai_metrics/class_attribution_class_faith_sanity_complex
python src/jaguar/experiments/run_xai_metrics.py --base_config base/xai_base --experiment_config _generated/eda_xai_metrics/class_attribution_class_faith_sanity_complex --experiment_name class_attribution_class_faith_sanity_complex

