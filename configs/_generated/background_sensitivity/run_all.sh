#!/usr/bin/env bash
set -e

python src/jaguar/experiments/run_background_sensitivity.py --base_config base/xai_base --experiment_config _generated/background_sensitivity/xai_ --experiment_name xai_

