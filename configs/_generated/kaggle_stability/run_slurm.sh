#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=32
#SBATCH --time=0-6:00:00
#SBATCH --mem=200GB
#SBATCH --account=kainmueller
#SBATCH --array=0-4
#SBATCH --nodelist=maxg[09,10,20]
#SBATCH --output=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.out
#SBATCH --error=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.err
#SBATCH -pkainmueller

CONFIGS=(
"_generated/kaggle_stability/stability_seed_42"
"_generated/kaggle_stability/stability_seed_123"
"_generated/kaggle_stability/stability_seed_256"
"_generated/kaggle_stability/stability_seed_512"
"_generated/kaggle_stability/stability_seed_1024"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "Running config: $CONFIG"

python src/jaguar/main.py \
  --base_config base/kaggle_base \
  --experiment_config "$CONFIG"