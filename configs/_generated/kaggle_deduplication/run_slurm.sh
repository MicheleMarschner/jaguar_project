#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=32
#SBATCH --time=0-6:00:00
#SBATCH --mem=200GB
#SBATCH --account=kainmueller
#SBATCH --array=0-4
#SBATCH --nodelist=maxg[09,10,20]
#SBATCH --partition=h100
#SBATCH --output=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.out
#SBATCH --error=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.err
#SBATCH -pkainmueller

CONFIGS=(
"_generated/kaggle_deduplication/closed_keep_all"
"_generated/kaggle_deduplication/closed_curated_traink_1_valk_50_p4"
"_generated/kaggle_deduplication/closed_curated_traink_3_valk_1_p4"
"_generated/kaggle_deduplication/closed_curated_traink_3_valk_3_p4"
"_generated/kaggle_deduplication/closed_curated_traink_1_valk_1_p4"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "Running config: $CONFIG"

python src/jaguar/main.py \
  --base_config base/scientific_base \
  --experiment_config "$CONFIG"