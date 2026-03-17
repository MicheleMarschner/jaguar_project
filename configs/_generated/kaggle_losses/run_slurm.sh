#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=32
#SBATCH --time=0-6:00:00
#SBATCH --mem=200GB
#SBATCH --account=kainmueller
#SBATCH --array=0-5
#SBATCH --nodelist=maxg[09,10,20]
#SBATCH --output=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.out
#SBATCH --error=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.err
#SBATCH -pkainmueller


export WANDB_DIR=/fast/AG_Kainmueller/data/jaguar_project/wandb/$SLURM_JOB_ID
export WANDB_CACHE_DIR=/fast/AG_Kainmueller/data/jaguar_project/wandb_cache
export WANDB_ARTIFACT_DIR=/fast/AG_Kainmueller/data/jaguar_project/wandb_artifacts
export WANDB_DATA_DIR=/fast/AG_Kainmueller/data/jaguar_project/wandb_data
CONFIGS=(
"_generated/kaggle_losses/loss_arcface"
"_generated/kaggle_losses/loss_cosface"
"_generated/kaggle_losses/loss_sphereface"
"_generated/kaggle_losses/loss_softmax"
"_generated/kaggle_losses/loss_triplet_bot_hard"
"_generated/kaggle_losses/loss_triplet_focal_bot_hard"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "Running config: $CONFIG"

python /fast/AG_Kainmueller/vguarin/jaguar_project/src/jaguar/main.py \
  --base_config base/kaggle_base \
  --experiment_config "$CONFIG"