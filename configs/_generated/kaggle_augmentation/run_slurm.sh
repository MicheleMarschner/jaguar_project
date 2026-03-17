#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=32
#SBATCH --time=0-6:00:00
#SBATCH --mem=200GB
#SBATCH --account=kainmueller
#SBATCH --array=0-7
#SBATCH --nodelist=maxg[10,20]
#SBATCH --output=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.out
#SBATCH --error=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.err
#SBATCH -pkainmueller


export WANDB_DIR=/fast/AG_Kainmueller/data/jaguar_project/wandb/$SLURM_JOB_ID
export WANDB_CACHE_DIR=/fast/AG_Kainmueller/data/jaguar_project/wandb_cache
export WANDB_ARTIFACT_DIR=/fast/AG_Kainmueller/data/jaguar_project/wandb_artifacts
export WANDB_DATA_DIR=/fast/AG_Kainmueller/data/jaguar_project/wandb_data
CONFIGS=(
"_generated/kaggle_augmentation/aug_none"
"_generated/kaggle_augmentation/aug_flip_only"
"_generated/kaggle_augmentation/aug_geom_noflip_nocol_noerase"
"_generated/kaggle_augmentation/aug_geom_nocol_noerase"
"_generated/kaggle_augmentation/aug_geom_noerase"
"_generated/kaggle_augmentation/aug_full_pipeline"
"_generated/kaggle_augmentation/aug_full_pipeline_erase025"
"_generated/kaggle_augmentation/aug_curr_baseline"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "Running config: $CONFIG"

python /fast/AG_Kainmueller/vguarin/jaguar_project/src/jaguar/main.py \
  --base_config base/kaggle_base \
  --experiment_config "$CONFIG"