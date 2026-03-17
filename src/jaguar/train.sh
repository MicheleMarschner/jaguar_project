#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=32
##SBATCH --nodes=1
#SBATCH --time=0-6:00:00
#SBATCH --mem=200GB
#SBATCH --account=kainmueller
#SBATCH --array=0-9
#SBATCH --nodelist=maxg[10,20]
#SBATCH --output=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.out
#SBATCH --error=/fast/AG_Kainmueller/vguarin/jaguar_project/logs/log_%j.err
#SBATCH --export=ALL
#SBATCH --partition=h100
#SBATCH -pkainmueller

echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"

BACKBONES=(
"EVA-02"
"MegaDescriptor-L"
"MiewID"
"DINO-Small"
"DINOv2-Base"
"DINOv3-Large"
"DINOv2_for_wildlife"
"ConvNeXt-V2"
"EfficientNet-B4"
"Swin-Transformer" 
)

BACKBONE=${BACKBONES[$SLURM_ARRAY_TASK_ID]}

echo "Running backbone: $BACKBONE"

echo "Using shared cache"

export TORCH_HOME=/fast/AG_Kainmueller/data/jaguar_project/.cache/torch
export HF_HOME=/fast/AG_Kainmueller/data/jaguar_project/.cache/huggingface
export XDG_CACHE_HOME=/fast/AG_Kainmueller/data/jaguar_project/.cache

# Run trainings
python src/jaguar/main.py \
    --config leaderboard_experiment_eva \
    --backbone "$BACKBONE"