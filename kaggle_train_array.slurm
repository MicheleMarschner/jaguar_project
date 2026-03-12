#!/bin/bash -eux
#SBATCH --job-name=jaguar-array
#SBATCH --account=sci-demelo-computer-vision
#SBATCH --partition=gpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/sc/home/michele.marschner/project/jaguar_project/logs/array_%A_%a.out
#SBATCH --error=/sc/home/michele.marschner/project/jaguar_project/logs/array_%A_%a.err

source /sc/home/michele.marschner/conda3/etc/profile.d/conda.sh
conda activate jaguar

cd /sc/home/michele.marschner/project/jaguar_project

mkdir -p logs

CONFIG_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" generated_runs.txt)
REL_PATH="${CONFIG_PATH#configs/}"
REL_PATH="${REL_PATH%.toml}"
NAME="$(basename "${CONFIG_PATH%.toml}")"

echo "Running $NAME"
echo "Config path: $CONFIG_PATH"
echo "Relative config: $REL_PATH"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi || true
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

PYTHONPATH=src python src/jaguar/experiments/run_ensemble.py \
  --base_config base/ensemble_base \
  --experiment_config "$REL_PATH" \
  --experiment_name "$NAME"
