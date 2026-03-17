#!/bin/bash
#SBATCH --job-name=jaguar-xai
#SBATCH --account=sci-demelo-computer-vision
#SBATCH --partition=gpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodelist=gx24
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/sc/home/michele.marschner/project/jaguar_project/logs/xai_%j.out
#SBATCH --error=/sc/home/michele.marschner/project/jaguar_project/logs/xai_%j.err

set -euxo pipefail

source /sc/home/michele.marschner/conda3/etc/profile.d/conda.sh
conda activate jaguar

cd /sc/home/michele.marschner/project/jaguar_project
mkdir -p logs

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi || true
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

PYTHONPATH=src python -m jaguar.experiments.experiment_runner \
  --base_config base/xai_base \
  --experiment_config experiments/eda_background_intervention