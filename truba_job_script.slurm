#!/bin/bash

# ==============================================================================
#                      *** SLURM SETTINGS ***
# ==============================================================================
#SBATCH --job-name=optuna-train
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --partition=barbun-cuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --account=kkaraca
#SBATCH --array=0-3

# ==============================================================================
#                      *** SLURM SETTINGS ***
# ==============================================================================

echo "İş Array ID (Model İndeksi): $SLURM_ARRAY_TASK_ID"
echo "-------------------------------------------------"

module purge
module load apps/truba-ai/gpu-2024.0

echo "Ortam yüklendi, Python betiği başlatılıyor..."
echo "-------------------------------------------------"

python -u train_v3.py --model_index "$SLURM_ARRAY_TASK_ID"