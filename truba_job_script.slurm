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
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --account=kkaraca
#SBATCH --array=0-4

# ==============================================================================
#                      *** SLURM SETTINGS ***
# ==============================================================================

## module load apps/truba-ai/gpu-2024.0
## python3 -m venv ~/atif-ulakbim
## source ~/atif-ulakbim/bin/activate
## pip install --upgrade pip
## pip install --upgrade transformers
## pip install comet-ml
## deactivate

echo "İş Array ID (Model İndeksi): $SLURM_ARRAY_TASK_ID"
echo "-------------------------------------------------"

module purge
module load apps/truba-ai/gpu-2024.0

echo "Sanal ortam etkinleştiriliyor..."
source ~/atif-ulakbim/bin/activate
echo "-------------------------------------------------"

echo "Ortam yüklendi, Python betiği başlatılıyor..."
echo "-------------------------------------------------"

python -u experiment_3_train_03.py --model_index "$SLURM_ARRAY_TASK_ID"