#!/bin/bash
#SBATCH --job-name=oct_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%1
#SBATCH --output=oct_train_%A_%a.log
set -euo pipefail
: "${OCT_REPO:?Set the absolute repository directory}"
: "${OCT_CONFIG:?Set the absolute config JSON path}"
: "${OCT_RUNS:?Set a new run root}"
: "${OCT_ARCHIVE:?Set a checkpoint mirror directory}"
OCT_PYTHON="${OCT_PYTHON:-python}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=42
cd "$OCT_REPO"
variants=(basic dynamic basic_noise na_dynamic)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"
"$OCT_PYTHON" -u src/train.py --config "$OCT_CONFIG" --variant "$variant" --seed 42 --out "$OCT_RUNS/${variant}_seed42" --archive "$OCT_ARCHIVE" --resume
