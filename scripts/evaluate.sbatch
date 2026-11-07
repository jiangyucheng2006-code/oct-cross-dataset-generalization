#!/bin/bash
#SBATCH --job-name=oct_evaluate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=05:00:00
#SBATCH --output=oct_evaluate_%j.log
set -euo pipefail
: "${OCT_REPO:?Set repository directory}"
: "${OCT_CONFIG:?Set matching config path}"
: "${OCT_RUNS:?Set completed run root}"
: "${OCT_ARCHIVE:?Set archive directory}"
OCT_PYTHON="${OCT_PYTHON:-python}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=42
cd "$OCT_REPO"
"$OCT_PYTHON" -u src/evaluate.py --root "$OCT_RUNS" --config "$OCT_CONFIG" --experiments configs/experiments.json --archive "$OCT_ARCHIVE"
"$OCT_PYTHON" -u src/aggregate.py --root "$OCT_RUNS"
