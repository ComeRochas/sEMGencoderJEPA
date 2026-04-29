#!/bin/bash
#SBATCH --job-name=precompute_raw_emg
#SBATCH --output=logs/precompute_raw_emg_%j.out
#SBATCH --error=logs/precompute_raw_emg_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

DATA_CONFIG=${DATA_CONFIG:-path/to/data_config.json}
CACHE_DIR=${CACHE_DIR:-path/to/cache_dir}
NUM_WORKERS=${NUM_WORKERS:-16}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python scripts/precompute_raw_emg.py \
  --data-config "$DATA_CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --num-workers "$NUM_WORKERS"
