#!/bin/bash
#SBATCH --job-name=jepa_pretrain_cached
#SBATCH --output=logs/jepa_pretrain_cached_%j.out
#SBATCH --error=logs/jepa_pretrain_cached_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

CACHE_DIR=${CACHE_DIR:-path/to/cache_dir}
OUTPUT_DIR=${OUTPUT_DIR:-runs/jepa_pretrain_cached}

python train_jepa_pretrain.py \
  --use-cache \
  --cache-dir "$CACHE_DIR" \
  --output-directory "$OUTPUT_DIR" \
  --wandb
