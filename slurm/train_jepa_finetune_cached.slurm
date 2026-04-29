#!/bin/bash
#SBATCH --job-name=jepa_finetune_cached
#SBATCH --output=logs/jepa_finetune_cached_%j.out
#SBATCH --error=logs/jepa_finetune_cached_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

CACHE_DIR=${CACHE_DIR:-path/to/cache_dir}
OUTPUT_DIR=${OUTPUT_DIR:-runs/jepa_finetune_cached}
PRETRAINED_ENCODER=${PRETRAINED_ENCODER:-path/to/pretrained_encoder.pt}

python train_jepa_finetune_ctc.py \
  --use-cache \
  --cache-dir "$CACHE_DIR" \
  --pretrained-encoder "$PRETRAINED_ENCODER" \
  --output-directory "$OUTPUT_DIR" \
  --wandb
