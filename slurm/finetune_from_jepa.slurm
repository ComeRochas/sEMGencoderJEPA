#!/bin/bash
#SBATCH --job-name=finetune_from_jepa
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cr4206/sEMGencoderJEPA/logs/finetune_from_jepa_%j.out
#SBATCH --error=/scratch/cr4206/sEMGencoderJEPA/logs/finetune_from_jepa_%j.err

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMGencoderJEPA/data}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/cr4206/sEMGencoderJEPA/runs/jepa_finetune}
PRETRAINED_ENCODER=${PRETRAINED_ENCODER:-/scratch/cr4206/sEMGencoderJEPA/runs/jepa_pretrain/pretrained_encoder.pt}

cd /scratch/cr4206/sEMGencoderJEPA
export PYTHONPATH=/scratch/cr4206/sEMGencoderJEPA${PYTHONPATH:+:$PYTHONPATH}

mkdir -p /scratch/cr4206/sEMGencoderJEPA/logs "$OUTPUT_DIR"

echo "[finetune_from_jepa] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[finetune_from_jepa] cache_dir=$CACHE_DIR output_dir=$OUTPUT_DIR pretrained_encoder=$PRETRAINED_ENCODER"

"$PYTHON" -u scripts/finetune_from_jepa.py \
  --cache-dir "$CACHE_DIR" \
  --pretrained-encoder "$PRETRAINED_ENCODER" \
  --output-directory "$OUTPUT_DIR" \
  --wandb

echo "[finetune_from_jepa] finished at=$(date -Is)"
