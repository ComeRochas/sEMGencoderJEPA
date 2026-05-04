# semg-jepa

EMG-to-text silent speech framework based on [Gaddy & Klein (2021)](https://github.com/dgaddy/silent_speech).

Three training pipelines:
1. **Baseline CTC** — supervised, raw EMG → characters
2. **JEPA pretraining** — self-supervised student/teacher encoder (BYOL-style)
3. **JEPA fine-tuning** — CTC head on pretrained encoder

## Setup

```bash
export PYTHONPATH=/scratch/cr4206/sEMGencoderJEPA
PYTHON=/scratch/cr4206/envs/silent_speech/bin/python
```

## Workflow

### 1. Precompute cache (already done)

```bash
sbatch slurm/precompute_raw_emg.slurm
# produces data/{train,dev,test}.pt  (~677 MB total, 8055/200/99 samples)
```

### 2. Train

```bash
# Baseline CTC (200 epochs)
sbatch slurm/train_baseline.slurm

# JEPA self-supervised pretraining (100 epochs)
sbatch slurm/train_jepa.slurm

# Fine-tune CTC from JEPA encoder (80 epochs, after pretraining)
sbatch slurm/finetune_from_jepa.slurm
```

Override any variable at submission time:
```bash
OUTPUT_DIR=/scratch/cr4206/sEMGencoderJEPA/runs/baseline_v2 sbatch slurm/train_baseline.slurm
```

### 3. Evaluate

Defaults: split=`test`, method=`beam`, grid-search on (tunes `beam_width × alpha × beta` on dev, then evaluates test with the best combo).

```bash
sbatch slurm/evaluate.slurm                                                    # default checkpoint
CHECKPOINTS="runs/baseline/best.pt runs/jepa_finetune/best.pt" \
  sbatch slurm/evaluate.slurm                                                  # multiple at once
GRID_SEARCH=0 SPLIT=dev METHOD=greedy sbatch slurm/evaluate.slurm              # quick check
```

Direct invocation:
```bash
$PYTHON scripts/evaluate.py --checkpoints runs/baseline/best.pt --split dev --method greedy
```

## Scripts

| File | Purpose |
|------|---------|
| `scripts/train_baseline.py` | Supervised CTC baseline |
| `scripts/train_jepa.py` | JEPA self-supervised pretraining |
| `scripts/finetune_from_jepa.py` | CTC fine-tune from JEPA encoder |
| `scripts/evaluate.py` | WER + CER evaluation (greedy/beam, optional dev grid search) |
| `scripts/precompute_raw_emg.py` | Precompute raw EMG cache |

## W&B

All training scripts support `--wandb` (offline by default; set `WANDB_MODE=online` for live sync).
Entity: `UMLforVideoLab` · Project: `JEPAforsEMG`

To sync offline runs:
```bash
wandb sync runs/baseline/wandb/
```
```bash
wandb sync runs/baseline/wandb/
```

## Data

- Raw EMG: `/scratch/cr4206/data/emg_data/emg_data/`
- Config: `configs/data_config.json`
- Cache: `data/{train,dev,test}.pt`
