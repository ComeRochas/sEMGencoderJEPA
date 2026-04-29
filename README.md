# semg-jepa

A lightweight project based on `dgaddy/silent_speech` for:

- baseline raw-EMG CTC training,
- JEPA self-supervised pretraining,
- CTC fine-tuning from JEPA checkpoints,
- CTC beam-search WER/CER evaluation.

## Data config

All live-loading scripts accept `--data-config path/to/config.json` with keys:

- `testset_file`
- `silent_data_directories`
- `voiced_data_directories`
- `remove_channels` (optional)

## Cache precomputation

```bash
python scripts/precompute_raw_emg.py \
  --data-config data_config.json \
  --cache-dir cache/raw_emg \
  --num-workers 16
```

Then validate:

```bash
python scripts/validate_raw_emg_cache.py \
  --data-config data_config.json \
  --cache-dir cache/raw_emg \
  --split train \
  --num-examples 20
```

## Train/eval from cache

- Baseline: `python train_baseline_ctc.py --use-cache --cache-dir cache/raw_emg`
- JEPA pretrain: `python train_jepa_pretrain.py --use-cache --cache-dir cache/raw_emg`
- JEPA finetune: `python train_jepa_finetune_ctc.py --use-cache --cache-dir cache/raw_emg --pretrained-encoder <pt>`
- Eval: `python evaluate_ctc.py --use-cache --cache-dir cache/raw_emg --split test --checkpoint <pt>`

## Optional W&B logging

Use `--wandb` to enable logging (offline mode by default).

Example:

```bash
python train_baseline_ctc.py \
  --use-cache \
  --cache-dir cache/raw_emg \
  --wandb \
  --wandb-entity UMLforVideoLab \
  --wandb-project JEPAforsEMG \
  --wandb-run-name baseline_raw_only
```
