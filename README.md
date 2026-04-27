# semg-jepa

A lightweight project based on `dgaddy/silent_speech` for:

- baseline raw-EMG CTC training,
- JEPA self-supervised pretraining,
- CTC fine-tuning from JEPA checkpoints,
- CTC beam-search WER/CER evaluation.

## Data config

All scripts accept `--data-config path/to/config.json` with keys:

- `testset_file`
- `silent_data_directories`
- `voiced_data_directories`
- `remove_channels` (optional)

## Scripts

- `train_baseline_ctc.py`
- `train_jepa_pretrain.py`
- `train_jepa_finetune_ctc.py`
- `evaluate_ctc.py`

## Optional W&B logging

Use `--wandb` to enable logging (offline mode by default).

Example:

```bash
python train_baseline_ctc.py \
  --data-config data_config.json \
  --wandb \
  --wandb-entity UMLforVideoLab \
  --wandb-project JEPAforsEMG \
  --wandb-run-name baseline_raw_only
```
