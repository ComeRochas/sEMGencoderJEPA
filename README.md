# semg-jepa

A lightweight project based on `dgaddy/silent_speech` for:

- baseline raw-EMG CTC training,
- JEPA self-supervised pretraining,
- CTC fine-tuning from JEPA checkpoints,
- CTC beam-search WER evaluation.

## Data config

All scripts accept `--data-config path/to/config.json` with keys:

- `normalizers_file`
- `testset_file`
- `silent_data_directories`
- `voiced_data_directories`
- `text_align_directory`
- `remove_channels` (optional)

## Scripts

- `train_baseline_ctc.py`
- `train_jepa_pretrain.py`
- `train_jepa_finetune_ctc.py`
- `evaluate_ctc.py`
