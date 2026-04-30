# sEMGencoderJEPA — Project Context

## What this project is

EMG-to-text silent speech framework. Three pipelines:
1. **Baseline CTC** — supervised, raw EMG → characters
2. **JEPA pretraining** — self-supervised student/teacher encoder on raw EMG (BYOL-style)
3. **JEPA fine-tuning** — CTC head on top of pretrained encoder

Based on Gaddy & Klein's `silent_speech` codebase. Core hypothesis: JEPA pretraining learns better EMG representations, especially in low-label regimes.

## Cluster & environment

- **Login node:** `torch-login-a-2` (NYU HPC)
- **Python env:** `/scratch/cr4206/envs/silent_speech/bin/python` (torch 2.0.1)
- **Package not pip-installed** — always set `PYTHONPATH=/scratch/cr4206/sEMGencoderJEPA`
- **Slurm account:** `torch_pr_39_tandon_advanced`
- **GPU partitions:** `h100_tandon` (15 nodes), `h200_public` (29 nodes), `l40s_public` — check idle count with `sinfo` before submitting
- **CPU partition:** `cpu_short` — used for precomputation (no GPU)
- **Do not over-poll squeue/sacct** — account can get rate-limited

## Data

- **Raw EMG:** `/scratch/cr4206/data/emg_data/emg_data/{silent_parallel_data,voiced_parallel_data,nonparallel_data}`
- **Testset JSON:** `/scratch/cr4206/silent_speech/testset_largedev.json`
- **Config:** `configs/data_config.json` — all absolute paths filled in
- **Precomputed cache (ready):** `data/{train,dev,test}.pt`
  - train: 8055 samples, 652 MB | dev: 200 samples | test: 99 samples
  - Each sample: `raw_emg` float16 [8*T, 8], `text` str, `text_int` long, `ctc_length` int T

Both **voiced and silent** utterances are in the cache and used for training. The CTC loss treats them identically — the `silent` flag is present in batches but not used for any special loss term. Voiced EMG provides cleaner signal and more training data; silent EMG is the target domain.

## Signal processing (`load_utterance` in `semg_jepa/read_emg.py`)

1. Load `{idx}_emg.npy` + adjacent files for context (1000 Hz, 8 channels)
2. Notch filter 60 Hz harmonics (up to 420 Hz), high-pass at 2 Hz
3. Crop back to utterance window
4. Subsample to 689.06 Hz (`emg_orig`)
5. Zero out `remove_channels` if specified
6. T = (len(emg_orig) − 8) // 8 → ~86.13 frames/s
7. Crop: `raw = emg_orig[8 : 8+8*T]`
8. Normalize: `raw = 50 * tanh(raw / 20 / 50)`
9. Returns dict: `{raw_emg [8T,8], text, book_location, ctc_length T}`

**No MFCC, no EMG features, no phonemes** — removed, never used in any training script.

## Ground truth labels

`text` from `{idx}_info.json` → `TextTransform.clean_text()`:
  - `unidecode()` → ASCII
  - `jiwer.RemovePunctuation()` → strip punctuation
  - lowercase
  - map to char indices in `"abcdefghijklmnopqrstuvwxyz0123456789 "` (37 chars)

CTC blank token = index 37. Vocab size = 38 total.

## Model architecture

`GaddyRawEMGEncoder`: `[B, 8T, 8]` → `[B, T, D]`
- 3× ResBlock Conv1d (stride=2 each = 8× temporal downsampling)
- Linear projection
- N-layer Transformer, relative positional embeddings (window=100)
- Default: model_size=768, num_layers=6, nhead=8, dim_feedforward=3072
- During training: random temporal shift 0–7 samples (data augmentation, clones input to avoid in-place mutation)

`BaselineCTCModel` = `GaddyRawEMGEncoder` + `CTCHead` (Linear → vocab+1)

`JEPAModel` (in `train_jepa.py`) = encoder + predictor MLP (student) + target_proj (teacher, EMA-updated)

## Training pipeline

All scripts read from `data/*.pt` via `CachedRawEMGDataset`. No live preprocessing.

```
precompute_raw_emg.slurm  →  data/{train,dev,test}.pt
                                      ↓
train_baseline_cached.slurm     →  runs/baseline/{best,last}.pt
train_jepa_pretrain_cached.slurm →  runs/jepa_pretrain/pretrained_encoder.pt
train_jepa_finetune_cached.slurm →  runs/jepa_finetune/{best,last}.pt
evaluate_ctc.py                 →  prints WER + CER
```

`build_batches(dataset, max_len)` in `cached_dataset.py` builds size-aware batches (total raw_emg samples ≤ max_len). Called fresh each epoch for reshuffling.

`combine_fixed_length(raw_emg_list, 1600)` reshapes variable-length list into `[B_chunks, 1600, 8]` before the encoder during training. NOT used during evaluation (model handles variable lengths natively).

## Evaluation

`evaluate(model, dataset, device, batch_size=16)` in `ctc_utils.py`:
- Batched evaluation (batch_size=16, sequences padded to max length in batch)
- `seq_lens` passed to CTCBeamDecoder so it ignores padded frames
- CTC beam search with language model (`lm.binary`, alpha=1.5, beta=1.85)
- Returns `(wer, cer)` — both logged during training

## W&B logging

- `--wandb` flag on all training scripts (off by default)
- Defaults: `entity="UMLforVideoLab"`, `project="JEPAforsEMG"`
- Mode: offline by default (`WANDB_MODE=offline`); set `WANDB_MODE=online` to stream live
- Sync offline runs: `wandb sync runs/baseline/wandb/`
- W&B dashboard: https://wandb.ai/UMLforVideoLab/JEPAforsEMG

## Script names

| Python script | Slurm script | Purpose |
|--------------|-------------|---------|
| `train_baseline.py` | `train_baseline_cached.slurm` | Supervised CTC |
| `train_jepa.py` | `train_jepa_pretrain_cached.slurm` | JEPA pretraining |
| `finetune_from_jepa.py` | `train_jepa_finetune_cached.slurm` | CTC finetune |
| `evaluate_ctc.py` | — | WER+CER eval |
| `scripts/precompute_raw_emg.py` | `precompute_raw_emg.slurm` | Cache builder |

## Output directories (all under /scratch/cr4206/sEMGencoderJEPA/)

| Path | Contents |
|------|---------|
| `data/` | Precomputed cache (.pt files) — **already done** |
| `logs/` | Slurm stdout/stderr |
| `runs/baseline/` | Baseline CTC checkpoints |
| `runs/jepa_pretrain/` | JEPA pretrain checkpoints + `pretrained_encoder.pt` |
| `runs/jepa_finetune/` | JEPA finetune checkpoints |

## How to run

```bash
# Precompute (already done — skip unless data changes)
sbatch slurm/precompute_raw_emg.slurm

# Train baseline
sbatch slurm/train_baseline_cached.slurm

# JEPA pretrain then finetune
sbatch slurm/train_jepa_pretrain_cached.slurm
# wait for pretrained_encoder.pt, then:
sbatch slurm/train_jepa_finetune_cached.slurm

# Evaluate
PYTHONPATH=/scratch/cr4206/sEMGencoderJEPA \
  /scratch/cr4206/envs/silent_speech/bin/python evaluate_ctc.py \
  --checkpoint runs/baseline/best.pt

# Override output dir at submission
OUTPUT_DIR=runs/baseline_v2 sbatch slurm/train_baseline_cached.slurm
```

## Key design decisions

- **Cache-only**: avoids scipy filter+resample on every epoch
- **No MFCC/EMG features/phonemes**: were computed by parent codebase but unused here — removed
- **build_batches per epoch**: new batch groupings each epoch without reloading data
- **Batched evaluation** (batch_size=16 with padding): ~16× faster than batch_size=1
- **In-place mutation fix**: `GaddyRawEMGEncoder` now clones input before temporal shift to avoid corrupting the shared batch tensor

## Experiment roadmap (from TODO.md)

1. ✅ Precompute cache (done: job 7650966, 0 errors)
2. Sanity-check: reproduce baseline CTC WER ~30%
3. Baseline experiments at data fractions: 100%, 50%, 25%, 10%, 5%, 1%
4. JEPA pretraining (full data)
5. JEPA fine-tuning at same fractions → compare WER to baseline
6. Joint CTC+JEPA (future)
7. Robustness tests under augmentation
8. Cross-session generalization
