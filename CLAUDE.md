# sEMGencoderJEPA — Project Context

## What this project is

EMG-to-text silent speech framework. Four pipelines:
1. **Baseline CTC** — supervised, raw EMG → characters
2. **JEPA pretraining** — self-supervised student/teacher encoder on raw EMG (BYOL-style) — **on hold, see status below**
3. **JEPA fine-tuning** — CTC head on top of pretrained encoder — on hold
4. **UML training** — dual-branch CTC: EMG + audio share a single Transformer
   ("Better Together: Unpaired Modality Learning", Gosztolai et al. 2025, arXiv:2510.08492).
   Two audio sources supported and currently swept in parallel:
   **LibriSpeech** (`train-clean-100`, ~100 h external) and **Gaddy-internal**
   audio (the voiced recordings shipped with the EMG dataset, ~15 h — same
   speakers and prompts as EMG, pairing not exploited).

Based on Gaddy & Klein's `silent_speech` codebase. Original hypothesis: JEPA / UML
auxiliary objectives learn better EMG representations, especially in low-label regimes.

## Status (current direction)

**UML is the active focus. JEPA is on hold.** Full-data results so far (test split, KenLM beam, no grid search):

| Init / training | Test WER | Test CER | vs baseline |
|---|---|---|---|
| Supervised CTC baseline (random init, 200 ep) | 0.328 | 0.141 | reference |
| JEPA pretrain → CTC finetune (200 ep) | ~0.33 (proj.) | ~0.14 | ≈ 0 (dev WER 0.403 vs baseline 0.396) |
| UML EMG-branch (no finetune) | 0.291 | 0.133 | **−3.7 pp** |
| UML separate-heads → finetune | **0.287** | **0.132** | **−4.0 pp** |
| UML shared-head → finetune | 0.292 | 0.138 | −3.6 pp |

Why JEPA gave no transfer at full labels: it pretrains on the same 16.9 h of EMG that supervised CTC already trains on, so there's no information advantage over the supervised baseline. UML wins because the shared transformer also sees ~100 h of LibriSpeech audio + a frozen `wav2vec2-base` prior, which is genuinely external information.

Where JEPA could still earn its keep: low-label fractions (1-10%), where SSL benefit traditionally appears even without external data. The data-fraction sweep is the only honest test; until then, JEPA is parked. See [TODO.md](TODO.md) section C.

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

`UMLModel` (in `uml/model.py`) = dual-branch model:
- EMG branch: `GaddyRawEMGEncoder` → `emg_ctc_head`
- Audio branch: `AudioFrontend` (frozen `facebook/wav2vec2-base` + trainable
  `nn.Linear(768, model_size)` projection) → SAME `transformer` instance →
  `audio_ctc_head`
- The Transformer is literally shared (`model.emg_encoder.transformer` is the
  exact `nn.Module` evaluated on both paths).
- `share_ctc_head` config flag (default `False`): if `True`, both branches use
  one common CTC head; if `False`, each branch has its own readout.
- AudioFrontend is always frozen on the wav2vec2 weights (only the
  768→model_size linear projection trains).
- Inference uses EMG branch only — `model(raw_emg)` delegates to `forward_emg`.

## Training pipeline

All scripts read from `data/*.pt` via `CachedRawEMGDataset`. No live preprocessing.

```
slurm/precompute_raw_emg.slurm  →  data/{train,dev,test}.pt   [already done]
                                           ↓
slurm/train_baseline.slurm      →  runs/baseline/{best,last}.pt
slurm/train_jepa.slurm          →  runs/jepa_pretrain/pretrained_encoder.pt
slurm/finetune_from_jepa.slurm  →  runs/jepa_finetune/{best,last}.pt
slurm/evaluate.slurm            →  prints WER + CER (optionally grid-searches LM hyperparams on dev)
```

`build_batches(dataset, max_len)` in `cached_dataset.py` builds size-aware batches (total raw_emg samples ≤ max_len). Called fresh each epoch for reshuffling.

`combine_fixed_length(raw_emg_list, 1600)` reshapes variable-length list into `[B_chunks, 1600, 8]` before the encoder during training. NOT used during evaluation (model handles variable lengths natively).

## UML pipeline specifics

Implements ["Better Together: Unpaired Modality Learning"](https://arxiv.org/pdf/2510.08492)
adapted to silent-speech CTC. The two modalities **share a single Transformer**;
otherwise each has its own frontend + CTC head and is trained on its own
(unpaired) labelled data.

### Datasets & preprocessing

- **EMG**: same `data/{train,dev,test}.pt` cache used by the baseline. Pipeline
  matches Gaddy: notch + highpass + 689 Hz subsample + tanh normalize, fp16 on
  disk. No special UML preprocessing.
- **Audio (two interchangeable sources)** — both precomputed once on CPU into
  the same payload schema (`audio: list[fp16 (T,)]`, `text_int: list[int64 (L,)]`,
  `version: 1`). The cache is read by `uml/audio_dataset.py`, which is
  format-agnostic.
  - **LibriSpeech** (`scripts/precompute_audio.py` → `data/libri_cache/train-clean-100.pt`,
    ~11 GB, 28 539 utterances, ~100 h, mean duration 6-12 s).
  - **Gaddy-internal** (`scripts/precompute_audio_gaddy.py` →
    `data/gaddy_audio_cache/gaddy_internal.pt`, ~1.7 GB, 7 052 utterances,
    ~15 h, mean 7.5 s, **long-tail up to 54 s**). Walks
    `voiced_parallel_data/` + `nonparallel_data/`, reads each
    `{idx}_audio_clean.flac` paired with the transcript in `{idx}_info.json`.
    Silent session dirs are skipped on purpose (their `audio_clean.flac` is a
    re-recording of the same parallel sentences already in
    `voiced_parallel_data/`).
  - Per-sample pipeline (both): FLAC → fp32 → resample 16 kHz (no-op,
    everything is already 16 kHz) → zero-mean/unit-variance (matches
    `Wav2Vec2FeatureExtractor`) → fp16 → transcript encoded via the **shared**
    `TextTransform` (same 37-char vocab as EMG, blank @ 37).

### Architecture (`uml/model.py`)

- `EMG → GaddyRawEMGEncoder → emg_ctc_head` (vanilla EMG path).
- `audio → AudioFrontend → SHARED transformer → audio_ctc_head` where
  `AudioFrontend` is `facebook/wav2vec2-base` (FROZEN, 95M params) followed by
  a trainable `nn.Linear(768, model_size)` projection.
- The Transformer is `model.emg_encoder.transformer` — the **exact same
  `nn.Module`** used on both paths.
- CTC heads: configurable. `share_ctc_head=False` (default) gives each branch
  its own readout; `share_ctc_head=True` makes both branches use a single
  shared `CTCHead`.
- Audio frame rate: wav2vec2-base downsamples 16 kHz → ~50 Hz. Mask-aware
  output lengths come from `_get_feat_extract_output_lengths`.

### Training loop (`scripts/train_uml.py`)

Two scheduling modes via `--epoch-mode`:

**`alternate` (default, recommended)** — per optimizer step, 1 EMG batch + 1
audio batch (both backward, then `optim.step()`):
1. **EMG sub-step** — same as `train_baseline.py`: `combine_fixed_length(raw_emg, 1600)`
   → encoder/transformer forward → `decollate_tensor` → per-sample CTC.
   `loss_emg.backward()` (no scaling).
2. **Audio sub-step** — pull next batch from a cycling audio loader
   (`audio_batch_size=8` default, reshuffled at each cycle). Forward through
   frozen wav2vec2 → projection → shared transformer → audio CTC head.
   `(lambda_uml * loss_audio).backward()`.
3. After both backwards: optional grad clip, `optim.step()`, `optim.zero_grad()`.
   Per-step gradient is the sum of EMG and audio gradients on the shared
   transformer.

**`both`** — per step, 1 batch from a single modality, sampled uniformly from
the union of all EMG and all audio batches (so every batch from both datasets
is processed exactly once per epoch). With LibriSpeech audio dominates by
~20× (n_audio_batches ≈ 3500 vs n_emg_batches ≈ 170); with Gaddy-internal it's
~5×. Without rebalancing `lambda_uml`, the EMG path drifts (transformer aligns
to audio statistics and the EMG CTC head can't recover during finetune —
documented failure mode, job 8426709).

LR schedule (both modes): linear warmup (1000 steps) then
`MultiStepLR([125,150,175], 0.5)`. Optimizer = AdamW, lr=3e-4, weight_decay=0.

**`clip_grad_norm`** — set to `0.0` (disabled) by default for new runs. The
original 1.0 worked with LibriSpeech but causes CTC all-blank collapse on the
EMG branch when paired with the noisier Gaddy-internal audio path (long-tail
durations create occasional gradient spikes; clipping at 1.0 drowns the EMG
signal under the audio direction). If you re-enable it, set ≥10.0.

### Data exposure

- The EMG epoch (~8055 samples ÷ ~size-aware batches) defines an "epoch".
- LibriSpeech is consumed via `_cycle(loader)`: it is reshuffled when the loop
  restarts, so over the full run every audio sample is seen multiple times.
- LibriSpeech (28k) >> EMG batches per epoch (~hundreds): the audio loader
  doesn't finish a full pass within a single EMG epoch — but across many
  epochs all audio data is seen.
- All data is seen "equally" only in the long-run sense — per step, exactly
  one EMG batch and one audio batch contribute (each weighted by its own
  loss term).

### Validation & checkpoints

- Validation: EMG-only, on `dev`, via `ctc_utils.evaluate(eval_wrapper, ...)`.
  `eval_wrapper` is a thin `nn.Module` calling `model.forward_emg(raw)`.
- Saved per epoch:
  - `runs/uml/last.pt` and `last_<ts>.pt` — full UMLModel state
  - `runs/uml/last_emg_branch.pt` — encoder + EMG CTC head only, keys
    remapped to `encoder.*` / `ctc_head.*` so it loads directly into
    `BaselineCTCModel` (and `FinetuneCTCModel`)
- Saved when `dev_wer` improves: `best.pt`, `best_<ts>.pt`, `best_emg_branch.pt`.
- At end of run: `pretrained_encoder.pt` = `model.emg_encoder.state_dict()`,
  drop-in for `--pretrained-encoder` in `finetune_from_jepa.py`.

### Fine-tuning back to EMG-only (post-UML)

Two paths, depending on whether the EMG CTC head should be warm-started:

- **`scripts/finetune_from_uml.py`** (preferred for UML-init finetunes) — loads
  encoder **and** EMG CTC head from `*_emg_branch.pt` (or extracts them from a
  full UMLModel `last.pt` / `best.pt`). All optim hyperparameters mirror
  `train_baseline` (lr=3e-4, warmup=1000, grad_accum=2, max_batch_len=88000,
  eval_method=beam).
- **`scripts/finetune_from_jepa.py`** — encoder-only init (head reset). Use
  this when the upstream pretrain had no CTC head (JEPA) or you deliberately
  want to throw the head away. The config now uses the same baseline-aligned
  knobs (88000 / grad_accum=2 / beam eval).

### Configuration

`configs/train_uml.yaml`: knobs include `lambda_uml` (audio loss weight,
default 1.0), `share_ctc_head` (default false), `audio_batch_size`,
`max_batch_len` (EMG), `model_size`, `num_layers`, `dropout`, LR schedule.

## Evaluation

`evaluate(model, dataset, device, method, ...)` in `ctc_utils.py`:
- Internally splits into `compute_log_probs` (one batched GPU forward) + decode step
- `method="greedy"`: GPU argmax + CTC collapse — fast (~2s on 200 samples)
- `method="beam"`: pyctcdecode + KenLM (`data/lm.binary`) + unigrams (`data/unigrams.txt`, auto-built from LibriSpeech vocab filtered through KenLM if missing — leakage-free generic English, see `semg_jepa/unigrams.py`); decoded in parallel on CPU pool — ~25–90s on 200 samples
- `grid_search(...)`: one forward pass, then iterates over `(beam_width, alpha, beta)` combos reusing the cached log_probs. Used by `scripts/evaluate.py --grid-search` to tune on dev before reporting test
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
| `scripts/train_baseline.py` | `slurm/train_baseline.slurm` | Supervised CTC |
| `scripts/train_jepa.py` | `slurm/train_jepa.slurm` | JEPA pretraining |
| `scripts/finetune_from_jepa.py` | `slurm/finetune_from_jepa.slurm` | CTC finetune (encoder only; head reinitialized) |
| `scripts/finetune_from_uml.py` | `slurm/finetune_from_uml.slurm` | CTC finetune (encoder + EMG CTC head loaded) |
| `scripts/train_uml.py` | `slurm/train_uml.slurm` | UML dual-branch CTC with LibriSpeech audio |
| `scripts/train_uml.py` | `slurm/train_uml_gaddy_audio.slurm` | Same script; reads `data/gaddy_audio_cache/gaddy_internal.pt` instead |
| `scripts/evaluate.py` | `slurm/evaluate.slurm` | WER+CER eval (greedy/beam, optional grid search) |
| `scripts/precompute_raw_emg.py` | `slurm/precompute_raw_emg.slurm` | EMG cache builder |
| `scripts/precompute_audio.py` | `slurm/precompute_audio.slurm` | LibriSpeech cache builder (UML) |
| `scripts/precompute_audio_gaddy.py` | `slurm/precompute_audio_gaddy.slurm` | Gaddy-internal audio cache builder (UML) |

## Output directories (all under /scratch/cr4206/sEMGencoderJEPA/)

| Path | Contents |
|------|---------|
| `data/` | Precomputed cache (.pt files) — **already done** |
| `logs/` | Slurm stdout/stderr |
| `runs/baseline/` | Baseline CTC checkpoints |
| `runs/jepa_pretrain/` | JEPA pretrain checkpoints + `pretrained_encoder.pt` |
| `runs/jepa_finetune/` | JEPA finetune checkpoints |
| `runs/uml/` | UML checkpoints + `pretrained_encoder.pt` (EMG-branch encoder) + `*_emg_branch.pt` (encoder + EMG CTC head) |
| `runs/uml_gaddy_audio_lambda*/`, `runs/uml_libri_lambda*/` | Per-config UML output dirs for the audio-source × λ sweep |
| `runs/finetune_uml_*/` | `finetune_from_uml.py` outputs (one per UML run) |
| `data/libri_cache/` | LibriSpeech cache — `<split>.pt`, ~11 GB for `train-clean-100` |
| `data/gaddy_audio_cache/` | Gaddy-internal audio cache — `gaddy_internal.pt`, ~1.7 GB |

## How to run

```bash
# Precompute (already done — skip unless data changes)
sbatch slurm/precompute_raw_emg.slurm

# Train baseline
sbatch slurm/train_baseline.slurm

# JEPA pretrain then finetune
sbatch slurm/train_jepa.slurm
# wait for pretrained_encoder.pt, then:
sbatch slurm/finetune_from_jepa.slurm

# UML training with LibriSpeech audio
sbatch slurm/precompute_audio.slurm        # data/libri_cache/train-clean-100.pt (~11 GB)
sbatch slurm/train_uml.slurm

# UML training with Gaddy-internal audio
sbatch slurm/precompute_audio_gaddy.slurm  # data/gaddy_audio_cache/gaddy_internal.pt (~1.7 GB)
sbatch slurm/train_uml_gaddy_audio.slurm

# Finetune EMG-only from a UML EMG-branch checkpoint (loads encoder + head):
EMG_BRANCH=runs/uml/best_emg_branch.pt sbatch slurm/finetune_from_uml.slurm

# Evaluate (single checkpoint, defaults: split=test, method=beam, grid_search=on)
sbatch slurm/evaluate.slurm
# Multiple checkpoints in one job
CHECKPOINTS="runs/baseline/best.pt runs/jepa_finetune/best.pt" sbatch slurm/evaluate.slurm
# Disable grid search
GRID_SEARCH=0 sbatch slurm/evaluate.slurm

# Override output dir at submission
OUTPUT_DIR=runs/baseline_v2 sbatch slurm/train_baseline.slurm
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
