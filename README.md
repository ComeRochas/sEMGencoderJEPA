# semg-jepa

EMG-to-text silent speech framework based on [Gaddy & Klein (2021)](https://github.com/dgaddy/silent_speech).

Four training pipelines:
1. **Baseline CTC** — supervised, raw EMG → characters
2. **JEPA pretraining** — self-supervised student/teacher encoder (BYOL+VICReg hybrid) *(on hold — see Status)*
3. **JEPA fine-tuning** — CTC head on pretrained encoder *(on hold)*
4. **UML training** — dual-branch CTC: EMG + audio sharing one Transformer
   ([Better Together, Gosztolai et al. 2025](https://arxiv.org/pdf/2510.08492)). Two audio sources supported:
   LibriSpeech `train-clean-100` (external, ~100 h) and **Gaddy-internal audio**
   (the voiced recordings shipped with the EMG dataset, ~15 h — same speakers/prompts as EMG).

## Status

**UML is the active direction; JEPA is parked.** Full-data results (test split, KenLM beam decoding):

| Initialization / training | Test WER | Test CER | Δ WER vs baseline |
|---|---|---|---|
| Supervised CTC baseline | 0.328 | 0.141 | reference |
| JEPA pretrain → CTC finetune | ~0.33 (projected; dev 0.403) | – | ≈ 0 |
| UML EMG-branch directly | 0.291 | 0.133 | **−3.7 pp** |
| UML separate-heads → finetune | **0.287** | **0.132** | **−4.0 pp** |
| UML shared-head → finetune | 0.292 | 0.138 | −3.6 pp |

JEPA pretraining itself converges cleanly (variance/covariance regularizers stable, no representation collapse), but the resulting encoder gives no measurable transfer benefit at full labels — supervised CTC already extracts the same information from the same 16.9 h of EMG. UML wins because it brings genuinely external information (LibriSpeech audio + frozen `wav2vec2-base`) into the shared transformer.

The remaining open question for JEPA is whether it helps at low label fractions (1-10%), where SSL traditionally pays off. That is the one experiment kept on the JEPA side of the roadmap; otherwise the focus is on pushing UML further. See [TODO.md](TODO.md).

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

# Fine-tune CTC from JEPA encoder (200 epochs, after pretraining)
sbatch slurm/finetune_from_jepa.slurm

# UML training, LibriSpeech audio (needs LibriSpeech cache first)
sbatch slurm/precompute_audio.slurm   # one-shot: data/libri_cache/train-clean-100.pt
sbatch slurm/train_uml.slurm          # dual-branch CTC, 200 epochs

# UML training, Gaddy-internal audio (precompute the voiced_clean recordings)
sbatch slurm/precompute_audio_gaddy.slurm   # data/gaddy_audio_cache/gaddy_internal.pt
sbatch slurm/train_uml_gaddy_audio.slurm

# Fine-tune on EMG-only from a UML EMG-branch checkpoint
EMG_BRANCH=runs/uml/best_emg_branch.pt sbatch slurm/finetune_from_uml.slurm
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
| `scripts/finetune_from_jepa.py` | CTC fine-tune from JEPA encoder (encoder only; head reinitialized) |
| `scripts/finetune_from_uml.py` | CTC fine-tune from a UML EMG-branch checkpoint (encoder **and** EMG CTC head loaded) |
| `scripts/train_uml.py` | UML dual-branch CTC (EMG + audio, shared Transformer) — same script for both audio sources |
| `scripts/evaluate.py` | WER + CER evaluation (greedy/beam, optional dev grid search) |
| `scripts/precompute_raw_emg.py` | Precompute raw EMG cache |
| `scripts/precompute_audio.py` | Precompute LibriSpeech cache (UML) |
| `scripts/precompute_audio_gaddy.py` | Precompute Gaddy-internal audio cache (UML) |

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
- LibriSpeech (UML, external audio): `/scratch/cr4206/data/librispeech/LibriSpeech/<split>/...`
- LibriSpeech cache: `data/libri_cache/train-clean-100.pt` (~11 GB, 28 539 utterances, ~100 h)
- Gaddy-internal audio (UML, in-distribution): voiced recordings from `voiced_parallel_data/` + `nonparallel_data/`
- Gaddy audio cache: `data/gaddy_audio_cache/gaddy_internal.pt` (~1.7 GB, 7 052 utterances, ~15 h)

## UML pipeline

Implements unpaired multimodal learning ([Gosztolai et al. 2025](https://arxiv.org/pdf/2510.08492)).
A single Transformer is shared between an EMG branch and an audio branch; the
two branches are trained with separate CTC losses on their own labelled data.
The audio is treated as unpaired even when the audio recordings happen to come
from the same dataset as the EMG (Gaddy-internal mode).

```
loss = loss_emg(EMG → CNN → SharedTransformer → emg_ctc_head)
     + lambda_uml · loss_audio(audio → wav2vec2 (frozen) → linear → SharedTransformer → audio_ctc_head)
```

**Audio source (configurable):**
- **LibriSpeech** (`scripts/precompute_audio.py` + `slurm/train_uml.slurm`) —
  external read-speech, ~100 h, well-conditioned audio gradient direction.
- **Gaddy-internal** (`scripts/precompute_audio_gaddy.py` + `slurm/train_uml_gaddy_audio.slurm`) —
  the voiced recordings shipped with the EMG dataset, ~15 h. Same speakers
  and prompts as the EMG. Pairing is **not** exploited (the audio loader
  yields batches independently of the EMG loader). With the smaller audio
  volume and long-tail utterance durations (up to ~54 s), gradient clipping
  must be disabled (`clip_grad_norm: 0.0`) to avoid clipping the EMG
  component away — see the comment in `configs/train_uml_gaddy_audio.yaml`.

**Scheduling modes:**
- `epoch_mode: alternate` (default) — per optimizer step, 1 EMG batch + 1
  audio batch (both backward, then `optim.step()`). Balanced co-training.
- `epoch_mode: both` — per step, 1 batch from a single modality, sampling
  uniformly from the union of all EMG and all audio batches. Audio dominates
  by ~20× with LibriSpeech, ~5× with Gaddy-internal; `lambda_uml` must be
  reduced to rebalance.

By default `share_ctc_head=false` (each branch has its own readout); flip it
to `true` to make both branches share a single CTC linear layer. Inference
uses the EMG branch only. Per-epoch checkpoints saved by `train_uml.py`:
- `last.pt` / `best.pt` — full UMLModel state (resume + audio branch).
- `last_emg_branch.pt` / `best_emg_branch.pt` — encoder + EMG CTC head only,
  remapped to `encoder.*` / `ctc_head.*` so they load directly into
  `BaselineCTCModel` via `scripts/finetune_from_uml.py`.
- `pretrained_encoder.pt` — encoder-only weights, drop-in for
  `--pretrained-encoder` in `scripts/finetune_from_jepa.py` (the JEPA-style
  finetune that does not re-use the EMG CTC head).

`uml/`:
- `audio_dataset.py` — char-level cache reader (format is identical for
  LibriSpeech and Gaddy-internal caches)
- `model.py` — `AudioFrontend` (frozen wav2vec2-base + linear) and `UMLModel`
  (the shared transformer is `model.emg_encoder.transformer` — literally the
  same `nn.Module` evaluated on both paths)
