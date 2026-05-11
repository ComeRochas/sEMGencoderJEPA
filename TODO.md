# sEMG silent-speech roadmap

**Active direction: UML. JEPA is on hold** — see results below and section D.

## Results so far (full data, test split, KenLM beam)

| Initialization / training | Test WER | Test CER | Source |
|---|---|---|---|
| Supervised CTC baseline | 0.328 | 0.141 | eval 8156489, ckpt `runs/baseline/best_20260506_1550.pt` |
| JEPA pretrain → CTC finetune | ~0.33 (projected; dev 0.403) | – | finetune 8174254 (not yet evaluated on test) |
| UML EMG-branch directly | 0.291 | 0.133 | eval 8176026, ckpt `runs/uml/best_emg_branch.pt` |
| UML separate-heads → finetune | **0.287** | **0.132** | finetune 8175992 + eval 8176028 |
| UML shared-head → finetune | 0.292 | 0.138 | finetune 8176001 + eval 8176029 |

UML wins by ~4 pp WER at full labels. JEPA gives no transfer at full labels.

## A. Setup / infrastructure (done)
- [x] Precompute raw EMG cache (8055/200/99 samples)
- [x] Fix in-place tensor mutation bug in `GaddyRawEMGEncoder`
- [x] Batched evaluation with CER metric in `ctc_utils.evaluate()`
- [x] Beam-search + KenLM evaluation with parallel CPU decoding
- [x] Grid search over (beam_width, alpha, beta) on dev in `scripts/evaluate.py --grid-search`
- [x] Leakage-free unigrams auto-built from LibriSpeech vocab via `semg_jepa/unigrams.py`
- [x] Match Gaddy's effective batch budget (`max_batch_len=88000`, grad-accum=2)
- [x] W&B offline logging on all training scripts

## B. Baseline (done)
- [x] Reproduce Gaddy reference codebase (job 8112504, ~32% val WER)
- [x] Train CTC baseline (200 epochs) — test WER 0.328, CER 0.141
- [x] Confirm WER and CER are logged to stdout and W&B; checkpoints save best/last per epoch

## C. UML — primary focus

### Priority experiments (advisor-ordered)

#### C.1 — UML `alternate` with **Gaddy-internal audio** as the audio modality
Replace the LibriSpeech audio cache with the audio recordings already sitting next
to every EMG file in Gaddy's dataset (`{idx}_audio_clean.flac` in
`voiced_parallel_data/` and `nonparallel_data/`, and the parallel-voiced clip
for `silent_parallel_data/`). Pairing with EMG is **not** exploited — the audio
loader yields audio batches independently of the EMG batches, exactly as the
LibriSpeech loader does. The hypothesis to test is that audio drawn from the
*same latent distribution* as the EMG (same speakers, same prompts, same
recording style) shapes the shared transformer more usefully than
out-of-distribution LibriSpeech audio.

Implementation plan:
- New cache builder `scripts/precompute_audio_gaddy.py`: walk
  `voiced_parallel_data/` + `nonparallel_data/`, for each `{idx}_info.json` load
  `{idx}_audio_clean.flac`, resample to 16 kHz, zero-mean/unit-var, fp16. For
  `silent_parallel_data/`, look up the parallel voiced clip via
  `voiced_data_locations` (same logic Gaddy uses for silent→voiced alignment).
  Encode the transcript via the same shared `TextTransform`. Write to
  `data/gaddy_audio_cache/train.pt`.
- Add `--audio-cache` flag to `scripts/train_uml.py` so we can swap caches
  without code changes.
- Train UML `alternate`, `separate_heads`, 200 epochs, same lr schedule as
  current best (`runs/uml/best.pt`).
- Evaluate UML EMG-branch directly + after EMG-only finetune.

Expected reading: if WER ≤ 0.287, in-distribution audio helps and the
LibriSpeech run was suboptimal; if WER significantly worse, then *external
linguistic breadth* (LibriSpeech) matters more than *acoustic-distribution
match*. Either result is a clean scientific signal.

A real risk to plan around: Gaddy's audio is ~17 h (≈ EMG duration), vs
LibriSpeech's 100 h. With far fewer audio batches per cycle in `alternate`, the
audio branch may overfit. Mitigation: lower audio batch size and/or stronger
audio dropout; keep `lambda_uml=1.0` to start, then sweep.

#### C.2 — UML in the **unsupervised setting of the Better Together paper**
Replace the dual-CTC objective with paired SSL objectives on both modalities,
sharing the same transformer. wav2vec2 stays frozen (the audio-side SSL is
upstream of our wav2vec2 features — we only need an SSL task *on top of* them).
EMG-side encoder + transformer is the part we pretrain.

Design sketch (to be refined):
- **Common SSL objective family**: masked-latent prediction with an EMA
  teacher, BYOL/data2vec style. For each modality, mask spans of the
  transformer input, predict the EMA-teacher representation of the masked
  positions. Same loss formula on both branches; only the input pipeline
  differs.
- Audio branch: `wav2vec2(audio) → proj → mask → shared_transformer →
  predictor → loss against EMA(shared_transformer(unmasked))`.
- EMG branch: `conv_blocks(raw_emg) → mask → shared_transformer → predictor →
  loss against EMA(shared_transformer(unmasked))`.
- Single optimizer step = EMG SSL grad + audio SSL grad (i.e., `alternate`
  scheduling adapted to SSL).
- After SSL convergence, throw away the predictor and audio branch; finetune
  the EMG encoder + transformer with CTC on labeled EMG.

Open design questions before coding:
- Use the **same predictor** for both modalities (forces shared structure
  end-to-end) or **separate predictors** (probably easier to converge — same
  argument as `separate_heads` winning in C).
- Mask ratio and span length per modality: EMG at 86 Hz vs audio post-wav2vec2
  at 50 Hz are not the same time granularity; pick mask spans by *seconds*
  not by frames.
- EMA teacher: one teacher transformer (shared) or two? Sharing keeps the
  unified-representation premise; two teachers (one per modality) is more
  flexible but blurs the point.

Compare against: the standalone JEPA pretrain (which gave ~0 transfer at full
data) **and** the CTC-UML (C.1 result). The expected gain of SSL-UML over
JEPA-only is the same mechanism as why CTC-UML beats baseline: the shared
transformer gets information from outside the EMG pool. Whether SSL gives more
or less than CTC-supervised audio is the empirical question.

#### C.3 — UML `both`: diagnose finetune failure and rebalance
Current state: `epoch_mode=both` UML run + EMG-only finetune (job 8426709)
produced no learning after 200 epochs of finetune. Need to:

1. **Diagnose.** Likely causes, in order of probability:
   - **Transformer drift toward audio.** With `both`, audio steps outnumber
     EMG steps by ~20× per epoch (n_audio_batches / n_emg_batches ≈ 3500/170).
     The shared transformer ends UML training tuned to ~50 Hz wav2vec2-feature
     statistics, not to EMG features. The EMG path's projection
     (`w_raw_in`) and the CTC head also saw ~20× fewer gradients than the
     audio side, so they're under-fit.
   - **LR schedule already collapsed.** Finetune uses
     `MultiStepLR([125,150,175], 0.5)` — by the time the encoder catches up
     (probably needs many epochs to recover from audio drift), the lr is
     already at lr/8. Plot `dev_wer` vs epoch from job 8426709 to confirm.
   - **`best_emg_branch.pt` checkpoint pathology.** If during UML-both
     training the EMG dev WER never improved past initialization, then
     `best.pt == init` or close to it. Inspect the saved logs.
   - **BN running stats**: should be EMG-only (BN sits in `conv_blocks`,
     which only runs on EMG path), so this is unlikely — but verify in the
     state_dict to be sure.

2. **Mitigations to try** (cheapest first):
   - **Rebalance batches in `both`**: per epoch, subsample audio to roughly
     `n_emg_batches` batches (random pick + reshuffle next epoch). Result is
     an `alternate`-like balance but with the "all batches seen exactly
     once" guarantee deliberately broken. Cleanest single-knob fix.
   - **Lower `lambda_uml`** in `both` (e.g., 0.1 or 0.05) so each audio step
     contributes proportionally less gradient. Compensates for the audio
     step *count* without touching the schedule.
   - **Two-phase schedule**: phase 1 `both` (transformer learns linguistic
     structure from audio breadth), phase 2 `alternate` with `lambda_uml=1.0`
     (rebalance + co-train), phase 3 EMG-only finetune.
   - **Boost finetune lr / re-warm-up**: longer warmup, higher peak lr in
     finetune, no MultiStepLR decay until epoch 50+. Lets the EMG path
     recover from audio drift.
   - **Drop `both` if none of the above recovers it.** The advisor goal is
     "use lots of audio without killing EMG"; `alternate` with longer
     training + `train-clean-360` already achieves that, and is the natural
     fallback.

3. **Write up the failure mode** in `RESULTS.md` either way — a documented
   negative result on `both` is useful evidence about the mechanism by which
   UML helps EMG.

### Other UML follow-ups (lower priority, do after C.1–C.3)
- [x] LibriSpeech `train-clean-100` cache built (~11 GB, 28539 utterances)
- [x] UML separate-heads training (200 epochs) → test WER 0.287 / CER 0.132 after finetune
- [x] UML shared-head training (200 epochs) → test WER 0.292 / CER 0.138 after finetune
- [x] Direct evaluation of UML EMG-branch without finetune — test WER 0.291 / CER 0.133
- [ ] **Data-fraction sweep on UML-finetune.** Train at label fractions {50%, 25%, 10%, 5%, 1%} starting from `runs/uml/pretrained_encoder.pt`. Compare to baseline at the same fractions to characterize how UML's WER advantage scales with label scarcity. *(Requires `--data-fraction` flag — `CachedRawEMGDataset.subset()` already exists.)*
- [ ] **Push UML further at full data.** Possible levers: longer training (300 ep), higher `lambda_uml`, larger LibriSpeech split (`train-clean-360` or `train-other-500`), unfreeze last 1-2 wav2vec2 layers near end of training.
- [ ] Robustness: evaluate UML EMG-branch under one/two dropped channels, additive noise, time masking. Compare to baseline robustness.
- [ ] Cross-session generalization: hold out one session, train UML on the rest, evaluate on held-out. Compare to baseline.

## D. JEPA — on hold
JEPA pretraining itself works (no collapse, schedules stable, predictor learning). The issue is downstream: at full labels, supervised CTC already extracts the same information from the same 16.9 h of EMG, so JEPA-init gives ~0 transfer benefit. The verdict at full data is essentially settled.

The single experimental direction worth keeping alive:
- [ ] **Low-label data-fraction sweep with JEPA-finetune.** This is the only regime where SSL traditionally pays off without external data. Run JEPA-finetune at {10%, 5%, 1%} labels alongside the baseline+UML sweeps in section C. Three possible outcomes:
  1. JEPA shows ≥3 pp WER edge over baseline at 5-10% labels → JEPA is publishable as a low-label result; resume tuning it.
  2. JEPA matches baseline at every fraction → drop it entirely from the project.
  3. UML+JEPA combined experiment (JEPA-pretrain → UML-train) shows compounding gains → revisit a multi-stage pipeline.

Until that sweep runs, no further investment in JEPA hyperparameters, architecture, or objective changes.

Things explicitly *not* worth doing now (deferred unless C/D's open experiments produce a reason to):
- ~~Switch to masked-prediction JEPA (I-JEPA / data2vec-2 style)~~ — only worth doing if low-label JEPA shows promise
- ~~Bigger predictor / architecture ablations on JEPA side~~
- ~~Augmentation severity sweep~~

## E. Reporting
- [ ] `RESULTS.md`: full-data results table (this file's summary)
- [ ] `RESULTS.md`: data-fraction sweep curves for baseline / UML / JEPA once C and D's sweeps are run
- [ ] `RESULTS.md`: robustness table (channel dropout, noise, time mask)
- [ ] Save W&B run links and best-checkpoint paths alongside each row
