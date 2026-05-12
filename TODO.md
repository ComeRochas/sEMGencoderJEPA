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

#### C.a — Does `alternate` give a signal? Sweep audio-source × λ
Six runs in flight (3 audio caches × 3 λ values), all with
`epoch_mode: alternate`, `share_ctc_head: false`, `clip_grad_norm: 0.0`.
Each followed by an EMG-only finetune (`finetune_from_uml.py`,
`eval_method: greedy`) for a single comparable test number.

| Audio | λ_uml | UML jobid | Finetune jobid | UML output | Finetune output |
|---|---|---|---|---|---|
| Gaddy-internal | 1.0 | 8508274 (RUN) | 8522327 | `runs/uml_gaddy_audio_lambda1.0/` | `runs/finetune_uml_gaddy_lambda1.0/` |
| Gaddy-internal | 0.5 | 8508275 (RUN) | 8522328 | `runs/uml_gaddy_audio_lambda0.5/` | `runs/finetune_uml_gaddy_lambda0.5/` |
| Gaddy-internal | 0.3 | 8508276 (PEND QOS) | 8522329 | `runs/uml_gaddy_audio_lambda0.3/` | `runs/finetune_uml_gaddy_lambda0.3/` |
| LibriSpeech    | 1.0 | 8522323 (PEND, h200) | 8522330 | `runs/uml_libri_lambda1.0/` | `runs/finetune_uml_libri_lambda1.0/` |
| LibriSpeech    | 0.5 | 8522324 (PEND, h200) | 8522331 | `runs/uml_libri_lambda0.5/` | `runs/finetune_uml_libri_lambda0.5/` |
| LibriSpeech    | 0.3 | 8522325 (PEND, h200) | 8522332 | `runs/uml_libri_lambda0.3/` | `runs/finetune_uml_libri_lambda0.3/` |

Hypotheses being tested simultaneously:
1. **Audio-source effect**: does in-distribution Gaddy audio shape the shared
   transformer better than out-of-distribution LibriSpeech? (3 vs 3 paired
   comparison at matched λ).
2. **λ effect**: does down-weighting the audio loss (0.3, 0.5 vs 1.0) help or
   hurt the EMG path? Looking for a non-flat curve.
3. **UML beats baseline**: does any of the 6 finetunes get below baseline
   (test WER 0.328)?

Decisions to make once results land:
- Pick the best (audio-source, λ) combination for downstream sweeps.
- If Gaddy-internal ≈ LibriSpeech at full labels, the gain comes from
  "any external audio + wav2vec2 prior" rather than from in-distribution
  match. If Gaddy clearly wins, the speaker/prompt-match story holds.

Lessons already baked in (do not repeat):
- `clip_grad_norm: 1.0` is incompatible with the noisier Gaddy audio path —
  it pinned the EMG branch in the CTC all-blank attractor (run 8450813,
  cancelled). Disabled for all 6 runs.
- `share_ctc_head: true` underperforms (0.292 vs 0.287 at full labels).
  Kept off.

#### C.b — Make `both` work: rebalance audio, document the failure mode
Current state: `epoch_mode=both` UML run + EMG-only finetune produced flat
loss for 200 epochs (job 8426709). Confirmed mechanism: with LibriSpeech +
`audio_batch_size=8`, audio batches outnumber EMG batches by ~20× per
epoch (3500 vs 170), so the shared transformer drifts to a ~50 Hz wav2vec2
regime and the EMG branch can't recover during finetune (lr already at lr/8
by the time it would catch up).

Tasks, in order:
- [ ] **Raise `audio_batch_size` to rebalance step counts.** With `bs=64`,
      `n_audio_batches ≈ 446`, ratio drops from ~20× to ~2.6×. With `bs=128`,
      ratio ~1.3×. Cleanest single-knob fix (no math on losses, just batch
      sizing). Try `bs=64` first.
- [ ] **In parallel, sweep `lambda_uml ∈ {0.05, 0.1, 0.3}` in `both`** to
      compensate for the residual step-count imbalance with default
      `audio_batch_size=8`. The two knobs reach similar balance from
      different sides — keep whichever gives best finetune WER.
- [ ] **If neither single fix recovers**, try the two-phase schedule:
      phase 1 = `both` (transformer learns linguistic structure from audio
      breadth), phase 2 = `alternate` with `lambda_uml=1.0` (co-train +
      rebalance), phase 3 = EMG-only finetune.
- [ ] **Document the failure mode and the fix** in `RESULTS.md` either way.

Open question: does `both` with proper rebalancing actually beat `alternate`,
or does it just match it? The advisor's stated goal is "exploit more audio
without killing EMG". `alternate` already does that (each audio batch sees
the EMG path's gradient at the same step); `both` only helps if seeing
*more distinct* audio batches per epoch lets the transformer converge to a
better linguistic representation. If C.a shows that audio-source effect is
strong (Gaddy wins) but step count is weak (`λ=1.0`, `λ=0.3` similar),
then `both` is unlikely to add anything in expectation.

#### C.c — UML in the **unsupervised setting** of the Better Together paper
Replace the dual-CTC objective with paired SSL objectives on both modalities,
sharing the same transformer. wav2vec2 stays frozen (the audio-side SSL is
upstream of our wav2vec2 features). The EMG-side encoder + transformer is the
part we pretrain.

Design sketch (to be refined):
- **Common SSL objective**: masked-latent prediction with an EMA teacher,
  data2vec/BYOL style. For each modality, mask spans of the transformer
  input, predict the EMA-teacher representation of the masked positions.
- Audio branch: `wav2vec2(audio) → proj → mask → shared_transformer →
  predictor → loss against EMA(shared_transformer(unmasked))`.
- EMG branch: `conv_blocks(raw_emg) → mask → shared_transformer →
  predictor → loss against EMA(shared_transformer(unmasked))`.
- Single optimizer step = EMG SSL grad + audio SSL grad (`alternate` adapted
  to SSL).
- After SSL convergence, finetune the EMG encoder + transformer with CTC on
  labeled EMG (reuse `finetune_from_uml.py` once we restore an EMG head, or
  `finetune_from_jepa.py`).

Open design questions before coding:
- One predictor or two (same argument as `separate_heads` winning in C.a).
- Mask spans by *seconds*, not by frames (EMG 86 Hz vs audio post-wav2vec2
  50 Hz are not the same granularity).
- One EMA teacher transformer (shared) or two (one per modality)? Shared
  keeps the unified-representation premise; separate is easier to converge
  but blurs the point.

Compare against: the standalone JEPA pretrain (~0 transfer at full data) and
the CTC-UML winner from C.a. SSL-UML wins iff the shared transformer gets
*more* information from the audio pool than the CTC-UML audio supervision
already extracts — uncertain a priori.

### Other UML follow-ups (lower priority, after C.a–C.c)
- [x] LibriSpeech `train-clean-100` cache built (~11 GB, 28539 utterances)
- [x] Gaddy-internal audio cache built (~1.7 GB, 7052 utterances, ~15 h)
- [x] UML separate-heads training (200 epochs) → test WER 0.287 / CER 0.132 after finetune (LibriSpeech, λ=1.0, May 7)
- [x] UML shared-head training (200 epochs) → test WER 0.292 / CER 0.138 after finetune (LibriSpeech, May 7)
- [x] Direct evaluation of UML EMG-branch without finetune — test WER 0.291 / CER 0.133
- [ ] **Data-fraction sweep on UML-finetune.** Train at label fractions {50%, 25%, 10%, 5%, 1%} starting from the best `runs/uml*/pretrained_encoder.pt` from C.a. Compare to baseline at the same fractions to characterize how UML's WER advantage scales with label scarcity. *(Requires `--data-fraction` flag — `CachedRawEMGDataset.subset()` already exists.)*
- [ ] **Push UML further at full data.** Possible levers: longer training (300 ep), larger LibriSpeech split (`train-clean-360` or `train-other-500`), unfreeze last 1-2 wav2vec2 layers near end of training.
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
