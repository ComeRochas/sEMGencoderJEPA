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
