# sEMG-JEPA Experiment Roadmap

## Setup / infrastructure
- [x] Precompute raw EMG cache (job 7650966, 0 errors — 8055/200/99 samples)
- [x] Fix in-place tensor mutation bug in `GaddyRawEMGEncoder`
- [x] Batched evaluation with CER metric in `ctc_utils.evaluate()`
- [x] W&B offline logging wired into all training scripts
- [x] Beam search + KenLM evaluation with parallel CPU decoding
- [x] Grid search over (beam_width, alpha, beta) on dev in `scripts/evaluate.py --grid-search`
- [x] Leakage-free unigrams auto-built from LibriSpeech vocab via `semg_jepa/unigrams.py` (no train-text leakage)
- [x] Fix spurious pyctcdecode "no unigrams / space token missing" warnings (was a probe-call artifact)
- [x] Match Gaddy's effective batch budget (`max_batch_len=88000`, grad-accum=2) for fair comparison

## A. Sanity checks / reproduction
- [x] Train Gaddy reference codebase (job 8112504) — converges to ~32% val WER, confirms our pipeline can match
- [x] Train CTC baseline matching Gaddy — 200 epochs → dev WER 0.396 (greedy), test WER 0.328 / CER 0.141 with KenLM beam (job 7989402 / eval 8156489)
- [x] Confirm WER and CER are logged to stdout and W&B
- [x] Confirm checkpoints save `best.pt` by WER and `last.pt` every epoch

## B. Baseline experiments
- [x] Full-data CTC baseline trained → `runs/baseline/best_20260506_1550.pt`
- [ ] Implement `--data-fraction` flag in `scripts/train_baseline.py` and `cached_dataset.py`
- [ ] Train CTC baseline at label fractions: 50%, 25%, 10%, 5%, 1%
- [ ] Record WER/CER for each fraction in `RESULTS.md`

## C. JEPA pretraining experiments
- [x] Initial JEPA run (job 8087607) — diagnosed representation collapse: loss plateau at exactly 1.0, var-hinge structurally broken by L2-normalization (per-dim std capped at 1/√D)
- [x] Rework JEPA loss into VICReg-style hybrid: smooth-L1 invariance on LayerNorm-stabilized features + scale-correct variance hinge + off-diagonal covariance regularizer
- [x] Cosine EMA-momentum schedule (0.996 → 0.9999)
- [x] LR warmup (1000 steps) + cosine decay to 0
- [x] Per-component loss + per-feature `std_mean` / `std_min` diagnostics logged to stdout and W&B (`diag/feat_std_min` is the canary against collapse)
- [ ] **In progress**: re-run JEPA pretraining (job 8161494). Early signs healthy (inv dropping, std_min ~0.87, no plateau). Re-check at epoch 30 and 50.
- [ ] If `std_min` < 0.2 at any point → bump `--var-weight` to 50
- [ ] If `cov` climbs past ~3 → bump `--cov-weight` to 2–5
- [ ] Confirm `pretrained_encoder.pt` is saved at end of run

## D. JEPA fine-tuning experiments
- [x] Initial finetune-from-JEPA run (job 8112501) — dev WER 0.482 (greedy) at epoch 80, no improvement over baseline. Expected, given the pretrained encoder was collapsed.
- [x] Add LR warmup + MultiStepLR decay (milestones=[50,60,70], gamma=0.5) to `finetune_from_jepa.py` to match baseline's schedule
- [ ] Re-run finetune from the new (non-collapsed) pretrained encoder
- [ ] Compare full-data WER/CER vs baseline (target: at minimum match, ideally 1–3 pp better)
- [ ] Fine-tune at label fractions 50%, 25%, 10%, 5%, 1% (where SSL gains are expected to be largest)
- [ ] Compare per-fraction WER/CER vs baseline → produce data-efficiency curve

## E. JEPA architecture / objective improvements (if results warrant)
- [ ] Switch to masked-prediction JEPA (I-JEPA / data2vec-2 style): mask 30–50% of time tokens, predict teacher features at masked positions from unmasked context. Likely the largest single lever beyond the current fixes.
- [ ] Replace 2-layer MLP predictor with a 2–4 layer transformer predictor (only meaningful with masked-prediction)
- [ ] Tune `inv_weight`, `var_weight`, `cov_weight` jointly if collapse pressure shifts
- [ ] Architecture ablations: `model_size=256/512`, `num_layers=4`
- [ ] Augmentation ablations: channel-dropout-only / time-mask-only / no temporal shift
- [ ] Augmentation severity sweep

## F. Joint CTC + JEPA (future)
- [ ] Implement joint supervised CTC + JEPA auxiliary loss
- [ ] Compare three-way: CTC-only / JEPA pretrain+finetune / joint CTC+JEPA

## G. Robustness experiments
- [ ] Add test-time corruption options to `scripts/evaluate.py` (channel dropout, noise, time mask)
- [ ] Evaluate baseline and JEPA models under one/two dropped channels
- [ ] Plot WER/CER degradation curves vs corruption severity

## H. Cross-session generalization
- [ ] Add session hold-out split in `cached_dataset.py`
- [ ] Train on subset of sessions, test on held-out → compare baseline vs JEPA

## I. Reporting
- [ ] `RESULTS.md`: results table — model × label fraction × (WER, CER, eval method)
- [ ] `RESULTS.md`: robustness table — model × corruption × (WER, CER)
- [ ] Save W&B run links and best-checkpoint paths alongside each row
