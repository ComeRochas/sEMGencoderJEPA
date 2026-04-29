# sEMG-JEPA Experiment Roadmap


## Precomputation / cache
- [ ] Run `python scripts/precompute_raw_emg.py --data-config <cfg.json> --cache-dir <cache_dir> --num-workers 16` on the full dataset.
- [ ] Validate cache vs live raw_only loader with `python scripts/validate_raw_emg_cache.py --data-config <cfg.json> --cache-dir <cache_dir> --split train --num-examples 20`.
- [ ] Run one cached baseline smoke test (`train_baseline_ctc.py --use-cache --cache-dir <cache_dir>`).
- [ ] Compare epoch time: live raw_only vs cached.
- [ ] Train full baseline from cache.
- [ ] Train JEPA pretraining from cache.
- [ ] Fine-tune JEPA checkpoint from cache.

## A. Sanity checks / reproduction
- [ ] Verify `raw_only=True` gives same `raw_emg` shape and `text_int` as full loader on a few examples (`python - <<'PY' ...` quick check using `EMGDataset(..., raw_only=True/False)`).
- [ ] Reproduce Gaddy-style CTC baseline with `raw_only=True` using `train_baseline_ctc.py`.
- [ ] Confirm WER and CER are logged locally and to W&B (`--wandb`).
- [ ] Confirm checkpoints save `best.pt` by WER and `last.pt` every epoch.
- [ ] Confirm `evaluate_ctc.py` reports both WER and CER.

## B. Baseline experiments
- [ ] Train full-data CTC baseline: `python train_baseline_ctc.py --use-cache --cache-dir <cache_dir> --output-directory runs/baseline_full`.
- [ ] Train CTC baseline with label fractions: 50%, 25%, 10%, 5%, 1% using `--data-fraction 0.5/0.25/0.1/0.05/0.01`.
- [ ] Record WER/CER for each fraction in `RESULTS.md`.
- [ ] Validate `--data-fraction` behavior (same seed, expected subset size).

## C. JEPA pretraining experiments
- [ ] Train JEPA pretraining with default augmentations: `python train_jepa_pretrain.py --use-cache --cache-dir <cache_dir> --output-directory runs/jepa_pretrain_default`.
- [ ] Log cosine loss, variance loss, and embedding std (`train/jepa_cosine_loss`, `train/variance_loss`, `train/embedding_std_mean`).
- [ ] Check for representation collapse (watch low embedding std and stalled cosine).
- [ ] Save and archive `pretrained_encoder.pt`.
- [ ] Add optional embedding-quality evaluation script if needed.

## D. JEPA fine-tuning experiments
- [ ] Fine-tune CTC from pretrained JEPA encoder on 100% labels.
- [ ] Fine-tune CTC from pretrained JEPA encoder on 50%, 25%, 10%, 5%, 1% labels using `train_jepa_finetune_ctc.py --data-fraction ...`.
- [ ] Compare against CTC-only baseline at each data fraction.
- [ ] Run frozen-encoder linear-probe CTC with `--freeze-encoder`.
- [ ] Compare full fine-tuning vs frozen encoder.

## E. Joint CTC + JEPA experiments
- [ ] Implement joint supervised CTC + JEPA auxiliary training.
- [ ] Add `train_joint_ctc_jepa.py` or a flag in `train_jepa_finetune_ctc.py`.
- [ ] Compare:
  - CTC only
  - JEPA pretrain + CTC finetune
  - CTC + JEPA joint training
  - JEPA pretrain + CTC + JEPA joint training
- [ ] Track WER/CER and collapse metrics.

## F. Robustness experiments
- [ ] Add test-time corruption options to `evaluate_ctc.py`:
  - `--eval-channel-dropout`
  - `--eval-noise-std`
  - `--eval-amp-scale`
  - `--eval-time-mask`
- [ ] Evaluate baseline and JEPA models under one dropped channel.
- [ ] Evaluate under two dropped channels.
- [ ] Evaluate under increasing Gaussian noise.
- [ ] Evaluate under amplitude rescaling.
- [ ] Plot WER/CER degradation curves.

## G. Cross-session / generalization experiments
- [ ] Add support for holding out sessions in `EMGDataset` split logic.
- [ ] Train on subset of sessions, test on held-out sessions.
- [ ] Compare baseline vs JEPA-pretrained model.
- [ ] Analyze whether JEPA improves session transfer.

## H. Architecture ablations
- [ ] Try smaller encoder sizes for faster iteration:
  - `model_size=256, num_layers=4`
  - `model_size=512, num_layers=4`
  - `model_size=768, num_layers=6`
- [ ] Compare sequence-level JEPA loss vs pooled/global JEPA loss.
- [ ] Try predictor dimensions: 128, 256, 512 (`JEPAModel(..., proj_dim=...)`).
- [ ] Try EMA momentum: 0.99, 0.996, 0.999.

## I. Augmentation ablations
- [ ] JEPA with channel dropout only.
- [ ] JEPA with time masking only.
- [ ] JEPA with noise/amplitude only.
- [ ] JEPA with no temporal shift.
- [ ] Tune strong vs weak augmentation severity.

## J. Reporting
- [ ] Create a results table: model, label fraction, WER, CER.
- [ ] Create robustness table: model, corruption type, WER, CER.
- [ ] Write a short methodology description.
- [ ] Save W&B run links in `TODO.md` or `RESULTS.md`.

---

## Maintainer note
- Current package layout remains intentionally flat (`semg_jepa/*.py`) for import stability. If we move to `semg_jepa/{data,models,training,...}/`, do it in one dedicated refactor PR with import/tests updates.
