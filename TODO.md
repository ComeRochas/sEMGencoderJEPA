# sEMG-JEPA Experiment Roadmap

## Setup / infrastructure
- [x] Precompute raw EMG cache (job 7650966, 0 errors — 8055/200/99 samples)
- [x] Fix in-place tensor mutation bug in `GaddyRawEMGEncoder`
- [x] Batched evaluation with CER metric in `ctc_utils.evaluate()`
- [x] W&B offline logging wired into all training scripts

## A. Sanity checks / reproduction
- [ ] Run `sbatch slurm/train_baseline.slurm` for a quick smoke test
- [ ] Confirm WER and CER are logged to stdout and W&B
- [ ] Confirm checkpoints save `best.pt` by WER and `last.pt` every epoch
- [ ] Confirm `evaluate_ctc.py` reports both WER and CER

## B. Baseline experiments
- [ ] Train full-data CTC baseline → `runs/baseline/`
- [ ] Train CTC baseline with label fractions: 50%, 25%, 10%, 5%, 1% using `--data-fraction` (yet to be implemented)
- [ ] Record WER/CER for each fraction in `RESULTS.md`

## C. JEPA pretraining experiments
- [ ] Train JEPA pretraining: `sbatch slurm/train_jepa.slurm` → `runs/jepa_pretrain/`
- [ ] Monitor cosine loss, variance loss, embedding std (watch for representation collapse)
- [ ] Confirm `pretrained_encoder.pt` is saved

## D. JEPA fine-tuning experiments
- [ ] Fine-tune from JEPA encoder: `sbatch slurm/finetune_from_jepa.slurm` → `runs/jepa_finetune/`
- [ ] Fine-tune at same label fractions as baseline (50%, 25%, 10%, 5%, 1%)
- [ ] Compare WER/CER vs baseline at each fraction

## E. Joint CTC + JEPA experiments (future)
- [ ] Implement joint supervised CTC + JEPA auxiliary training
- [ ] Compare: CTC-only / JEPA pretrain+finetune / joint CTC+JEPA

## F. Robustness experiments
- [ ] Add test-time corruption options to `evaluate_ctc.py` (channel dropout, noise, time mask)
- [ ] Evaluate baseline and JEPA models under one/two dropped channels
- [ ] Plot WER/CER degradation curves

## G. Cross-session generalization
- [ ] Add session hold-out split in dataset
- [ ] Train on subset of sessions, test on held-out → compare baseline vs JEPA

## H. Architecture ablations
- [ ] Smaller encoder: `model_size=256/512, num_layers=4`
- [ ] Predictor dimensions: 128, 256, 512
- [ ] EMA momentum: 0.99, 0.996, 0.999

## I. Augmentation ablations
- [ ] JEPA with channel dropout only / time masking only / no temporal shift
- [ ] Tune augmentation severity

## J. Reporting
- [ ] Create results table: model, label fraction, WER, CER
- [ ] Create robustness table: model, corruption type, WER, CER
- [ ] Save W&B run links in `RESULTS.md`
