import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from semg_jepa.architecture import BaselineCTCModel
from semg_jepa.cached_dataset import CachedRawEMGDataset, build_batches
from semg_jepa.ctc_utils import evaluate
from semg_jepa.data_utils import combine_fixed_length, decollate_tensor
from semg_jepa.wandb_utils import finish_wandb, init_wandb, wandb_log


def train(args):
    run = init_wandb(args)

    trainset = CachedRawEMGDataset(args.cache_dir, "train")
    devset = CachedRawEMGDataset(args.cache_dir, "dev")
    n_chars = len(devset.text_transform.chars)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = BaselineCTCModel(
        model_size=args.model_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        vocab_size=n_chars,
    ).to(device)

    if args.start_training_from:
        model.load_state_dict(torch.load(args.start_training_from, map_location=device), strict=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[125, 150, 175], gamma=0.5)

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group["lr"] = new_lr

    def schedule_lr(iteration):
        iteration += 1
        if iteration <= args.learning_rate_warmup:
            set_lr(iteration * args.learning_rate / args.learning_rate_warmup)

    os.makedirs(args.output_directory, exist_ok=True)
    best_wer = float("inf")
    global_step = 0
    optim.zero_grad()

    for epoch in range(args.epochs):
        batches = build_batches(trainset, args.max_batch_len)
        dataloader = torch.utils.data.DataLoader(
            trainset,
            pin_memory=(device == "cuda"),
            num_workers=0,
            collate_fn=CachedRawEMGDataset.collate_raw,
            batch_sampler=batches,
        )
        losses = []
        for example in tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            schedule_lr(global_step)
            raw = combine_fixed_length(example["raw_emg"], args.fixed_raw_len).to(device)
            pred = F.log_softmax(model(raw), dim=-1)
            pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, example["lengths"]), batch_first=False)
            targets = nn.utils.rnn.pad_sequence(example["text_int"], batch_first=True).to(device)
            loss = F.ctc_loss(pred, targets, example["lengths"], example["text_int_lengths"], blank=n_chars)
            loss.backward()
            losses.append(loss.item())

            if (global_step + 1) % args.grad_accum_steps == 0:
                optim.step()
                optim.zero_grad()
            global_step += 1

        train_loss = float(np.mean(losses)) if losses else 0.0
        wer, cer = evaluate(model, devset, device)
        lr_sched.step()

        logging.info("epoch=%s train_loss=%.4f dev_wer=%.3f dev_cer=%.3f", epoch + 1, train_loss, wer, cer)
        wandb_log(run, {"epoch": epoch + 1, "train_loss": train_loss, "dev_wer": wer, "dev_cer": cer})

        torch.save(model.state_dict(), os.path.join(args.output_directory, "last.pt"))
        if wer < best_wer:
            best_wer = wer
            torch.save(model.state_dict(), os.path.join(args.output_directory, "best.pt"))

    finish_wandb(run)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMGencoderJEPA/data")
    p.add_argument("--output-directory", default="/scratch/cr4206/sEMGencoderJEPA/runs/baseline")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-batch-len", type=int, default=128000)
    p.add_argument("--fixed-raw-len", type=int, default=1600)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--learning-rate-warmup", type=int, default=1000)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--model-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--start-training-from", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="UMLforVideoLab")
    p.add_argument("--wandb-project", default="JEPAforsEMG")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=[])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train(parse_args())
