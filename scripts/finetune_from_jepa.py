import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from semg_jepa.architecture import CTCHead, GaddyRawEMGEncoder
from semg_jepa.cached_dataset import CachedRawEMGDataset, build_batches
from semg_jepa.ctc_utils import evaluate
from semg_jepa.data_utils import combine_fixed_length, decollate_tensor
from semg_jepa.wandb_utils import finish_wandb, init_wandb, wandb_log


class FinetuneCTCModel(nn.Module):
    def __init__(self, encoder, ctc_head):
        super().__init__()
        self.encoder = encoder
        self.ctc_head = ctc_head

    def forward(self, raw_emg):
        return self.ctc_head(self.encoder(raw_emg))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMGencoderJEPA/data")
    p.add_argument("--pretrained-encoder", required=True)
    p.add_argument("--output-directory", default="/scratch/cr4206/sEMGencoderJEPA/runs/jepa_finetune")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--max-batch-len", type=int, default=128000)
    p.add_argument("--fixed-raw-len", type=int, default=1600)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--model-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="UMLforVideoLab")
    p.add_argument("--wandb-project", default="JEPAforsEMG")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=[])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def train(args):
    run = init_wandb(args)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    trainset = CachedRawEMGDataset(args.cache_dir, "train")
    devset = CachedRawEMGDataset(args.cache_dir, "dev")
    n_chars = len(devset.text_transform.chars)

    encoder = GaddyRawEMGEncoder(model_size=args.model_size, num_layers=args.num_layers, dropout=args.dropout)
    encoder.load_state_dict(torch.load(args.pretrained_encoder, map_location=device), strict=False)
    head = CTCHead(model_size=args.model_size, vocab_size=n_chars)
    model = FinetuneCTCModel(encoder, head).to(device)

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    os.makedirs(args.output_directory, exist_ok=True)
    best_wer = float("inf")

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
        for example in tqdm.tqdm(dataloader, desc=f"Finetune epoch {epoch + 1}"):
            raw = combine_fixed_length(example["raw_emg"], args.fixed_raw_len).to(device)
            pred = F.log_softmax(model(raw), dim=-1)
            pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, example["lengths"]), batch_first=False)
            targets = nn.utils.rnn.pad_sequence(example["text_int"], batch_first=True).to(device)
            loss = F.ctc_loss(pred, targets, example["lengths"], example["text_int_lengths"], blank=n_chars)

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else 0.0
        wer, cer = evaluate(model, devset, device)
        logging.info("epoch=%s train_loss=%.4f dev_wer=%.3f dev_cer=%.3f", epoch + 1, train_loss, wer, cer)
        wandb_log(run, {"epoch": epoch + 1, "train_loss": train_loss, "dev_wer": wer, "dev_cer": cer})

        torch.save(model.state_dict(), os.path.join(args.output_directory, "last.pt"))
        if wer < best_wer:
            best_wer = wer
            torch.save(model.state_dict(), os.path.join(args.output_directory, "best.pt"))

    finish_wandb(run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train(parse_args())
