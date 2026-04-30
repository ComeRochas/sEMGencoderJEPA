import argparse
import copy
import logging
import os

import torch
import torch.nn.functional as F
import tqdm

from semg_jepa.architecture import GaddyRawEMGEncoder
from semg_jepa.augmentations import RawEMGAugment
from semg_jepa.cached_dataset import CachedRawEMGDataset, build_batches
from semg_jepa.data_utils import combine_fixed_length
from semg_jepa.wandb_utils import finish_wandb, init_wandb, wandb_log


class JEPAModel(torch.nn.Module):
    def __init__(self, encoder: GaddyRawEMGEncoder, proj_dim=256):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.w_raw_in.out_features
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, proj_dim),
        )
        self.target_proj = torch.nn.Linear(hidden, proj_dim)

    def forward_student(self, raw):
        return self.predictor(self.encoder(raw))

    def forward_teacher(self, raw):
        return self.target_proj(self.encoder(raw))


def update_ema(student, teacher, momentum):
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


def variance_regularizer(z, eps=1e-4):
    std = torch.sqrt(z.var(dim=(0, 1), unbiased=False) + eps)
    return torch.mean(F.relu(1 - std))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMGencoderJEPA/data")
    p.add_argument("--output-directory", default="/scratch/cr4206/sEMGencoderJEPA/runs/jepa_pretrain")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-batch-len", type=int, default=128000)
    p.add_argument("--fixed-raw-len", type=int, default=1600)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ema-momentum", type=float, default=0.996)
    p.add_argument("--var-weight", type=float, default=1.0)
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

    student = JEPAModel(GaddyRawEMGEncoder(
        model_size=args.model_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    weak_aug = RawEMGAugment(
        channel_dropout=0.05, time_mask_prob=0.2, time_mask_max=20,
        amp_scale=0.05, noise_std=0.005, temporal_shift=2,
    )
    strong_aug = RawEMGAugment(
        channel_dropout=0.2, time_mask_prob=0.7, time_mask_max=80,
        amp_scale=0.2, noise_std=0.02, temporal_shift=8,
    )

    optim = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    os.makedirs(args.output_directory, exist_ok=True)

    for epoch in range(args.epochs):
        batches = build_batches(trainset, args.max_batch_len)
        dataloader = torch.utils.data.DataLoader(
            trainset,
            pin_memory=(device == "cuda"),
            num_workers=0,
            collate_fn=CachedRawEMGDataset.collate_raw,
            batch_sampler=batches,
        )
        running = 0.0
        n = 0
        for example in tqdm.tqdm(dataloader, desc=f"JEPA pretrain epoch {epoch + 1}"):
            raw = combine_fixed_length(example["raw_emg"], args.fixed_raw_len).to(device)
            student_view = strong_aug(raw)
            teacher_view = weak_aug(raw)

            pred = F.normalize(student.forward_student(student_view), dim=-1)
            with torch.no_grad():
                target = F.normalize(teacher.forward_teacher(teacher_view), dim=-1)

            cosine_loss = 1 - (pred * target).sum(dim=-1).mean()
            var_loss = variance_regularizer(pred)
            loss = cosine_loss + args.var_weight * var_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            update_ema(student, teacher, args.ema_momentum)

            running += loss.item()
            n += 1

        avg_loss = running / max(1, n)
        logging.info("epoch=%s loss=%.4f", epoch + 1, avg_loss)
        wandb_log(run, {"epoch": epoch + 1, "train_loss": avg_loss})
        torch.save(student.state_dict(), os.path.join(args.output_directory, "student_last.pt"))

    torch.save(student.encoder.state_dict(), os.path.join(args.output_directory, "pretrained_encoder.pt"))
    finish_wandb(run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train(parse_args())
