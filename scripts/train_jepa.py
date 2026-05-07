import argparse
import copy
import logging
import os
import time

import torch
import torch.nn.functional as F

from semg_jepa.architecture import GaddyRawEMGEncoder
from semg_jepa.augmentations import RawEMGAugment
from semg_jepa.cached_dataset import CachedRawEMGDataset, build_batches
from semg_jepa.config_utils import parse_with_config, setup_stdout_logging
from semg_jepa.data_utils import combine_fixed_length
from semg_jepa.jepa_utils import (
    cosine_momentum_schedule,
    covariance_regularizer,
    feature_std_stats,
    update_ema,
    variance_regularizer,
    warmup_cosine_lr,
)
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to YAML config; CLI flags override its values.")
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMGencoderJEPA/data")
    p.add_argument("--output-directory", default="/scratch/cr4206/sEMGencoderJEPA/runs/jepa_pretrain")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-batch-len", type=int, default=88000)
    p.add_argument("--fixed-raw-len", type=int, default=1600)
    p.add_argument("--grad-accum-steps", type=int, default=2)

    # Optimizer / LR schedule
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--learning-rate-warmup", type=int, default=1000)
    p.add_argument("--min-learning-rate", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # EMA momentum schedule
    p.add_argument("--ema-momentum-base", type=float, default=0.996)
    p.add_argument("--ema-momentum-final", type=float, default=0.9999)

    # Loss weights (VICReg defaults)
    p.add_argument("--inv-weight", type=float, default=25.0)
    p.add_argument("--var-weight", type=float, default=25.0)
    p.add_argument("--cov-weight", type=float, default=1.0)
    p.add_argument("--var-target-std", type=float, default=1.0)

    # Architecture
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--model-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="UMLforVideoLab")
    p.add_argument("--wandb-project", default="JEPAforsEMG")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=[])
    p.add_argument("--cpu", action="store_true")
    return parse_with_config(p)


def train(args):
    run = init_wandb(args, default_name_prefix="jepa")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    trainset = CachedRawEMGDataset(args.cache_dir, "train")

    student = JEPAModel(
        GaddyRawEMGEncoder(
            model_size=args.model_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ),
        proj_dim=args.proj_dim,
    ).to(device)
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
    run_ts = time.strftime("%Y%m%d_%H%M")

    # Estimate total optimizer steps for the schedules. build_batches is stochastic
    # (shuffled each epoch) but batch *count* is essentially constant given a fixed
    # max_batch_len, so probing one epoch is enough.
    steps_per_epoch = max(1, len(build_batches(trainset, args.max_batch_len)) // args.grad_accum_steps)
    total_optim_steps = max(1, steps_per_epoch * args.epochs)
    logging.info(
        "estimated steps_per_epoch=%d total_optim_steps=%d warmup=%d",
        steps_per_epoch, total_optim_steps, args.learning_rate_warmup,
    )

    def set_lr(new_lr):
        for pg in optim.param_groups:
            pg["lr"] = new_lr

    optim_step = 0
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
        sums = {"inv": 0.0, "var": 0.0, "cov": 0.0, "total": 0.0,
                "std_mean": 0.0, "std_min": 0.0}
        n_batches = 0
        epoch_start = time.perf_counter()

        for batch_idx, example in enumerate(dataloader):
            raw = combine_fixed_length(example["raw_emg"], args.fixed_raw_len).to(device)
            student_view = strong_aug(raw)
            teacher_view = weak_aug(raw)

            pred = student.forward_student(student_view)
            with torch.no_grad():
                target = teacher.forward_teacher(teacher_view)

            # LayerNorm (no learnable affine) stabilizes scale so var-hinge target=1
            # is reachable and smooth-L1 lives on a controlled scale.
            d = pred.shape[-1]
            pred_n = F.layer_norm(pred, (d,))
            target_n = F.layer_norm(target, (d,))

            inv_loss = F.smooth_l1_loss(pred_n, target_n)

            z = pred_n.flatten(0, 1)  # [B*T, D] for VICReg-style stats
            var_loss = variance_regularizer(z, target_std=args.var_target_std)
            cov_loss = covariance_regularizer(z)

            loss = (args.inv_weight * inv_loss
                    + args.var_weight * var_loss
                    + args.cov_weight * cov_loss)

            (loss / args.grad_accum_steps).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # Update LR & EMA momentum on the optimizer-step clock.
                lr = warmup_cosine_lr(
                    optim_step, args.learning_rate_warmup, total_optim_steps,
                    args.learning_rate, args.min_learning_rate,
                )
                set_lr(lr)
                optim.step()
                optim.zero_grad()

                momentum = cosine_momentum_schedule(
                    optim_step, total_optim_steps,
                    args.ema_momentum_base, args.ema_momentum_final,
                )
                update_ema(student, teacher, momentum)
                optim_step += 1

            with torch.no_grad():
                std_mean, std_min = feature_std_stats(z)
            sums["inv"] += inv_loss.item()
            sums["var"] += var_loss.item()
            sums["cov"] += cov_loss.item()
            sums["total"] += loss.item()
            sums["std_mean"] += std_mean.item()
            sums["std_min"] += std_min.item()
            n_batches += 1

        avg = {k: v / max(1, n_batches) for k, v in sums.items()}
        cur_lr = optim.param_groups[0]["lr"]
        cur_mom = cosine_momentum_schedule(
            optim_step, total_optim_steps, args.ema_momentum_base, args.ema_momentum_final,
        )
        t_epoch = time.perf_counter() - epoch_start

        logging.info(
            "epoch=%d/%d step=%d lr=%.2e mom=%.4f "
            "inv=%.4f var=%.4f cov=%.4f total=%.4f "
            "std_mean=%.3f std_min=%.3f t_epoch=%.1fs",
            epoch + 1, args.epochs, optim_step, cur_lr, cur_mom,
            avg["inv"], avg["var"], avg["cov"], avg["total"],
            avg["std_mean"], avg["std_min"], t_epoch,
        )
        wandb_log(run, {
            "train/inv_loss": avg["inv"],
            "train/var_loss": avg["var"],
            "train/cov_loss": avg["cov"],
            "train/loss": avg["total"],
            "train/lr": cur_lr,
            "train/ema_momentum": cur_mom,
            "diag/feat_std_mean": avg["std_mean"],
            "diag/feat_std_min": avg["std_min"],
            "time/epoch": t_epoch,
            "epoch": epoch + 1,
        })

        torch.save(student.state_dict(), os.path.join(args.output_directory, f"student_last_{run_ts}.pt"))

    torch.save(student.encoder.state_dict(), os.path.join(args.output_directory, f"pretrained_encoder_{run_ts}.pt"))
    finish_wandb(run)


if __name__ == "__main__":
    setup_stdout_logging()
    train(parse_args())
