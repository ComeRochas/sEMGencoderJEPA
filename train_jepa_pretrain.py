import argparse
import copy
import json
import logging
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", default=None)
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output-directory", default="output_jepa_pretrain")
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
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="UMLforVideoLab")
    p.add_argument("--wandb-project", default="JEPAforsEMG")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    return p.parse_args()


def train(args):
    import torch
    import torch.nn.functional as F
    import tqdm

    from semg_jepa.architecture import GaddyRawEMGEncoder
    from semg_jepa.augmentations import RawEMGAugment
    from semg_jepa.cached_dataset import CachedRawEMGDataset
    from semg_jepa.data_utils import combine_fixed_length
    from semg_jepa.jepa_utils import embedding_std_mean, update_ema, variance_regularizer
    from semg_jepa.read_emg import EMGDataset, SizeAwareSampler
    from semg_jepa.wandb_utils import finish_wandb, init_wandb, wandb_log

    class JEPAModel(torch.nn.Module):
        def __init__(self, encoder: GaddyRawEMGEncoder):
            super().__init__()
            self.encoder = encoder
            hidden = encoder.w_raw_in.out_features
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, hidden),
            )

        def forward_student(self, raw):
            return self.predictor(self.encoder(raw))

        def forward_teacher(self, raw):
            return self.encoder(raw)

    if args.use_cache:
        if not args.cache_dir:
            raise ValueError("--use-cache requires --cache-dir")
        trainset = CachedRawEMGDataset(args.cache_dir, "train")
    else:
        if not args.data_config:
            raise ValueError("--data-config is required when not using --use-cache")
        with open(args.data_config) as f:
            data_config = json.load(f)
        trainset = EMGDataset(data_config, dev=False, test=False, raw_only=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dataloader = torch.utils.data.DataLoader(
        trainset,
        pin_memory=(device == "cuda"),
        num_workers=0,
        collate_fn=trainset.collate_raw,
        batch_sampler=SizeAwareSampler(trainset, args.max_batch_len),
    )

    student = JEPAModel(GaddyRawEMGEncoder(model_size=args.model_size, num_layers=args.num_layers, dropout=args.dropout)).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    weak_aug = RawEMGAugment(channel_dropout=0.05, time_mask_prob=0.2, time_mask_max=20, amp_scale=0.05, noise_std=0.005, temporal_shift=2)
    strong_aug = RawEMGAugment(channel_dropout=0.2, time_mask_prob=0.7, time_mask_max=80, amp_scale=0.2, noise_std=0.02, temporal_shift=8)

    optim = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    run = init_wandb(args)

    os.makedirs(args.output_directory, exist_ok=True)

    for epoch in range(args.epochs):
        loss_sum = cosine_sum = variance_sum = std_sum = 0.0
        n = 0
        for example in tqdm.tqdm(dataloader, desc="JEPA pretrain"):
            raw = combine_fixed_length(example["raw_emg"], args.fixed_raw_len).to(device)
            student_view = strong_aug(raw)
            teacher_view = weak_aug(raw)

            pred = F.normalize(student.forward_student(student_view), dim=-1)
            with torch.no_grad():
                target = F.normalize(teacher.forward_teacher(teacher_view), dim=-1)

            cosine_loss = 1 - (pred * target).sum(dim=-1).mean()
            var_loss = variance_regularizer(pred)
            emb_std = embedding_std_mean(pred)
            loss = cosine_loss + args.var_weight * var_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            update_ema(student, teacher, args.ema_momentum)

            loss_sum += loss.item()
            cosine_sum += cosine_loss.item()
            variance_sum += var_loss.item()
            std_sum += emb_std.item()
            n += 1

        train_loss = loss_sum / max(1, n)
        train_cosine = cosine_sum / max(1, n)
        train_variance = variance_sum / max(1, n)
        train_std = std_sum / max(1, n)

        logging.info(
            "epoch=%s train/loss=%.4f train/jepa_cosine_loss=%.4f train/variance_loss=%.4f train/embedding_std_mean=%.4f",
            epoch + 1,
            train_loss,
            train_cosine,
            train_variance,
            train_std,
        )

        wandb_log(
            run,
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/jepa_cosine_loss": train_cosine,
                "train/variance_loss": train_variance,
                "train/embedding_std_mean": train_std,
            },
        )

        torch.save(student.state_dict(), os.path.join(args.output_directory, "student_last.pt"))

    torch.save(student.encoder.state_dict(), os.path.join(args.output_directory, "pretrained_encoder.pt"))
    finish_wandb(run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train(parse_args())
