import argparse

import torch

from semg_jepa.architecture import BaselineCTCModel
from semg_jepa.cached_dataset import CachedRawEMGDataset
from semg_jepa.ctc_utils import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMGencoderJEPA/data")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dataset = CachedRawEMGDataset(args.cache_dir, args.split)
    n_chars = len(dataset.text_transform.chars)
    model = BaselineCTCModel(vocab_size=n_chars).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    wer, cer = evaluate(model, dataset, device)
    print(f"WER ({args.split}): {wer:.4f}   CER ({args.split}): {cer:.4f}")


if __name__ == "__main__":
    main(parse_args())
