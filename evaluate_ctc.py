import argparse
import json

import torch

from semg_jepa.architecture import BaselineCTCModel
from semg_jepa.ctc_utils import evaluate_wer
from semg_jepa.read_emg import EMGDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main(args):
    with open(args.data_config) as f:
        data_config = json.load(f)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    testset = EMGDataset(data_config, test=True)
    n_chars = len(testset.text_transform.chars)
    model = BaselineCTCModel(vocab_size=n_chars).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)

    wer = evaluate_wer(model, testset, device)
    print(f"WER: {wer:.4f}")


if __name__ == "__main__":
    main(parse_args())
