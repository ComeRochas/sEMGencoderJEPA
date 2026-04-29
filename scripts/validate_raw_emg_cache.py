#!/usr/bin/env python
import argparse
import json


def parse_args():
    p = argparse.ArgumentParser(description="Validate cached raw-EMG data against live raw_only loading.")
    p.add_argument("--data-config", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--split", choices=["train", "dev", "test"], default="train")
    p.add_argument("--num-examples", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()

    import torch

    from semg_jepa.cached_dataset import CachedRawEMGDataset
    from semg_jepa.read_emg import EMGDataset

    with open(args.data_config) as f:
        config = json.load(f)

    live = EMGDataset(config, dev=args.split == "dev", test=args.split == "test", raw_only=True)
    cached = CachedRawEMGDataset(args.cache_dir, args.split)

    n = min(args.num_examples, len(live), len(cached))
    max_abs_diff = 0.0
    mismatched_texts = 0
    length_diffs = []

    for i in range(n):
        ex_live = live[i]
        ex_cache = cached[i]

        if ex_live["text"] != ex_cache["text"] or not torch.equal(ex_live["text_int"], ex_cache["text_int"]):
            mismatched_texts += 1

        if ex_live["raw_emg"].shape != ex_cache["raw_emg"].shape:
            print(f"shape mismatch idx={i}: live={tuple(ex_live['raw_emg'].shape)} cache={tuple(ex_cache['raw_emg'].shape)}")

        len_diff = ex_live["length"] - ex_cache["length"]
        length_diffs.append(len_diff)

        diff = (ex_live["raw_emg"].float() - ex_cache["raw_emg"].float()).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)

    mean_len_diff = sum(length_diffs) / max(1, len(length_diffs))
    max_len_diff = max([abs(x) for x in length_diffs], default=0)

    print("Validation summary")
    print(f"checked_examples: {n}")
    print(f"mean_length_diff: {mean_len_diff:.6f}")
    print(f"max_length_diff: {max_len_diff}")
    print(f"max_abs_raw_diff: {max_abs_diff:.6f}")
    print(f"mismatched_texts: {mismatched_texts}")


if __name__ == "__main__":
    main()
