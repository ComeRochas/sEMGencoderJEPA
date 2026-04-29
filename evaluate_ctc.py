import argparse
import json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", default=None)
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--split", choices=["train", "dev", "test"], default="test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def _load_checkpoint_robust(model, checkpoint_path, device):
    import torch

    state = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing and all(k.startswith("encoder.") or k.startswith("ctc_head.") for k in state.keys()):
        remapped = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                remapped[k] = v
            elif k.startswith("ctc_head."):
                remapped[k] = v
        model.load_state_dict(remapped, strict=False)
    return missing, unexpected


def main(args):
    import torch

    from semg_jepa.architecture import BaselineCTCModel
    from semg_jepa.cached_dataset import CachedRawEMGDataset
    from semg_jepa.ctc_utils import evaluate_text_metrics
    from semg_jepa.read_emg import EMGDataset

    if args.use_cache:
        if not args.cache_dir:
            raise ValueError("--use-cache requires --cache-dir")
        dataset = CachedRawEMGDataset(args.cache_dir, args.split)
    else:
        if not args.data_config:
            raise ValueError("--data-config is required when not using --use-cache")
        with open(args.data_config) as f:
            data_config = json.load(f)
        dataset = EMGDataset(data_config, dev=args.split == "dev", test=args.split == "test", raw_only=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    n_chars = len(dataset.text_transform.chars)
    model = BaselineCTCModel(model_size=args.model_size, num_layers=args.num_layers, dropout=args.dropout, vocab_size=n_chars).to(device)
    _load_checkpoint_robust(model, args.checkpoint, device)

    metrics = evaluate_text_metrics(model, dataset, device)
    print(f"Split: {args.split}")
    print(f"WER: {metrics['wer']:.4f}")
    print(f"CER: {metrics['cer']:.4f}")


if __name__ == "__main__":
    main(parse_args())
