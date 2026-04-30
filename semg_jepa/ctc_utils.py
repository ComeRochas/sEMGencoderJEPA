import torch
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder

from semg_jepa.metrics import compute_cer, compute_wer


def build_decoder(chars):
    labels = list(chars) + [""]  # blank at index len(chars)
    try:
        return build_ctcdecoder(labels, kenlm_model_path="data/lm.binary", alpha=1.5, beta=1.85)
    except Exception as e:
        print(f"[ctc_utils] LM-backed decoder unavailable ({type(e).__name__}: {e}); "
              f"falling back to no-LM beam search. WER will be higher.", flush=True)
        return build_ctcdecoder(labels)


def _collate_eval(batch):
    """Pad raw_emg sequences to the longest in the batch."""
    raw_list = [ex["raw_emg"] for ex in batch]
    seq_lens = torch.tensor([r.shape[0] // 8 for r in raw_list])
    raw_padded = torch.nn.utils.rnn.pad_sequence(raw_list, batch_first=True)
    texts = [ex["text"] for ex in batch]
    return raw_padded, seq_lens, texts


def evaluate(model, dataset, device, batch_size=16):
    """Return (wer, cer) on dataset using batched CTC beam-search decoding."""
    model.eval()
    decoder = build_decoder(dataset.text_transform.chars)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=_collate_eval,
    )
    references, predictions = [], []
    with torch.no_grad():
        for raw_padded, seq_lens, texts in dataloader:
            raw_padded = raw_padded.to(device)
            log_probs = F.log_softmax(model(raw_padded), -1).cpu().numpy()
            for i in range(len(texts)):
                T = int(seq_lens[i])
                predictions.append(decoder.decode(log_probs[i, :T]))
                references.append(dataset.text_transform.clean_text(texts[i]))
    model.train()
    return compute_wer(references, predictions), compute_cer(references, predictions)
