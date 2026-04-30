import jiwer
import torch
import torch.nn.functional as F
import tqdm
from pyctcdecode import build_ctcdecoder


def build_decoder(chars):
    labels = list(chars) + [""]  # blank at index len(chars)
    return build_ctcdecoder(labels, kenlm_model_path="lm.binary", alpha=1.5, beta=1.85)


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
        for raw_padded, seq_lens, texts in tqdm.tqdm(dataloader, "Evaluate", disable=None):
            raw_padded = raw_padded.to(device)
            log_probs = F.log_softmax(model(raw_padded), -1).cpu().numpy()
            for i in range(len(texts)):
                T = int(seq_lens[i])
                predictions.append(decoder.decode(log_probs[i, :T]))
                references.append(dataset.text_transform.clean_text(texts[i]))
    model.train()
    return jiwer.wer(references, predictions), jiwer.cer(references, predictions)
