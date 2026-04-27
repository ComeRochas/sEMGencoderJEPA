import torch
import torch.nn.functional as F
import tqdm
from ctcdecode import CTCBeamDecoder

from .metrics import compute_text_metrics


def build_decoder(chars):
    blank_id = len(chars)
    return CTCBeamDecoder(chars + "_", blank_id=blank_id, log_probs_input=True, model_path="lm.binary", alpha=1.5, beta=1.85)


def decode_predictions(model, dataset, device, batch_size=1):
    model.eval()
    decoder = build_decoder(dataset.text_transform.chars)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    references, predictions = [], []
    with torch.no_grad():
        for example in tqdm.tqdm(dataloader, "Evaluate", disable=None):
            raw = example["raw_emg"].to(device)
            log_probs = F.log_softmax(model(raw), -1)
            beam_results, _, _, out_lens = decoder.decode(log_probs)
            pred_int = beam_results[0, 0, : out_lens[0, 0]].tolist()
            predictions.append(dataset.text_transform.int_to_text(pred_int))
            references.append(dataset.text_transform.clean_text(example["text"][0]))
    model.train()
    return references, predictions


def evaluate_text_metrics(model, dataset, device, batch_size=1):
    references, predictions = decode_predictions(model, dataset, device, batch_size=batch_size)
    return compute_text_metrics(references, predictions)
