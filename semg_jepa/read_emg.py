import json
import logging
import os
import random
import re
import string
from copy import copy
from functools import lru_cache

import numpy as np
import scipy
import torch

from .data_utils import TextTransform, get_emg_features, load_audio


def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, "highpass", fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)


def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1, 8):
        signal = notch(signal, freq * harmonic, sample_frequency)
    return signal


def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    return np.interp(sample_times, times, signal)


def apply_to_all(function, signal_array, *args, **kwargs):
    return np.stack([function(signal_array[:, i], *args, **kwargs) for i in range(signal_array.shape[1])], 1)


def _preprocess_raw_emg(base_dir, index, remove_channels=None):
    remove_channels = remove_channels or []
    raw_emg = np.load(os.path.join(base_dir, f"{index}_emg.npy"))

    before = os.path.join(base_dir, f"{index - 1}_emg.npy")
    after = os.path.join(base_dir, f"{index + 1}_emg.npy")
    raw_emg_before = np.load(before) if os.path.exists(before) else np.zeros([0, raw_emg.shape[1]])
    raw_emg_after = np.load(after) if os.path.exists(after) else np.zeros([0, raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0] - raw_emg_after.shape[0], :]
    emg_orig = apply_to_all(subsample, x, 689.06, 1000)

    for c in remove_channels:
        emg_orig[:, int(c)] = 0

    return emg_orig.astype(np.float32)


def _compute_ctc_length_from_raw(emg_orig):
    # mimic framing alignment used in the original pipeline where raw starts at offset 8.
    t = max(0, (emg_orig.shape[0] - 8) // 8)
    return int(t)


def load_utterance(base_dir, index, remove_channels=None, raw_only=True, limit_length=False):
    index = int(index)
    emg_orig = _preprocess_raw_emg(base_dir, index, remove_channels=remove_channels)

    t = _compute_ctc_length_from_raw(emg_orig)
    raw_emg = emg_orig[8:8 + 8 * t, :]
    assert raw_emg.shape[0] == 8 * t

    with open(os.path.join(base_dir, f"{index}_info.json")) as f:
        info = json.load(f)

    if raw_only:
        return {
            "raw_emg": raw_emg,
            "ctc_length": t,
            "text": info["text"],
            "book_location": (info["book"], info["sentence_index"]),
        }

    # Full path retained for debug/comparison with prior behavior.
    emg = apply_to_all(subsample, emg_orig, 516.79, 689.06)
    emg_features = get_emg_features(emg)
    mfccs = load_audio(os.path.join(base_dir, f"{index}_audio_clean.flac"), max_frames=min(emg_features.shape[0], 800 if limit_length else float("inf")))
    if emg_features.shape[0] > mfccs.shape[0]:
        emg_features = emg_features[: mfccs.shape[0], :]
    t_full = emg_features.shape[0]
    raw_emg_full = emg_orig[8:8 + 8 * t_full, :]
    assert raw_emg_full.shape[0] == 8 * t_full

    return {
        "raw_emg": raw_emg_full,
        "ctc_length": t_full,
        "text": info["text"],
        "book_location": (info["book"], info["sentence_index"]),
        "emg_features": emg_features,
        "audio_features": mfccs,
    }


class EMGDirectory:
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index


class SizeAwareSampler(torch.utils.data.Sampler):
    def __init__(self, emg_dataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch, batch_length = [], 0
        for idx in indices:
            directory_info, file_idx = self.dataset.example_indices[idx]
            with open(os.path.join(directory_info.directory, f"{file_idx}_info.json")) as f:
                info = json.load(f)
            if not np.any([l in string.ascii_letters for l in info["text"]]):
                continue
            length = sum([emg_len for emg_len, _, _ in info["chunks"]])
            if length > self.max_len:
                logging.warning("Example %s too long for configured max_len=%s", idx, self.max_len)
            if length + batch_length > self.max_len:
                yield batch
                batch, batch_length = [], 0
            batch.append(idx)
            batch_length += length


class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, config, limit_length=False, dev=False, test=False, no_testset=False, raw_only=True):
        self.config = config
        self.raw_only = raw_only

        if no_testset:
            devset, testset = [], []
        else:
            with open(config["testset_file"]) as f:
                testset_json = json.load(f)
            devset, testset = testset_json["dev"], testset_json["test"]

        directories = []
        for sd in config.get("silent_data_directories", []):
            for session_dir in sorted(os.listdir(sd)):
                directories.append(EMGDirectory(len(directories), os.path.join(sd, session_dir), True))
        has_silent = len(config.get("silent_data_directories", [])) > 0
        for vd in config.get("voiced_data_directories", []):
            for session_dir in sorted(os.listdir(vd)):
                directories.append(EMGDirectory(len(directories), os.path.join(vd, session_dir), False, exclude_from_testset=has_silent))

        self.example_indices = []
        for directory_info in directories:
            for fname in os.listdir(directory_info.directory):
                m = re.match(r"(\d+)_info.json", fname)
                if m is None:
                    continue
                idx = int(m.group(1))
                with open(os.path.join(directory_info.directory, fname)) as f:
                    info = json.load(f)
                if info["sentence_index"] < 0:
                    continue
                location = [info["book"], info["sentence_index"]]
                in_test, in_dev = location in testset, location in devset
                if (test and in_test and not directory_info.exclude_from_testset) or (dev and in_dev and not directory_info.exclude_from_testset) or (not test and not dev and not in_test and not in_dev):
                    self.example_indices.append((directory_info, idx))

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        self.limit_length = limit_length
        self.text_transform = TextTransform()

    def __len__(self):
        return len(self.example_indices)

    def subset(self, fraction):
        result = copy(self)
        result.example_indices = self.example_indices[: int(fraction * len(self.example_indices))]
        return result

    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        utterance = load_utterance(
            directory_info.directory,
            idx,
            remove_channels=self.config.get("remove_channels", []),
            raw_only=self.raw_only,
            limit_length=self.limit_length,
        )

        raw_emg = utterance["raw_emg"]
        t = utterance["ctc_length"]
        raw_emg = raw_emg / 20
        raw_emg = 50 * np.tanh(raw_emg / 50.0)

        text_int = np.array(self.text_transform.text_to_int(utterance["text"]), dtype=np.int64)
        session_ids = np.full(t, directory_info.session_index, dtype=np.int64)

        return {
            "raw_emg": torch.from_numpy(raw_emg).pin_memory(),
            "text": utterance["text"],
            "text_int": torch.from_numpy(text_int).pin_memory(),
            "session_ids": torch.from_numpy(session_ids).pin_memory(),
            "silent": directory_info.silent,
            "book_location": utterance["book_location"],
            "length": t,
        }

    @staticmethod
    def collate_raw(batch):
        return {
            "raw_emg": [ex["raw_emg"] for ex in batch],
            "session_ids": [ex["session_ids"] for ex in batch],
            "lengths": [ex["length"] for ex in batch],
            "silent": [ex["silent"] for ex in batch],
            "text_int": [ex["text_int"] for ex in batch],
            "text_int_lengths": [ex["text_int"].shape[0] for ex in batch],
            "text": [ex["text"] for ex in batch],
            "book_location": [ex["book_location"] for ex in batch],
        }
