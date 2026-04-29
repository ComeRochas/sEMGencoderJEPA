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

from .data_utils import FeatureNormalizer, TextTransform, get_emg_features, load_audio, phoneme_inventory, read_phonemes


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


def load_utterance(base_dir, index, remove_channels=None, limit_length=False, text_align_directory="text_alignments"):
    remove_channels = remove_channels or []
    index = int(index)
    raw_emg = np.load(os.path.join(base_dir, f"{index}_emg.npy"))
    before, after = os.path.join(base_dir, f"{index - 1}_emg.npy"), os.path.join(base_dir, f"{index + 1}_emg.npy")
    raw_emg_before = np.load(before) if os.path.exists(before) else np.zeros([0, raw_emg.shape[1]])
    raw_emg_after = np.load(after) if os.path.exists(after) else np.zeros([0, raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0] - raw_emg_after.shape[0], :]
    emg_orig = apply_to_all(subsample, x, 689.06, 1000)
    emg = apply_to_all(subsample, x, 516.79, 1000)

    for c in remove_channels:
        emg[:, int(c)] = 0
        emg_orig[:, int(c)] = 0

    emg_features = get_emg_features(emg)
    mfccs = load_audio(os.path.join(base_dir, f"{index}_audio_clean.flac"), max_frames=min(emg_features.shape[0], 800 if limit_length else float("inf")))
    if emg_features.shape[0] > mfccs.shape[0]:
        emg_features = emg_features[:mfccs.shape[0], :]
    emg = emg[6:6 + 6 * emg_features.shape[0], :]
    emg_orig = emg_orig[8:8 + 8 * emg_features.shape[0], :]

    with open(os.path.join(base_dir, f"{index}_info.json")) as f:
        info = json.load(f)

    sess = os.path.basename(base_dir)
    tg_fname = f"{text_align_directory}/{sess}/{sess}_{index}_audio.TextGrid"
    phonemes = read_phonemes(tg_fname, mfccs.shape[0]) if os.path.exists(tg_fname) else np.zeros(mfccs.shape[0], dtype=np.int64) + phoneme_inventory.index("sil")

    return mfccs, emg_features, info["text"], (info["book"], info["sentence_index"]), phonemes, emg_orig.astype(np.float32)


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
    def __init__(self, config, limit_length=False, dev=False, test=False, no_testset=False, no_normalizers=False):
        self.config = config
        self.text_align_directory = config.get("text_align_directory", "text_alignments")

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
        self.voiced_data_locations = {}
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
                if not directory_info.silent:
                    self.voiced_data_locations[(info["book"], info["sentence_index"])] = (directory_info, idx)

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        self.no_normalizers = no_normalizers
        if not self.no_normalizers:
            import pickle
            self.mfcc_norm, self.emg_norm = pickle.load(open(config["normalizers_file"], "rb"))

        sample_mfccs, sample_emg, _, _, _, _ = load_utterance(
            self.example_indices[0][0].directory,
            self.example_indices[0][1],
            remove_channels=config.get("remove_channels", []),
            text_align_directory=self.text_align_directory,
        )
        self.num_speech_features = sample_mfccs.shape[1]
        self.num_features = sample_emg.shape[1]
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
        mfccs, emg, text, book_location, phonemes, raw_emg = load_utterance(
            directory_info.directory,
            idx,
            remove_channels=self.config.get("remove_channels", []),
            limit_length=self.limit_length,
            text_align_directory=self.text_align_directory,
        )
        raw_emg = raw_emg / 20
        raw_emg = 50 * np.tanh(raw_emg / 50.0)
        if not self.no_normalizers:
            mfccs = self.mfcc_norm.normalize(mfccs)
            emg = self.emg_norm.normalize(emg)
            emg = 8 * np.tanh(emg / 8.0)

        session_ids = np.full(emg.shape[0], directory_info.session_index, dtype=np.int64)
        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)
        result = {
            "audio_features": torch.from_numpy(mfccs).pin_memory(),
            "emg": torch.from_numpy(emg).pin_memory(),
            "raw_emg": torch.from_numpy(raw_emg).pin_memory(),
            "text": text,
            "text_int": torch.from_numpy(text_int).pin_memory(),
            "session_ids": torch.from_numpy(session_ids).pin_memory(),
            "silent": directory_info.silent,
            "phonemes": torch.from_numpy(phonemes).pin_memory(),
            "book_location": book_location,
        }

        if directory_info.silent:
            voiced_directory, voiced_idx = self.voiced_data_locations[book_location]
            voiced_mfccs, voiced_emg, _, _, _, _ = load_utterance(voiced_directory.directory, voiced_idx, text_align_directory=self.text_align_directory)
            if not self.no_normalizers:
                voiced_mfccs = self.mfcc_norm.normalize(voiced_mfccs)
                voiced_emg = self.emg_norm.normalize(voiced_emg)
                voiced_emg = 8 * np.tanh(voiced_emg / 8.0)
            result["parallel_voiced_audio_features"] = torch.from_numpy(voiced_mfccs).pin_memory()
            result["parallel_voiced_emg"] = torch.from_numpy(voiced_emg).pin_memory()
        return result

    @staticmethod
    def collate_raw(batch):
        audio_features, audio_feature_lengths, parallel_emg = [], [], []
        for ex in batch:
            if ex["silent"]:
                audio_features.append(ex["parallel_voiced_audio_features"])
                audio_feature_lengths.append(ex["parallel_voiced_audio_features"].shape[0])
                parallel_emg.append(ex["parallel_voiced_emg"])
            else:
                audio_features.append(ex["audio_features"])
                audio_feature_lengths.append(ex["audio_features"].shape[0])
                parallel_emg.append(np.zeros(1))

        return {
            "audio_features": audio_features,
            "audio_feature_lengths": audio_feature_lengths,
            "emg": [ex["emg"] for ex in batch],
            "raw_emg": [ex["raw_emg"] for ex in batch],
            "parallel_voiced_emg": parallel_emg,
            "phonemes": [ex["phonemes"] for ex in batch],
            "session_ids": [ex["session_ids"] for ex in batch],
            "lengths": [ex["emg"].shape[0] for ex in batch],
            "silent": [ex["silent"] for ex in batch],
            "text_int": [ex["text_int"] for ex in batch],
            "text_int_lengths": [ex["text_int"].shape[0] for ex in batch],
        }
