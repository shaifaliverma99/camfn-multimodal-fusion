"""EEG modality datasets.

Two real, open sources are wired up (see docs/DATA_SOURCES.md):

1. `CHBMITWindowDataset` — raw multi-channel scalp EEG from CHB-MIT
   (`.edf` files), windowed and preprocessed with the functions in
   `preprocessing.py`. Stands in for TUH EEG.
2. `EpilepticSeizureCSVDataset` — the open "Epileptic Seizure Recognition"
   dataset: single-channel, 178-sample windows, 5-way activity label. Useful
   as a fast, dependency-light EEG source for smoke-testing.

Both yield `[C, T]` float tensors so `EEGEncoder` doesn't care which one it
was handed.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import bandpass_filter, notch_filter, simplified_asr_reject

try:
    import mne

    mne.set_log_level("ERROR")
    _HAS_MNE = True
except ImportError:  # pragma: no cover
    _HAS_MNE = False


class CHBMITWindowDataset(Dataset):
    """Fixed-length windows cut from CHB-MIT `.edf` recordings.

    Every window gets the full spec-3 pipeline: bandpass -> notch ->
    simplified artifact rejection. There is no per-window seizure label
    wired up yet (that requires parsing the CHB-MIT `*-summary.txt`
    seizure-interval annotations); `label` is a placeholder 0 until that
    parser is added. This dataset is meant to validate the preprocessing +
    encoder path against real multi-channel EEG, not to train a seizure
    detector out of the box.
    """

    def __init__(self, root: str | Path, window_seconds: float = 4.0, target_fs: float = 256.0, max_files: int | None = None):
        if not _HAS_MNE:
            raise ImportError("mne is required for CHBMITWindowDataset (pip install mne)")
        self.root = Path(root)
        self.window_seconds = window_seconds
        self.target_fs = target_fs
        self._windows: list[tuple[Path, int, float]] = []  # (file, start_sample, fs)

        edf_files = sorted(self.root.glob("chb*/*.edf"))
        if max_files is not None:
            edf_files = edf_files[:max_files]
        for edf_path in edf_files:
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
            except Exception:
                continue
            fs = raw.info["sfreq"]
            n_samples = raw.n_times
            win_samples = int(window_seconds * fs)
            if win_samples <= 0 or n_samples < win_samples:
                continue
            n_windows = n_samples // win_samples
            for w in range(n_windows):
                self._windows.append((edf_path, w * win_samples, fs))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        edf_path, start, fs = self._windows[idx]
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        win_samples = int(self.window_seconds * fs)
        data = raw.get_data(start=start, stop=start + win_samples)  # [C, T]

        data = bandpass_filter(data, fs=fs)
        try:  # CHB-MIT is US-recorded -> 60 Hz mains; skip if fs too low to filter it

            data = notch_filter(data, fs=fs, freq=60.0)
        except ValueError:
            pass
        data = simplified_asr_reject(data)

        tensor = torch.tensor(data, dtype=torch.float32)
        return {
            "eeg": tensor,
            "seizure_label": 0,  # placeholder — see class docstring
            "source_path": str(edf_path),
        }


class EpilepticSeizureCSVDataset(Dataset):
    """The open Kaggle/UCI 'Epileptic Seizure Recognition' dataset.

    178-sample single-channel EEG segments, 5-class activity label
    (1 = seizure activity; 2-5 = non-seizure states per the dataset's
    codebook). Loaded as `[1, 178]` tensors so it shares the `EEGEncoder`
    with CHB-MIT windows.
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.rows: list[tuple[np.ndarray, int]] = []
        with open(self.csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            label_idx = header.index("y")
            feature_idxs = [i for i, h in enumerate(header) if h not in ("y",) and h != header[0]]
            for row in reader:
                features = np.array([float(row[i]) for i in feature_idxs], dtype=np.float32)
                label = int(row[label_idx])
                self.rows.append((features, label))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        features, label = self.rows[idx]
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, 178]
        return {
            "eeg": tensor,
            "seizure_label": 1 if label == 1 else 0,  # 1=seizure activity per codebook
            "source_path": str(self.csv_path),
        }
