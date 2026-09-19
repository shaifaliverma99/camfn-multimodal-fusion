"""Vocal-acoustic modality dataset — UCI Parkinson's voice dataset.

Real, open, tabular dataset of 22 acoustic biomarker features (jitter,
shimmer, HNR, RPDE, DFA, spread1/2, D2, PPE, ...) per phonation recording,
with a healthy/PD `status` label. Stands in for PPMI/mPower voice/UPDRS
streams — see docs/DATA_SOURCES.md.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

FEATURE_COLUMNS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
    "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]  # 22 features, matches Datasets/Parkinson_voice/processed_npy shapes


class ParkinsonVoiceDataset(Dataset):
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.rows: list[tuple[list[float], int]] = []
        with open(self.data_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                features = [float(row[c]) for c in FEATURE_COLUMNS]
                label = int(row["status"])  # 1 = Parkinson's, 0 = healthy
                self.rows.append((features, label))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        features, label = self.rows[idx]
        tensor = torch.tensor(features, dtype=torch.float32)
        return {
            "acoustic": tensor,  # [22]
            "pd_voice_label": label,
        }
