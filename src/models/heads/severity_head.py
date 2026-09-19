"""Sub-type / severity progression regression heads.

Three independent scalar regressors, one per disease's severity proxy:
CDR-like (AD), UPDRS-III-like (PD), Seizure-Frequency-Index-like (Epilepsy).
Each is trained only against samples where that label is actually
meaningful (see src/training/losses.py::MultiTaskLoss) — this head just
always emits all three; masking happens in the loss, not here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SeverityHead(nn.Module):
    def __init__(self, d_model: int = 128, hidden: int = 64, dropout: float = 0.1):
        super().__init__()

        def _regressor():
            return nn.Sequential(
                nn.Linear(d_model, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 1)
            )

        self.cdr_like = _regressor()
        self.updrs_like = _regressor()
        self.seizure_freq_index_like = _regressor()

    def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "cdr_like": self.cdr_like(fused).squeeze(-1),
            "updrs_like": self.updrs_like(fused).squeeze(-1),
            "seizure_freq_index_like": self.seizure_freq_index_like(fused).squeeze(-1),
        }
