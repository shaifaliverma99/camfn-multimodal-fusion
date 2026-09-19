"""Primary diagnostic classification head."""
from __future__ import annotations

import torch
import torch.nn as nn


class DiagnosisHead(nn.Module):
    """[B, d_model] -> [B, n_classes] logits over
    {HC, AD, PD, Epilepsy, Comorbid} (see multimodal_dataset.DIAGNOSIS_CLASSES).
    """

    def __init__(self, d_model: int = 128, n_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_model, n_classes)
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)
