"""Transformer encoder for tabular / short temporal feature inputs.

Used for the UCI Parkinson's voice features (`[B, F]`). Also accepts a true
sequence `[B, L, F]` if a future dataset supplies one (e.g. per-frame
acoustic features rather than one summary vector per recording).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TabularTemporalEncoder(nn.Module):
    def __init__(self, in_features: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, F] -> [B, 1, F] (single-token "sequence")
        tokens = self.input_proj(x)  # [B, L, d_model]
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)  # [B, 1+L, d_model]
        out = self.encoder(seq)
        pooled = out[:, 0]      # CLS output
        rest = out[:, 1:]       # per-input tokens, for cross-attention use
        return pooled, rest
