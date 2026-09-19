"""EEG encoder: 1D-CNN + BiLSTM + temporal attention pooling.

Input: [B, C, T] (C = channel count, fixed per instantiation — CHB-MIT and
the Epileptic-Seizure-Recognition dataset have different channel counts, so
a given `EEGEncoder` is built for one or the other, not both; see
docs/DATA_SOURCES.md and multimodal_dataset.py's collate_fn note).
Output: pooled `[B, d_model]` and the per-timestep token sequence
`[B, T', d_model]` for cross-attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttentionPool(nn.Module):
    """Additive attention pooling over the time axis.

    score_t = w^T tanh(W h_t);  alpha = softmax_t(score_t);  out = sum_t alpha_t h_t
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, d_model]
        scores = self.score(torch.tanh(self.proj(h)))  # [B, T, 1]
        alpha = torch.softmax(scores, dim=1)
        return (alpha * h).sum(dim=1)  # [B, d_model]


class EEGEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 128, cnn_channels: int = 32, lstm_hidden: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(lstm_hidden * 2, d_model)
        self.attn_pool = TemporalAttentionPool(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, T]
        feats = self.cnn(x)                    # [B, cnn_channels*2, T]
        feats = feats.transpose(1, 2)           # [B, T, cnn_channels*2]
        lstm_out, _ = self.lstm(feats)          # [B, T, lstm_hidden*2]
        tokens = self.proj(lstm_out)            # [B, T, d_model]
        pooled = self.attn_pool(tokens)         # [B, d_model]
        return pooled, tokens
