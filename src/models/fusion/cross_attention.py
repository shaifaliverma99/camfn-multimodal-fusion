"""Cross-modal attention: pairwise (spatial-vs-temporal) and modality-token
(all-present-modalities) fusion. See docs/ARCHITECTURE.md section 3.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """Q from `query_seq`, K/V from `kv_seq`. Standard post-LN transformer
    block: multi-head cross-attention -> residual -> FFN -> residual.

    query_seq: [B, Lq, d_model]; kv_seq: [B, Lkv, d_model].
    Returns: [B, Lq, d_model].
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(inplace=True), nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_seq: torch.Tensor, kv_seq: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query_seq, kv_seq, kv_seq)
        x = self.norm1(query_seq + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class ModalityTokenTransformer(nn.Module):
    """Self-attention over a sequence of one pooled token per modality.

    Works with a variable number of present modalities per batch (the
    caller supplies exactly the tokens that should participate).
    tokens: [B, M, d_model] -> returns [B, M, d_model].
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)
