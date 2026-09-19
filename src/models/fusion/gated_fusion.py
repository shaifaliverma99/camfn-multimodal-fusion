"""Dynamic Gated Fusion Unit — adaptively weighs modality contributions,
robust to missing/noisy streams. See docs/ARCHITECTURE.md section 3.3.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DynamicGatedFusion(nn.Module):
    """tokens: [B, M, d_model], presence_mask: [B, M] (1=present, 0=absent).

    g_m = sigmoid(MLP([t_m ; mask_m]))
    fused = sum_m (mask_m * g_m * t_m) / (sum_m mask_m * g_m + eps)
    """

    def __init__(self, d_model: int = 128, hidden: int = 64, eps: float = 1e-6):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model + 1, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
        )
        self.eps = eps

    def forward(self, tokens: torch.Tensor, presence_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, M, D = tokens.shape
        mask_feat = presence_mask.unsqueeze(-1)                  # [B, M, 1]
        gate_input = torch.cat([tokens, mask_feat], dim=-1)      # [B, M, D+1]
        gates = torch.sigmoid(self.gate_mlp(gate_input)).squeeze(-1)  # [B, M]
        weighted = presence_mask * gates                          # [B, M]
        fused = (weighted.unsqueeze(-1) * tokens).sum(dim=1)       # [B, D]
        denom = weighted.sum(dim=1, keepdim=True) + self.eps       # [B, 1]
        fused = fused / denom
        return fused, gates
