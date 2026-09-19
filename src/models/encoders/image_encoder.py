"""2D and 3D image encoders.

`ImageEncoder2D` is what's actually exercised in this skeleton (OASIS-1 MRI
slices, Parkinson drawings — both 2D grayscale images). `Volumetric3DEncoder`
is an architecturally-correct stub for true volumetric NIfTI data
(ADNI/OASIS-3-style 3D-ResNet), included per the spec but untested here
since no volumetric data is present in this repo — see docs/DATA_SOURCES.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ImageEncoder2D(nn.Module):
    """4-block CNN -> global average pool -> linear projection to d_model.

    Input: [B, 1, H, W]. Output: pooled `[B, d_model]` and a token grid
    `[B, H'*W', d_model]` (flattened spatial map before pooling) for use as
    the query/key/value sequence in `CrossAttentionLayer`.
    """

    def __init__(self, d_model: int = 128, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c * 4, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.conv(x)                      # [B, d_model, H', W']
        pooled = self.pool(feat_map).flatten(1)       # [B, d_model]
        tokens = feat_map.flatten(2).transpose(1, 2)  # [B, H'*W', d_model]
        return pooled, tokens


class Volumetric3DEncoder(nn.Module):
    """3D-conv analogue of `ImageEncoder2D` for real NIfTI volumes.

    Not exercised by any dataset in this repo. Kept as the documented
    extension point for real ADNI/OASIS-3 3D T1/PET volumes.
    """

    def __init__(self, d_model: int = 128, in_channels: int = 1, base_channels: int = 8):
        super().__init__()
        c = base_channels
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(c, c * 2, 3, padding=1), nn.BatchNorm3d(c * 2), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(c * 2, d_model, 3, padding=1), nn.BatchNorm3d(d_model), nn.ReLU(inplace=True), nn.MaxPool3d(2),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.conv(x)                       # [B, d_model, D', H', W']
        pooled = self.pool(feat_map).flatten(1)
        tokens = feat_map.flatten(2).transpose(1, 2)   # [B, D'*H'*W', d_model]
        return pooled, tokens
