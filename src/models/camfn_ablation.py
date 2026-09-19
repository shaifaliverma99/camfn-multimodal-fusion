"""Ablation baseline for the joint experiment: identical encoders, learned
placeholders, and heads to CAMFN, but fusion is a fixed masked mean over
present-modality embeddings instead of pairwise cross-attention +
ModalityTokenTransformer + DynamicGatedFusion.

Comparing this against `CAMFN` isolates exactly one thing: does the
learned cross-attention/gating fusion mechanism outperform the simplest
possible way to combine modality embeddings? See paper §Results
(fusion ablation) and docs/RESULTS.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.multimodal_dataset import MODALITIES
from .encoders.eeg_encoder import EEGEncoder
from .encoders.image_encoder import ImageEncoder2D
from .encoders.tabular_encoder import TabularTemporalEncoder
from .heads.classification_head import DiagnosisHead
from .heads.severity_head import SeverityHead


class NaiveMeanFusionCAMFN(nn.Module):
    """Same inputs/outputs as `CAMFN.forward`, but fusion = masked mean."""

    def __init__(
        self,
        d_model: int = 128,
        n_diagnosis_classes: int = 5,
        eeg_in_channels: int = 1,
        acoustic_in_features: int = 22,
        n_heads: int = 4,  # accepted for interface parity with CAMFN; unused here
        n_fusion_layers: int = 2,  # ditto
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mri_encoder = ImageEncoder2D(d_model=d_model)
        self.motor_encoder = ImageEncoder2D(d_model=d_model)
        self.eeg_encoder = EEGEncoder(in_channels=eeg_in_channels, d_model=d_model)
        self.acoustic_encoder = TabularTemporalEncoder(in_features=acoustic_in_features, d_model=d_model, n_heads=n_heads)

        self.placeholders = nn.ParameterDict({m: nn.Parameter(torch.zeros(d_model)) for m in MODALITIES})
        for p in self.placeholders.values():
            nn.init.normal_(p, std=0.02)

        self.diagnosis_head = DiagnosisHead(d_model, n_diagnosis_classes, dropout)
        self.severity_head = SeverityHead(d_model)
        self.eps = 1e-6

    def _apply_placeholder(self, pooled: torch.Tensor, presence: torch.Tensor, modality: str) -> torch.Tensor:
        placeholder = self.placeholders[modality].unsqueeze(0).expand_as(pooled)
        presence = presence.unsqueeze(-1)
        return presence * pooled + (1 - presence) * placeholder

    def forward(self, batch: dict) -> dict:
        presence_mask = batch["presence_mask"]  # [B, len(MODALITIES)]
        B = batch["mri"].shape[0]

        pooled_mri, _ = self.mri_encoder(batch["mri"])
        pooled_motor, _ = self.motor_encoder(batch["motor"])
        pooled_acoustic, _ = self.acoustic_encoder(batch["acoustic"])
        if batch.get("eeg") is not None:
            pooled_eeg, _ = self.eeg_encoder(batch["eeg"])
        else:
            pooled_eeg = self.placeholders["eeg"].unsqueeze(0).expand(B, -1)

        idx = {m: MODALITIES.index(m) for m in MODALITIES}
        pooled_mri = self._apply_placeholder(pooled_mri, presence_mask[:, idx["mri"]], "mri")
        pooled_eeg = self._apply_placeholder(pooled_eeg, presence_mask[:, idx["eeg"]], "eeg")
        pooled_motor = self._apply_placeholder(pooled_motor, presence_mask[:, idx["motor"]], "motor")
        pooled_acoustic = self._apply_placeholder(pooled_acoustic, presence_mask[:, idx["acoustic"]], "acoustic")

        stacked = torch.stack([pooled_mri, pooled_eeg, pooled_motor, pooled_acoustic], dim=1)  # [B,4,d]
        mask = presence_mask.unsqueeze(-1)  # [B,4,1]
        fused = (mask * stacked).sum(dim=1) / (mask.sum(dim=1) + self.eps)  # masked mean, no learned gate

        diagnosis_logits = self.diagnosis_head(fused)
        severity = self.severity_head(fused)
        return {
            "diagnosis_logits": diagnosis_logits,
            "severity": severity,
            "fusion_gates": presence_mask,  # uniform (unlearned) "gates" = presence itself, for interface parity
            "fused_embedding": fused,
        }
