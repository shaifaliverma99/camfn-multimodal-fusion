"""CAMFN — Cross-Attention Multimodal Fusion Network.

Ties together the four modality encoders, pairwise + modality-token
cross-attention, dynamic gated fusion, and the diagnosis/severity heads.
See docs/ARCHITECTURE.md for the full data-flow diagram and equations.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.multimodal_dataset import MODALITIES
from .encoders.eeg_encoder import EEGEncoder
from .encoders.image_encoder import ImageEncoder2D
from .encoders.tabular_encoder import TabularTemporalEncoder
from .fusion.cross_attention import CrossAttentionLayer, ModalityTokenTransformer
from .fusion.gated_fusion import DynamicGatedFusion
from .heads.classification_head import DiagnosisHead
from .heads.severity_head import SeverityHead


class CAMFN(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_fusion_layers: int = 2,
        dropout: float = 0.1,
        n_diagnosis_classes: int = 5,
        eeg_in_channels: int = 1,
        acoustic_in_features: int = 22,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Modality encoders ---
        self.mri_encoder = ImageEncoder2D(d_model=d_model)
        self.motor_encoder = ImageEncoder2D(d_model=d_model)
        self.eeg_encoder = EEGEncoder(in_channels=eeg_in_channels, d_model=d_model)
        self.acoustic_encoder = TabularTemporalEncoder(in_features=acoustic_in_features, d_model=d_model, n_heads=n_heads)

        # Learned placeholders substituted in for absent modalities before
        # fusion (see docs/ARCHITECTURE.md section 5).
        self.placeholders = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(d_model)) for m in MODALITIES}
        )
        for p in self.placeholders.values():
            nn.init.normal_(p, std=0.02)

        # --- Fusion engine ---
        self.cross_mri_eeg = CrossAttentionLayer(d_model, n_heads, dropout)
        self.cross_eeg_mri = CrossAttentionLayer(d_model, n_heads, dropout)
        self.modality_token_transformer = ModalityTokenTransformer(d_model, n_heads, n_fusion_layers, dropout)
        self.gated_fusion = DynamicGatedFusion(d_model)

        # --- Heads ---
        self.diagnosis_head = DiagnosisHead(d_model, n_diagnosis_classes, dropout)
        self.severity_head = SeverityHead(d_model)

    def _apply_placeholder(self, pooled: torch.Tensor, presence: torch.Tensor, modality: str) -> torch.Tensor:
        # pooled: [B, d_model], presence: [B] (1=present, 0=absent)
        placeholder = self.placeholders[modality].unsqueeze(0).expand_as(pooled)
        presence = presence.unsqueeze(-1)
        return presence * pooled + (1 - presence) * placeholder

    def forward(self, batch: dict) -> dict:
        B = batch["mri"].shape[0]
        device = batch["mri"].device
        presence_mask = batch["presence_mask"]  # [B, len(MODALITIES)], columns match MODALITIES order

        pooled_mri, tokens_mri = self.mri_encoder(batch["mri"])
        pooled_motor, tokens_motor = self.motor_encoder(batch["motor"])
        pooled_acoustic, tokens_acoustic = self.acoustic_encoder(batch["acoustic"])

        has_eeg_in_batch = batch.get("eeg") is not None
        if has_eeg_in_batch:
            pooled_eeg, tokens_eeg = self.eeg_encoder(batch["eeg"])
            # Pairwise spatial-vs-temporal cross-attention (spec section 3):
            # MRI spatial tokens query EEG temporal tokens, and vice versa.
            refined_mri_tokens = self.cross_mri_eeg(tokens_mri, tokens_eeg)
            refined_eeg_tokens = self.cross_eeg_mri(tokens_eeg, tokens_mri)
            pooled_mri = pooled_mri + refined_mri_tokens.mean(dim=1)
            pooled_eeg = pooled_eeg + refined_eeg_tokens.mean(dim=1)
        else:
            pooled_eeg = self.placeholders["eeg"].unsqueeze(0).expand(B, -1)

        eeg_idx = MODALITIES.index("eeg")
        mri_idx = MODALITIES.index("mri")
        motor_idx = MODALITIES.index("motor")
        acoustic_idx = MODALITIES.index("acoustic")

        pooled_mri = self._apply_placeholder(pooled_mri, presence_mask[:, mri_idx], "mri")
        pooled_eeg = self._apply_placeholder(pooled_eeg, presence_mask[:, eeg_idx], "eeg")
        pooled_motor = self._apply_placeholder(pooled_motor, presence_mask[:, motor_idx], "motor")
        pooled_acoustic = self._apply_placeholder(pooled_acoustic, presence_mask[:, acoustic_idx], "acoustic")

        # Order must match MODALITIES so presence_mask columns line up.
        stacked = torch.stack([pooled_mri, pooled_eeg, pooled_motor, pooled_acoustic], dim=1)  # [B, 4, d]
        refined = self.modality_token_transformer(stacked)  # [B, 4, d]
        fused, gates = self.gated_fusion(refined, presence_mask)  # [B, d], [B, 4]

        diagnosis_logits = self.diagnosis_head(fused)
        severity = self.severity_head(fused)

        return {
            "diagnosis_logits": diagnosis_logits,
            "severity": severity,
            "fusion_gates": gates,  # [B, 4] — inspect for explainability / debugging
            "fused_embedding": fused,
        }
