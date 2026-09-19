"""Multi-task loss: cross-entropy for diagnosis + masked MSE for severity.

Severity targets are only meaningful for certain cohorts (CDR-like for
Alzheimer samples, UPDRS-like for Parkinson's, seizure-frequency-index-like
for epilepsy). This skeleton's datasets don't all carry a real continuous
severity target yet (only OASIS-1 supplies `cdr_like`, derived from its
4-way severity bucket, not a true clinical CDR sum-of-boxes) — the masks
below are wired so training only backprops severity loss where a target is
actually supplied in the batch, rather than pretending 0.0 is a real label
for cohorts that don't have one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, severity_weight: float = 0.5):
        super().__init__()
        self.severity_weight = severity_weight

    def forward(self, outputs: dict, batch: dict) -> dict:
        valid = batch["diagnosis_label_valid"]
        if valid.any():
            diag_loss = F.cross_entropy(
                outputs["diagnosis_logits"][valid], batch["diagnosis_label"][valid]
            )
        else:
            diag_loss = torch.zeros((), device=outputs["diagnosis_logits"].device)

        severity_loss = torch.zeros((), device=outputs["diagnosis_logits"].device)
        n_terms = 0
        for key in ("cdr_like",):  # only target actually present across current datasets
            valid_key = f"{key}_valid"
            if key in batch and valid_key in batch and batch[valid_key].any():
                mask = batch[valid_key]
                target = batch[key][mask]
                pred = outputs["severity"][key][mask]
                severity_loss = severity_loss + F.mse_loss(pred, target)
                n_terms += 1
        if n_terms > 0:
            severity_loss = severity_loss / n_terms

        total = diag_loss + self.severity_weight * severity_loss
        return {"total": total, "diagnosis_loss": diag_loss, "severity_loss": severity_loss}
