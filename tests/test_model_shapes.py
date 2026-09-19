"""Pytest wrapper around scripts/smoke_test.py's assertions, plus a couple
of targeted unit tests for the fusion primitives.

Run: `pytest camfn_multimodal_system/tests` (from /workspace) or
`pytest tests` (from camfn_multimodal_system/).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_test import main as run_smoke_test  # noqa: E402
from src.models.fusion.gated_fusion import DynamicGatedFusion  # noqa: E402


def test_smoke_test_script_runs():
    run_smoke_test()  # raises AssertionError internally if any shape is wrong


def test_gated_fusion_ignores_absent_modalities():
    torch.manual_seed(0)
    fusion = DynamicGatedFusion(d_model=8)
    tokens = torch.randn(2, 3, 8)
    # Modality 1 absent for both samples -> changing its token value must
    # not change the fused output at all (it's fully masked out).
    presence = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    fused_a, _ = fusion(tokens, presence)
    tokens_perturbed = tokens.clone()
    tokens_perturbed[:, 1, :] += 100.0  # perturb the absent modality's token
    fused_b, _ = fusion(tokens_perturbed, presence)
    assert torch.allclose(fused_a, fused_b, atol=1e-5)


def test_gated_fusion_all_absent_is_finite():
    fusion = DynamicGatedFusion(d_model=8)
    tokens = torch.randn(1, 3, 8)
    presence = torch.zeros(1, 3)
    fused, gates = fusion(tokens, presence)
    assert torch.isfinite(fused).all()
