#!/usr/bin/env python3
"""End-to-end shape/robustness check for CAMFN.

This does NOT train or report any accuracy — it only confirms tensors flow
through every encoder, the fusion stack, and both heads with correct
shapes, including when modalities are missing. Run:

    python camfn_multimodal_system/scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.data.multimodal_dataset import MODALITIES  # noqa: E402
from src.models.camfn import CAMFN  # noqa: E402


def make_dummy_batch(batch_size: int, image_size: int, eeg_channels: int, eeg_len: int, acoustic_dim: int, presence: list[float]) -> dict:
    presence_mask = torch.tensor([presence] * batch_size, dtype=torch.float32)
    return {
        "mri": torch.randn(batch_size, 1, image_size, image_size),
        "motor": torch.randn(batch_size, 1, image_size, image_size),
        "acoustic": torch.randn(batch_size, acoustic_dim),
        "eeg": torch.randn(batch_size, eeg_channels, eeg_len),
        "presence_mask": presence_mask,
        "diagnosis_label": torch.randint(0, 5, (batch_size,)),
        "diagnosis_label_valid": torch.ones(batch_size, dtype=torch.bool),
    }


def main() -> None:
    torch.manual_seed(0)
    model = CAMFN(d_model=64, n_heads=4, n_fusion_layers=2, eeg_in_channels=1, acoustic_in_features=22)
    model.eval()

    print("MODALITIES order:", MODALITIES)

    # Case 1: all modalities present
    batch = make_dummy_batch(4, image_size=64, eeg_channels=1, eeg_len=178, acoustic_dim=22, presence=[1, 1, 1, 1])
    with torch.no_grad():
        out = model(batch)
    assert out["diagnosis_logits"].shape == (4, 5), out["diagnosis_logits"].shape
    assert out["severity"]["cdr_like"].shape == (4,)
    assert out["fusion_gates"].shape == (4, 4)
    print("[OK] all-modalities-present batch:", {k: v.shape for k, v in out["severity"].items()})

    # Case 2: only MRI present (EEG batch tensor still provided but presence=0
    # for eeg/motor/acoustic — mimics a single-cohort batch from
    # MultimodalNeuroDataset)
    batch2 = make_dummy_batch(4, image_size=64, eeg_channels=1, eeg_len=178, acoustic_dim=22, presence=[1, 0, 0, 0])
    with torch.no_grad():
        out2 = model(batch2)
    assert out2["diagnosis_logits"].shape == (4, 5)
    print("[OK] MRI-only batch, gates:", out2["fusion_gates"].mean(dim=0).tolist())

    # Case 3: no EEG tensor supplied at all (batch["eeg"] is None), as
    # produced by MultimodalNeuroDataset.collate_fn when no sample in the
    # batch carries EEG.
    batch3 = make_dummy_batch(4, image_size=64, eeg_channels=1, eeg_len=178, acoustic_dim=22, presence=[1, 0, 1, 1])
    batch3["eeg"] = None
    with torch.no_grad():
        out3 = model(batch3)
    assert out3["diagnosis_logits"].shape == (4, 5)
    print("[OK] eeg=None batch (placeholder path) works")

    # Case 4: gradient check — make sure backward doesn't crash through the
    # placeholder/gating path.
    model.train()
    batch4 = make_dummy_batch(2, image_size=64, eeg_channels=1, eeg_len=178, acoustic_dim=22, presence=[1, 0, 0, 1])
    out4 = model(batch4)
    loss = out4["diagnosis_logits"].sum() + sum(v.sum() for v in out4["severity"].values())
    loss.backward()
    n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"[OK] backward pass succeeded, {n_params_with_grad} parameter tensors received gradients")

    print("\nAll smoke tests passed. This confirms shape correctness only — no accuracy claims.")


if __name__ == "__main__":
    main()
