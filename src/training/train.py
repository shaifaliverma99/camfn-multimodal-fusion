"""Skeleton training loop.

Builds the four real, open cohorts described in docs/DATA_SOURCES.md,
wraps them into a MultimodalNeuroDataset, and runs a few epochs of CAMFN to
confirm the whole pipeline is wired correctly end to end. This is NOT a
tuned, converged model — no accuracy numbers from a run of this script
should be quoted anywhere as representative system performance without a
real training campaign (more epochs, proper splits, hyperparameter search).

Usage:
    python -m src.training.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from ..data.acoustic_dataset import ParkinsonVoiceDataset
from ..data.eeg_dataset import EpilepticSeizureCSVDataset
from ..data.mri_dataset import OASIS1SliceDataset
from ..data.motor_dataset import ParkinsonDrawingDataset
from ..data.multimodal_dataset import (
    CohortWrapperDataset,
    MultimodalNeuroDataset,
    _alzheimer_label,
    _epilepsy_label,
    _pd_motor_label,
    _pd_voice_label,
    collate_fn,
)
from ..models.camfn import CAMFN
from ..utils.seed import set_seed
from .losses import MultiTaskLoss
from .metrics import compute_classification_metrics


def build_dataset(cfg: dict) -> MultimodalNeuroDataset:
    root = Path(cfg["data"]["root"])
    image_size = cfg["data"]["image_size"]
    fixed_shapes = {
        "mri": (1, image_size, image_size),
        "motor": (1, image_size, image_size),
        "acoustic": (22,),
    }

    alz = OASIS1SliceDataset(root / cfg["data"]["alzheimer_dir"], image_size=image_size)
    motor = ParkinsonDrawingDataset(root / cfg["data"]["parkinson_drawings_dir"], task="spiral", split="training", image_size=image_size)
    voice = ParkinsonVoiceDataset(root / cfg["data"]["parkinson_voice_file"])
    eeg = EpilepticSeizureCSVDataset(root / cfg["data"]["epilepsy_csv"])

    cohorts = [
        CohortWrapperDataset(alz, "mri", _alzheimer_label, "oasis1_alzheimer", fixed_shapes),
        CohortWrapperDataset(motor, "motor", _pd_motor_label, "parkinson_drawings", fixed_shapes),
        CohortWrapperDataset(voice, "acoustic", _pd_voice_label, "uci_parkinson_voice", fixed_shapes),
        CohortWrapperDataset(eeg, "eeg", _epilepsy_label, "epileptic_seizure_recognition", fixed_shapes),
    ]
    return MultimodalNeuroDataset(cohorts)


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(cfg)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg["seed"])
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg["training"]["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=cfg["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["training"]["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=cfg["training"]["num_workers"],
    )

    model = CAMFN(
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_fusion_layers=cfg["model"]["n_fusion_layers"],
        dropout=cfg["model"]["dropout"],
        n_diagnosis_classes=cfg["model"]["n_diagnosis_classes"],
        eeg_in_channels=1,       # Epileptic-Seizure-Recognition rows are single-channel
        acoustic_in_features=22,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    loss_fn = MultiTaskLoss()

    def _to_device(batch: dict) -> dict:
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = _to_device(batch)
            outputs = model(batch)
            losses = loss_fn(outputs, batch)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            running_loss += losses["total"].item()
        print(f"epoch {epoch}: train_loss={running_loss / max(1, len(train_loader)):.4f}")

        model.eval()
        all_true, all_pred, all_proba = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch)
                outputs = model(batch)
                valid = batch["diagnosis_label_valid"].cpu().numpy()
                if not valid.any():
                    continue
                proba = torch.softmax(outputs["diagnosis_logits"], dim=-1).cpu().numpy()[valid]
                pred = proba.argmax(axis=-1)
                true = batch["diagnosis_label"].cpu().numpy()[valid]
                all_true.append(true)
                all_pred.append(pred)
                all_proba.append(proba)
        if all_true:
            metrics = compute_classification_metrics(
                np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_proba),
                n_classes=cfg["model"]["n_diagnosis_classes"],
            )
            print(f"epoch {epoch}: val_metrics={metrics}")
        else:
            print(f"epoch {epoch}: no valid-labeled validation samples this split")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(args.config)
