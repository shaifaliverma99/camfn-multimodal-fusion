"""Real training + evaluation runs against the real, open datasets in this
repo. Produces genuine metrics — nothing here is simulated or invented.

Two kinds of experiment:
1. Per-modality baselines: each modality's encoder + a small classifier
   head, trained and evaluated on its own real cohort with a proper
   held-out test split (subject-grouped for the voice dataset to avoid
   leakage across a subject's repeated recordings).
2. The full CAMFN joint experiment: all four cohorts combined via
   MultimodalNeuroDataset, trained/evaluated with the dynamic gated fusion
   path active. Framed honestly (see docs/DATA_SOURCES.md): this tests the
   architecture's missing-modality robustness across cohorts, not true
   per-patient multimodal fusion benefit, since no dataset here has more
   than one real modality per subject.

Run:
    python experiments/run_real_experiments.py
Outputs:
    experiments/results/real_results.csv
    experiments/results/*.log (per-experiment console log, also printed)
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.acoustic_dataset import ParkinsonVoiceDataset  # noqa: E402
from src.data.eeg_dataset import EpilepticSeizureCSVDataset  # noqa: E402
from src.data.mri_dataset import OASIS1SliceDataset  # noqa: E402
from src.data.motor_dataset import ParkinsonDrawingDataset  # noqa: E402
from src.data.multimodal_dataset import (  # noqa: E402
    CohortWrapperDataset,
    MultimodalNeuroDataset,
    _alzheimer_label,
    _epilepsy_label,
    _pd_motor_label,
    _pd_voice_label,
    collate_fn,
)
from src.models.camfn import CAMFN  # noqa: E402
from src.models.encoders.eeg_encoder import EEGEncoder  # noqa: E402
from src.models.encoders.image_encoder import ImageEncoder2D  # noqa: E402
from src.models.encoders.tabular_encoder import TabularTemporalEncoder  # noqa: E402
from src.training.metrics import compute_classification_metrics  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
PRED_DIR = Path(__file__).resolve().parent / "results" / "predictions"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


class SingleModalityClassifier(nn.Module):
    """encoder -> pooled embedding -> linear classifier. Used for every
    per-modality baseline below (same head shape, different encoder).
    """

    def __init__(self, encoder: nn.Module, d_model: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.encoder(x)
        return self.head(pooled)


def run_training(
    model: nn.Module,
    train_items: list,
    val_items: list,
    test_items: list,
    x_key: str,
    y_key: str,
    epochs: int,
    batch_size: int,
    lr: float,
    n_classes: int,
    log,
    checkpoint_path: Path | None = None,
    predictions_path: Path | None = None,
):
    def make_loader(items, shuffle):
        xs = torch.stack([it[x_key] for it in items])
        ys = torch.tensor([it[y_key] for it in items], dtype=torch.long)
        ds = torch.utils.data.TensorDataset(xs, ys)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_items, True)
    val_loader = make_loader(val_items, False)
    test_loader = make_loader(test_items, False)

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        model.eval()
        with torch.no_grad():
            val_true, val_pred = [], []
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                pred = model(xb).argmax(dim=-1).cpu().numpy()
                val_pred.append(pred)
                val_true.append(yb.numpy())
            val_acc = float((np.concatenate(val_true) == np.concatenate(val_pred)).mean())
        log(f"  epoch {epoch}: train_loss={running / len(train_loader):.4f} val_acc={val_acc:.4f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        log(f"  saved checkpoint -> {checkpoint_path}")

    model.eval()
    with torch.no_grad():
        test_true, test_pred, test_proba = [], [], []
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            test_proba.append(proba)
            test_pred.append(proba.argmax(axis=-1))
            test_true.append(yb.numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_proba = np.concatenate(test_proba)

    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(predictions_path, y_true=test_true, y_pred=test_pred, y_proba=test_proba)

    metrics = compute_classification_metrics(test_true, test_pred, test_proba, n_classes=n_classes)
    metrics["best_val_acc"] = best_val_acc
    metrics["n_train"] = len(train_items)
    metrics["n_val"] = len(val_items)
    metrics["n_test"] = len(test_items)
    return metrics


def run_kfold_training(
    items: list,
    labels: list,
    groups: list | None,
    x_key: str,
    y_key: str,
    model_ctor,
    epochs: int,
    batch_size: int,
    lr: float,
    n_classes: int,
    log,
    n_splits: int = 5,
    checkpoint_path: Path | None = None,
    predictions_path: Path | None = None,
):
    """K-fold cross-validation over the *entire* pool of items (train+test
    combined), so every real sample gets evaluated exactly once as a
    held-out test point instead of just the ~30 samples a single fixed
    split would leave out. Far more defensible for small (n~100-200)
    datasets than one lucky/unlucky split.

    Uses GroupKFold (by subject) when `groups` is given (voice), else
    StratifiedKFold. The checkpoint saved (if any) is from fold 0 only --
    a single deployable model for the dashboard, not the CV estimate.
    """
    idx = np.arange(len(items))
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(idx, labels, groups=groups))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = list(splitter.split(idx, labels))

    all_true, all_pred, all_proba = [], [], []
    fold_accs = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        fold_train = [items[i] for i in train_idx]
        fold_test = [items[i] for i in test_idx]
        fold_train_labels = [labels[i] for i in train_idx]
        if groups is not None:
            # Keep the inner train/val carve-out subject-grouped too, so a
            # person's samples can't split across train and val within a
            # fold (val only affects checkpoint selection, not the
            # reported test metric, but there's no reason to leak it).
            fold_train_groups = [groups[i] for i in train_idx]
            inner_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
            tr_rel, val_rel = next(inner_gss.split(np.arange(len(fold_train)), fold_train_labels, groups=fold_train_groups))
            fold_train_items = [fold_train[i] for i in tr_rel]
            fold_val_items = [fold_train[i] for i in val_rel]
        else:
            fold_train_items, fold_val_items = train_test_split(
                fold_train, test_size=0.2, stratify=fold_train_labels, random_state=SEED
            )

        model = model_ctor()
        log(f"  -- fold {fold}: n_train={len(fold_train_items)} n_val={len(fold_val_items)} n_test={len(fold_test)}")
        fold_metrics = run_training(
            model, fold_train_items, fold_val_items, fold_test, x_key=x_key, y_key=y_key,
            epochs=epochs, batch_size=batch_size, lr=lr, n_classes=n_classes, log=log,
            checkpoint_path=checkpoint_path if (fold == 0 and checkpoint_path is not None) else None,
        )
        fold_accs.append(fold_metrics["accuracy"])

        # Recompute this fold's raw predictions so we can pool across all folds.
        model.eval()
        xs = torch.stack([it[x_key] for it in fold_test]).to(DEVICE)
        ys = np.array([it[y_key] for it in fold_test])
        with torch.no_grad():
            proba = torch.softmax(model(xs), dim=-1).cpu().numpy()
        all_true.append(ys)
        all_pred.append(proba.argmax(axis=-1))
        all_proba.append(proba)

    pooled_true = np.concatenate(all_true)
    pooled_pred = np.concatenate(all_pred)
    pooled_proba = np.concatenate(all_proba)
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(predictions_path, y_true=pooled_true, y_pred=pooled_pred, y_proba=pooled_proba)
    metrics = compute_classification_metrics(pooled_true, pooled_pred, pooled_proba, n_classes=n_classes)
    metrics["n_folds"] = n_splits
    metrics["n_test_total"] = len(pooled_true)
    metrics["per_fold_accuracy"] = fold_accs
    return metrics


def experiment_alzheimer(log, results: list, save_ckpt: bool = True, save_predictions: bool = False):
    log("\n=== Experiment: Alzheimer (OASIS-1 MRI slices), HC vs AD, subject-grouped split ===")
    ds = OASIS1SliceDataset(REPO_ROOT / "Datasets/Alzheimer/Data", image_size=64)
    items = [ds[i] for i in _sample_indices(len(ds), max_n=12000, seed=SEED)]
    labels = [0 if it["severity_class"] == 0 else 1 for it in items]
    for it, lab in zip(items, labels):
        it["y"] = lab
    groups = [it["subject_id"] for it in items]
    n_subjects = len(set(groups))
    log(f"  {len(items)} slices from {n_subjects} unique subjects (subject-grouped split -- see mri_dataset.py)")
    train_items, val_items, test_items = _grouped_train_val_test_split(items, labels, groups, seed=SEED)

    model = SingleModalityClassifier(ImageEncoder2D(d_model=64), d_model=64, n_classes=2)
    metrics = run_training(
        model, train_items, val_items, test_items, x_key="mri", y_key="y",
        epochs=6, batch_size=64, lr=3e-4, n_classes=2, log=log,
        checkpoint_path=(CKPT_DIR / "alzheimer_mri.pt") if save_ckpt else None,
        predictions_path=(PRED_DIR / "alzheimer_hc_vs_ad.npz") if save_predictions else None,
    )
    metrics["note"] = (
        f"subject-grouped split (GroupShuffleSplit) over {n_subjects} unique OASIS-1 subjects -- "
        "a prior version of this experiment split individual slices without subject grouping, which "
        "leaks the same patient's scans across train/test and inflates accuracy; see docs/RESULTS.md."
    )
    log(f"RESULT alzheimer_hc_vs_ad: {metrics}")
    results.append({"experiment": "alzheimer_hc_vs_ad", "modality": "mri", "dataset": "OASIS-1 slices", **_flatten(metrics)})


def experiment_pd_motor(log, results: list, save_ckpt: bool = True, save_predictions: bool = False):
    """5-fold *subject-grouped* CV over the combined train+test pool
    (n=102 per task). The dataset's own provided training/testing split
    is NOT subject-disjoint (same subject IDs, e.g. V01-V11, appear on
    both sides -- see motor_dataset.py), so besides pooling train+test
    into one CV loop (avoiding a single lucky/unlucky n=30 split), we
    also group folds by subject so no person's drawings cross a fold
    boundary.
    """
    for task in ("spiral", "wave"):
        log(f"\n=== Experiment: Parkinson's motor drawings ({task}), healthy vs PD, 5-fold subject-grouped CV ===")
        train_ds = ParkinsonDrawingDataset(REPO_ROOT / "Datasets/Parkinson", task=task, split="training", image_size=96)
        test_ds = ParkinsonDrawingDataset(REPO_ROOT / "Datasets/Parkinson", task=task, split="testing", image_size=96)
        items = [train_ds[i] for i in range(len(train_ds))] + [test_ds[i] for i in range(len(test_ds))]
        for it in items:
            it["y"] = it["pd_motor_label"]
        labels = [it["y"] for it in items]
        groups = [it["subject_key"] for it in items]
        n_subjects = len(set(groups))
        log(f"  {len(items)} drawings from {n_subjects} unique (class, subject) groups")

        metrics = run_kfold_training(
            items, labels, groups=groups, x_key="motor", y_key="y",
            model_ctor=lambda: SingleModalityClassifier(ImageEncoder2D(d_model=64), d_model=64, n_classes=2),
            epochs=25, batch_size=8, lr=1e-3, n_classes=2, log=log, n_splits=5,
            checkpoint_path=(CKPT_DIR / f"pd_motor_{task}.pt") if save_ckpt else None,
            predictions_path=(PRED_DIR / f"pd_motor_{task}.npz") if save_predictions else None,
        )
        metrics["note"] = (
            f"5-fold GroupKFold CV by subject over the full train+test pool ({n_subjects} groups, n={len(items)}); "
            "the dataset's own train/test split is not subject-disjoint (same IDs on both sides), which a prior "
            "version of this experiment did not correct for; checkpoint is fold-0's model only."
        )
        log(f"RESULT pd_motor_{task}: {metrics}")
        results.append({"experiment": f"pd_motor_{task}", "modality": "motor(drawing)", "dataset": f"Parkinson Drawings ({task})", **_flatten(metrics)})


def experiment_pd_voice(log, results: list, save_ckpt: bool = True, save_predictions: bool = False):
    """5-fold *subject-grouped* CV (GroupKFold) over all 195 recordings --
    every recording is evaluated exactly once, with a subject never split
    across train/test within a fold, avoiding both the leakage risk of a
    naive random split and the small-n fragility of one 30-sample split.
    """
    log("\n=== Experiment: Parkinson's voice (UCI dysphonia features), healthy vs PD, 5-fold subject-grouped CV ===")
    ds = ParkinsonVoiceDataset(REPO_ROOT / "Datasets/Parkinson_voice/parkinsons.data")
    items = [ds[i] for i in range(len(ds))]
    groups = _voice_subject_groups(REPO_ROOT / "Datasets/Parkinson_voice/parkinsons.data")
    labels = [it["pd_voice_label"] for it in items]
    for it, lab in zip(items, labels):
        it["y"] = lab

    metrics = run_kfold_training(
        items, labels, groups=groups, x_key="acoustic", y_key="y",
        model_ctor=lambda: SingleModalityClassifier(TabularTemporalEncoder(in_features=22, d_model=64), d_model=64, n_classes=2),
        epochs=40, batch_size=16, lr=1e-3, n_classes=2, log=log, n_splits=5,
        checkpoint_path=(CKPT_DIR / "pd_voice.pt") if save_ckpt else None,
        predictions_path=(PRED_DIR / "pd_voice.npz") if save_predictions else None,
    )
    metrics["note"] = "5-fold GroupKFold CV by subject (n=195, ~31 subjects); checkpoint is fold-0's model only"
    log(f"RESULT pd_voice: {metrics}")
    results.append({"experiment": "pd_voice", "modality": "acoustic", "dataset": "UCI Parkinson's voice", **_flatten(metrics)})


def experiment_epilepsy(log, results: list, save_ckpt: bool = True, save_predictions: bool = False):
    log("\n=== Experiment: Epilepsy (Epileptic Seizure Recognition), seizure vs non-seizure ===")
    ds = EpilepticSeizureCSVDataset(REPO_ROOT / "Datasets/Epilepsy/Epileptic Seizure Recognition.csv")
    items = [ds[i] for i in range(len(ds))]
    labels = [it["seizure_label"] for it in items]
    for it, lab in zip(items, labels):
        it["y"] = lab
    train_items, temp_items = train_test_split(items, test_size=0.3, stratify=labels, random_state=SEED)
    temp_labels = [it["y"] for it in temp_items]
    val_items, test_items = train_test_split(temp_items, test_size=0.5, stratify=temp_labels, random_state=SEED)

    model = SingleModalityClassifier(EEGEncoder(in_channels=1, d_model=64), d_model=64, n_classes=2)
    metrics = run_training(
        model, train_items, val_items, test_items, x_key="eeg", y_key="y",
        epochs=10, batch_size=64, lr=1e-3, n_classes=2, log=log,
        checkpoint_path=(CKPT_DIR / "epilepsy_eeg.pt") if save_ckpt else None,
        predictions_path=(PRED_DIR / "epilepsy_seizure_vs_not.npz") if save_predictions else None,
    )
    metrics["note"] = "rows are pre-shuffled by the dataset distributor; standard random split matches published practice on this corpus"
    log(f"RESULT epilepsy: {metrics}")
    results.append({"experiment": "epilepsy_seizure_vs_not", "modality": "eeg", "dataset": "Epileptic Seizure Recognition", **_flatten(metrics)})


def _build_joint_split(image_size: int = 64):
    """Builds the combined 4-cohort dataset and a subject-grouped 70/15/15
    split. Factored out so CAMFN and the fusion ablation train/test on the
    *identical* split -- a fair comparison isolating the fusion mechanism.

    Each cohort is split *independently* by its own subject/group key
    (MRI: OASIS-1 subject; motor: (class, subject) key; voice: subject),
    then the per-cohort train/val/test index sets are mapped to global
    indices and concatenated. A single pooled random split (the original
    version of this function) ignores subject grouping entirely and lets
    the same person's samples land in both train and test for three of
    the four cohorts -- see docs/RESULTS.md. The EEG cohort has no
    recoverable subject ID in this corpus, so it falls back to a plain
    stratified split (unavoidable with this specific dataset).
    """
    fixed_shapes = {"mri": (1, image_size, image_size), "motor": (1, image_size, image_size), "acoustic": (22,)}
    alz = OASIS1SliceDataset(REPO_ROOT / "Datasets/Alzheimer/Data", image_size=image_size)
    motor = ParkinsonDrawingDataset(REPO_ROOT / "Datasets/Parkinson", task="spiral", split="training", image_size=image_size)
    voice = ParkinsonVoiceDataset(REPO_ROOT / "Datasets/Parkinson_voice/parkinsons.data")
    eeg = EpilepticSeizureCSVDataset(REPO_ROOT / "Datasets/Epilepsy/Epileptic Seizure Recognition.csv")

    alz_idx = _sample_indices(len(alz), max_n=8000, seed=SEED)
    alz_subset = Subset(alz, alz_idx)

    cohorts = [
        CohortWrapperDataset(alz_subset, "mri", _alzheimer_label, "oasis1_alzheimer", fixed_shapes),
        CohortWrapperDataset(motor, "motor", _pd_motor_label, "parkinson_drawings", fixed_shapes),
        CohortWrapperDataset(voice, "acoustic", _pd_voice_label, "uci_parkinson_voice", fixed_shapes),
        CohortWrapperDataset(eeg, "eeg", _epilepsy_label, "epileptic_seizure_recognition", fixed_shapes),
    ]
    full = MultimodalNeuroDataset(cohorts)

    # (labels, groups) per cohort, read directly from each dataset's
    # lightweight index (not via __getitem__, to avoid loading/decoding
    # every image/signal twice just to compute a split).
    alz_labels = [0 if alz.samples[i][1] == 0 else 1 for i in alz_idx]
    alz_groups = [alz.samples[i][2] for i in alz_idx]

    motor_labels = [s[1] for s in motor.samples]
    motor_groups = [s[2] for s in motor.samples]

    voice_labels = [r[1] for r in voice.rows]
    voice_groups = _voice_subject_groups(REPO_ROOT / "Datasets/Parkinson_voice/parkinsons.data")

    eeg_labels = [1 if r[1] == 1 else 0 for r in eeg.rows]

    per_cohort = [
        (len(alz_idx), alz_labels, alz_groups),
        (len(motor.samples), motor_labels, motor_groups),
        (len(voice.rows), voice_labels, voice_groups),
        (len(eeg.rows), eeg_labels, None),
    ]

    train_parts, val_parts, test_parts = [], [], []
    for cohort_idx, (n, labels, groups) in enumerate(per_cohort):
        tr, va, te = _grouped_index_split(n, labels, groups, seed=SEED)
        offset = full._offsets[cohort_idx]
        train_parts.append(offset + tr)
        val_parts.append(offset + va)
        test_parts.append(offset + te)

    rng = np.random.RandomState(SEED)
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)  # DataLoader(shuffle=True) reshuffles every epoch anyway; just avoids cohort-ordered batches on the very first pass
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_loader = DataLoader(Subset(full, train_idx), batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(full, val_idx), batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(Subset(full, test_idx), batch_size=64, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, len(train_idx), len(val_idx), len(test_idx)


def _to_device(batch: dict) -> dict:
    return {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _eval_joint_loader(model, loader):
    model.eval()
    all_true, all_pred, all_proba = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch)
            out = model(batch)
            valid = batch["diagnosis_label_valid"].cpu().numpy()
            if not valid.any():
                continue
            proba = torch.softmax(out["diagnosis_logits"], dim=-1).cpu().numpy()[valid]
            all_proba.append(proba)
            all_pred.append(proba.argmax(axis=-1))
            all_true.append(batch["diagnosis_label"].cpu().numpy()[valid])
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_proba)


def _train_joint_model(model, train_loader, val_loader, epochs, log):
    from src.training.losses import MultiTaskLoss

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = MultiTaskLoss()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = _to_device(batch)
            out = model(batch)
            losses = loss_fn(out, batch)
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            running += losses["total"].item()
        val_true, val_pred, _ = _eval_joint_loader(model, val_loader)
        val_acc = float((val_true == val_pred).mean())
        log(f"  epoch {epoch}: train_loss={running / len(train_loader):.4f} val_acc={val_acc:.4f}")
    return model


def experiment_camfn_joint(log, results: list, save_ckpt: bool = True, save_predictions: bool = False):
    log("\n=== Experiment: CAMFN joint (all cohorts, dynamic gated fusion, missing-modality masking) ===")
    train_loader, val_loader, test_loader, n_train, n_val, n_test = _build_joint_split()

    model = CAMFN(d_model=64, n_heads=4, n_fusion_layers=2, eeg_in_channels=1, acoustic_in_features=22, n_diagnosis_classes=5)
    model = _train_joint_model(model, train_loader, val_loader, epochs=6, log=log)

    if save_ckpt:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CKPT_DIR / "camfn_joint.pt")
        log(f"  saved checkpoint -> {CKPT_DIR / 'camfn_joint.pt'}")

    test_true, test_pred, test_proba = _eval_joint_loader(model, test_loader)
    if save_predictions:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(PRED_DIR / "camfn_joint_5way.npz", y_true=test_true, y_pred=test_pred, y_proba=test_proba)
    metrics = compute_classification_metrics(test_true, test_pred, test_proba, n_classes=5)
    metrics["n_train"] = n_train
    metrics["n_val"] = n_val
    metrics["n_test"] = n_test
    metrics["note"] = (
        "5-way diagnosis across combined cohorts; each test sample carries exactly one real "
        "modality (others masked absent) -- this validates missing-modality-robust routing "
        "through one shared fusion+classification head, not true joint multimodal benefit "
        "(no dataset here has >1 modality per patient)."
    )
    log(f"RESULT camfn_joint_5way: {metrics}")
    results.append({"experiment": "camfn_joint_5way", "modality": "mri+eeg+motor+acoustic (masked)", "dataset": "combined (4 cohorts)", **_flatten(metrics)})


def experiment_fusion_ablation(log, results: list, save_ckpt: bool = False, save_predictions: bool = False):
    """Ablation: does CAMFN's cross-attention + dynamic gated fusion beat
    the simplest possible alternative (a fixed masked mean of the same
    encoders' embeddings, same placeholders, same heads)? Trained and
    evaluated on the *identical* split as experiment_camfn_joint (same
    seed -> `_build_joint_split()` is deterministic) so any difference is
    attributable to the fusion mechanism, not the data split.
    """
    log("\n=== Experiment: Fusion ablation -- NaiveMeanFusion vs CAMFN (identical split) ===")
    from src.models.camfn_ablation import NaiveMeanFusionCAMFN

    train_loader, val_loader, test_loader, n_train, n_val, n_test = _build_joint_split()

    model = NaiveMeanFusionCAMFN(d_model=64, n_heads=4, n_fusion_layers=2, eeg_in_channels=1, acoustic_in_features=22, n_diagnosis_classes=5)
    model = _train_joint_model(model, train_loader, val_loader, epochs=6, log=log)

    test_true, test_pred, test_proba = _eval_joint_loader(model, test_loader)
    if save_predictions:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(PRED_DIR / "fusion_ablation_naive_mean.npz", y_true=test_true, y_pred=test_pred, y_proba=test_proba)
    metrics = compute_classification_metrics(test_true, test_pred, test_proba, n_classes=5)
    metrics["n_train"] = n_train
    metrics["n_val"] = n_val
    metrics["n_test"] = n_test
    metrics["note"] = (
        "Ablation baseline: same encoders/placeholders/heads as CAMFN, fusion replaced by a "
        "fixed masked mean (no cross-attention, no learned gate). Compare directly against "
        "camfn_joint_5way from the same seed."
    )
    log(f"RESULT fusion_ablation_naive_mean: {metrics}")
    results.append({"experiment": "fusion_ablation_naive_mean", "modality": "mri+eeg+motor+acoustic (masked)", "dataset": "combined (4 cohorts)", **_flatten(metrics)})


def _sample_indices(n: int, max_n: int, seed: int) -> list[int]:
    if n <= max_n:
        return list(range(n))
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(n, size=max_n, replace=False).tolist())


def _grouped_index_split(
    n: int, labels: list, groups: list | None, seed: int,
    test_frac: float = 0.3, val_frac_of_temp: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits range(n) into (train, val, test) index arrays such that no
    `groups` value (e.g. subject ID) appears in more than one split.
    Falls back to a plain stratified split if `groups` is None (only used
    where no subject/group identifier can be recovered from the data at
    all -- currently just the Epileptic Seizure Recognition rows).
    """
    idx = np.arange(n)
    if groups is None:
        train_idx, temp_idx = train_test_split(idx, test_size=test_frac, stratify=labels, random_state=seed)
        temp_labels = [labels[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(temp_idx, test_size=val_frac_of_temp, stratify=temp_labels, random_state=seed)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        train_idx, temp_idx = next(gss.split(idx, labels, groups=groups))
        temp_groups = [groups[i] for i in temp_idx]
        temp_labels = [labels[i] for i in temp_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_temp, random_state=seed)
        val_idx_rel, test_idx_rel = next(gss2.split(np.arange(len(temp_idx)), temp_labels, groups=temp_groups))
        val_idx = temp_idx[val_idx_rel]
        test_idx = temp_idx[test_idx_rel]
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def _grouped_train_val_test_split(
    items: list, labels: list, groups: list | None, seed: int,
    test_frac: float = 0.3, val_frac_of_temp: float = 0.5,
) -> tuple[list, list, list]:
    """Item-list convenience wrapper around `_grouped_index_split`."""
    train_idx, val_idx, test_idx = _grouped_index_split(len(items), labels, groups, seed, test_frac, val_frac_of_temp)
    return [items[i] for i in train_idx], [items[i] for i in val_idx], [items[i] for i in test_idx]


def _voice_subject_groups(csv_path: Path) -> list[str]:
    import csv as _csv
    groups = []
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            # names look like "phon_R01_S01_1" -> subject id "S01"
            name = row["name"]
            subject = name.split("_")[2]
            groups.append(subject)
    return groups


def _flatten(metrics: dict) -> dict:
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}_class{sub_k}"] = sub_v
        else:
            flat[k] = v
    return flat


def main():
    global SEED
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-checkpoints", action="store_true", default=False)
    parser.add_argument("--save-predictions", action="store_true", default=False,
                         help="Save raw (y_true, y_pred, y_proba) per experiment to results/predictions/ "
                              "for generating real confusion-matrix / ROC-curve figures.")
    args = parser.parse_args()

    SEED = args.seed
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    log(f"Device: {DEVICE}, seed: {SEED}")
    results: list[dict] = []

    experiment_alzheimer(log, results, save_ckpt=args.save_checkpoints, save_predictions=args.save_predictions)
    experiment_pd_motor(log, results, save_ckpt=args.save_checkpoints, save_predictions=args.save_predictions)
    experiment_pd_voice(log, results, save_ckpt=args.save_checkpoints, save_predictions=args.save_predictions)
    experiment_epilepsy(log, results, save_ckpt=args.save_checkpoints, save_predictions=args.save_predictions)
    experiment_camfn_joint(log, results, save_ckpt=args.save_checkpoints, save_predictions=args.save_predictions)
    experiment_fusion_ablation(log, results, save_predictions=args.save_predictions)

    log(f"\nTotal wall time: {time.time() - t0:.1f}s")

    for r in results:
        r["seed"] = SEED

    all_keys = sorted({k for r in results for k in r.keys()})
    out_csv = RESULTS_DIR / f"real_results_seed{SEED}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    with open(RESULTS_DIR / f"run_seed{SEED}.log", "w") as f:
        f.write("\n".join(log_lines))

    log(f"\nWrote {out_csv} and run_seed{SEED}.log")


if __name__ == "__main__":
    main()
