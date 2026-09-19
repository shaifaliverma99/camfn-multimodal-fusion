"""Generates real confusion-matrix and ROC-curve figures from actual
model predictions -- reads the .npz files written by
run_real_experiments.py when run with --save-predictions, computes
sklearn confusion matrices / ROC curves directly from those real
(y_true, y_pred, y_proba) arrays, and saves PNGs. Nothing here is drawn
by hand or estimated -- every pixel traces back to a real held-out
prediction.

Run (after `python experiments/run_real_experiments.py --seed 42
--save-checkpoints --save-predictions`):
    python experiments/generate_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc

from src.data.multimodal_dataset import DIAGNOSIS_CLASSES

PRED_DIR = Path(__file__).resolve().parent / "results" / "predictions"
FIG_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"

BINARY_EXPERIMENTS = {
    "alzheimer_hc_vs_ad": ("HC", "AD"),
    "pd_motor_spiral": ("healthy", "PD"),
    "pd_motor_wave": ("healthy", "PD"),
    "pd_voice": ("healthy", "PD"),
    "epilepsy_seizure_vs_not": ("non-seizure", "seizure"),
}
MULTICLASS_EXPERIMENTS = ["camfn_joint_5way", "fusion_ablation_naive_mean"]


def _load(name: str) -> dict | None:
    path = PRED_DIR / f"{name}.npz"
    if not path.exists():
        print(f"[skip] {name}: no predictions file at {path} (run with --save-predictions first)")
        return None
    data = np.load(path)
    return {"y_true": data["y_true"], "y_pred": data["y_pred"], "y_proba": data["y_proba"]}


def plot_binary_confusion_and_roc(name: str, class_names: tuple[str, str]) -> None:
    d = _load(name)
    if d is None:
        return
    y_true, y_pred, y_proba = d["y_true"], d["y_pred"], d["y_proba"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{name}\nconfusion matrix (n={len(y_true)})")

    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=axes[1])
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[1].set_title(f"{name}\nROC curve (AUC={roc_auc:.3f})")

    fig.tight_layout()
    out = FIG_DIR / f"{name}_cm_roc.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] wrote {out}")


def plot_multiclass_confusion(name: str) -> None:
    d = _load(name)
    if d is None:
        return
    y_true, y_pred = d["y_true"], d["y_pred"]
    present_classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels_present = [DIAGNOSIS_CLASSES[c] for c in present_classes]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    ConfusionMatrixDisplay(cm, display_labels=labels_present).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title(f"{name}\nconfusion matrix (n={len(y_true)})\n(Comorbid omitted: 0 test samples)")
    fig.tight_layout()
    out = FIG_DIR / f"{name}_cm.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] wrote {out}")


def main() -> None:
    for name, class_names in BINARY_EXPERIMENTS.items():
        plot_binary_confusion_and_roc(name, class_names)
    for name in MULTICLASS_EXPERIMENTS:
        plot_multiclass_confusion(name)


if __name__ == "__main__":
    main()
