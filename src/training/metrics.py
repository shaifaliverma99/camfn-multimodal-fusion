"""Diagnostic classification metrics, per spec section 5 (Accuracy, F1,
AUC-ROC, Sensitivity, Specificity). Computed only over samples with a valid
diagnosis label (see multimodal_dataset.py on why some rows are excluded).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_classes: int
) -> dict:
    """
    y_true, y_pred: [N] int class indices.
    y_proba: [N, n_classes] softmax probabilities.
    """
    result: dict = {}
    result["accuracy"] = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    result["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)), zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    sensitivity, specificity = {}, {}
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity[c] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        specificity[c] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    result["sensitivity_per_class"] = sensitivity
    result["specificity_per_class"] = specificity

    present_classes = sorted(set(y_true.tolist()))
    if n_classes == 2:
        # Binary: sklearn wants the single positive-class score, not a 2-col array.
        try:
            result["auc_roc"] = float(roc_auc_score(y_true, y_proba[:, 1])) if len(present_classes) >= 2 else float("nan")
        except ValueError:
            result["auc_roc"] = float("nan")
    else:
        # Multiclass one-vs-rest, macro-averaged over only the classes that
        # actually have >=1 positive example in y_true (a class with zero
        # examples — e.g. "Comorbid" here — has no well-defined OvR AUC and
        # is honestly excluded rather than making the whole average NaN).
        per_class_auc = []
        for c in range(n_classes):
            if c not in present_classes:
                continue
            try:
                per_class_auc.append(roc_auc_score((y_true == c).astype(int), y_proba[:, c]))
            except ValueError:
                continue
        result["auc_roc_ovr_macro"] = float(np.mean(per_class_auc)) if per_class_auc else float("nan")
        result["auc_roc_ovr_macro_n_classes_included"] = len(per_class_auc)

    return result
