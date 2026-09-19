"""Unifies the four single-modality cohorts into one dataset the CAMFN
model can train against.

**Important honesty note**: OASIS-1 (Alzheimer), Parkinson drawings,
UCI Parkinson's voice, and CHB-MIT/Epileptic-Seizure-Recognition are four
*independent* real-world cohorts — there is no shared patient across them,
so there is no genuine per-patient multimodal record to fuse. This module
does not pretend otherwise: every yielded sample carries only the
modality(ies) native to the cohort it came from, with every other modality
marked absent via the presence mask. That's enough to exercise and unit-test
the missing-modality fusion pathway (`DynamicGatedFusion`) against real
data, but it is **not** equivalent to true patient-level multimodal fusion —
that requires linked-record data such as ADNI/PPMI/TUH (see
docs/DATA_SOURCES.md).

Diagnosis label space: 0=HC, 1=AD, 2=PD, 3=Epilepsy, 4=Comorbid.
"Comorbid" (4) has no source cohort here and is simply an unused class slot
in the head — flagged, not silently hidden.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import Dataset

MODALITIES = ["mri", "eeg", "motor", "acoustic"]
DIAGNOSIS_CLASSES = ["HC", "AD", "PD", "Epilepsy", "Comorbid"]

# Static per-modality tensor shapes used to zero-fill absent modalities so
# batches can be stacked. `eeg` is intentionally excluded — see
# `MultimodalNeuroDataset` docstring on the single-EEG-source-per-instance
# limitation.
_FIXED_SHAPES = {
    "mri": None,      # filled in from image_size at construction time
    "motor": None,    # ditto
    "acoustic": (22,),
}


def _alzheimer_label(item: dict) -> tuple[int, bool]:
    label = 0 if item["severity_class"] == 0 else 1  # HC vs AD
    return label, True


def _pd_motor_label(item: dict) -> tuple[int, bool]:
    label = 0 if item["pd_motor_label"] == 0 else 2  # HC vs PD
    return label, True


def _pd_voice_label(item: dict) -> tuple[int, bool]:
    label = 0 if item["pd_voice_label"] == 0 else 2  # HC vs PD
    return label, True


def _epilepsy_label(item: dict) -> tuple[int, bool]:
    # Only rows with a *real* seizure-activity label (from the Epileptic
    # Seizure Recognition codebook, or a future CHB-MIT annotation parser)
    # get a valid diagnosis label. Non-seizure rows in these sources are
    # NOT reliably healthy controls (e.g. tumor-area / eyes-closed EEG
    # states also carry label 0 here) so they're excluded from the primary
    # diagnosis loss rather than mislabeled as HC.
    if item["seizure_label"] == 1:
        return 3, True  # Epilepsy
    return -1, False


class CohortWrapperDataset(Dataset):
    """Projects one single-modality dataset's items into the unified
    multimodal sample schema (all modality slots present, most masked off).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        modality_key: str,
        diagnosis_label_fn: Callable[[dict], tuple[int, bool]],
        cohort_name: str,
        fixed_shapes: dict[str, tuple[int, ...]],
    ):
        self.base_dataset = base_dataset
        self.modality_key = modality_key
        self.diagnosis_label_fn = diagnosis_label_fn
        self.cohort_name = cohort_name
        self.fixed_shapes = fixed_shapes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]
        label, valid = self.diagnosis_label_fn(item)
        sample = {"cohort": self.cohort_name, "diagnosis_label": label, "diagnosis_label_valid": valid}
        # Only OASIS-1 items carry a real (bucket-derived) severity target
        # today; propagate it with an explicit validity flag rather than
        # defaulting absent cohorts to a fake 0.0 target.
        if "cdr_like" in item:
            sample["cdr_like"] = item["cdr_like"]
            sample["cdr_like_valid"] = True
        else:
            sample["cdr_like"] = torch.tensor(0.0, dtype=torch.float32)
            sample["cdr_like_valid"] = False
        for m in MODALITIES:
            if m == self.modality_key:
                sample[m] = item[m]
                sample[f"{m}_present"] = True
            elif m == "eeg":
                sample[m] = None  # handled specially — see collate_fn
                sample[f"{m}_present"] = False
            else:
                shape = self.fixed_shapes[m]
                sample[m] = torch.zeros(shape, dtype=torch.float32)
                sample[f"{m}_present"] = False
        return sample


class MultimodalNeuroDataset(Dataset):
    """Concatenation of several `CohortWrapperDataset`s with a flat index.

    Limitation: at most one of the wrapped cohorts may supply the `eeg`
    modality per instance of this class (CHB-MIT windows and Epileptic-
    Seizure-Recognition rows have different channel counts and can't be
    zero-filled/stacked against each other without padding logic this
    skeleton doesn't implement). Mixing an EEG cohort with non-EEG cohorts
    (MRI/motor/acoustic) is fine and is the intended use.
    """

    def __init__(self, cohort_datasets: list[CohortWrapperDataset]):
        eeg_cohorts = [c for c in cohort_datasets if c.modality_key == "eeg"]
        if len(eeg_cohorts) > 1:
            raise ValueError(
                f"MultimodalNeuroDataset supports at most one EEG-yielding cohort per instance, "
                f"got {len(eeg_cohorts)}: {[c.cohort_name for c in eeg_cohorts]}"
            )
        self.cohorts = cohort_datasets
        self._lengths = [len(c) for c in cohort_datasets]
        self._offsets = [0]
        for length in self._lengths:
            self._offsets.append(self._offsets[-1] + length)
        self.has_eeg = len(eeg_cohorts) == 1

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> dict:
        for cohort_idx, cohort in enumerate(self.cohorts):
            if self._offsets[cohort_idx] <= idx < self._offsets[cohort_idx + 1]:
                return cohort[idx - self._offsets[cohort_idx]]
        raise IndexError(idx)


def collate_fn(batch: list[dict]) -> dict:
    """Stacks a list of unified samples into a batch dict.

    Returns:
        {
          "mri": [B,1,H,W], "motor": [B,1,H,W], "acoustic": [B,22],
          "eeg": [B,C,T] or None if no sample in the batch carries EEG,
          "presence_mask": [B, len(MODALITIES)] float (1=present, 0=absent),
          "diagnosis_label": [B] long,
          "diagnosis_label_valid": [B] bool,
          "cohort": list[str],
        }
    """
    out: dict = {}
    for m in ("mri", "motor", "acoustic"):
        out[m] = torch.stack([b[m] for b in batch], dim=0)

    eeg_tensors = [b["eeg"] for b in batch if b["eeg"] is not None]
    if eeg_tensors:
        shapes = {tuple(t.shape) for t in eeg_tensors}
        if len(shapes) > 1:
            raise ValueError(f"Inconsistent EEG shapes in one batch: {shapes}. See collate_fn limitation note.")
        eeg_shape = eeg_tensors[0].shape
        stacked = []
        for b in batch:
            stacked.append(b["eeg"] if b["eeg"] is not None else torch.zeros(eeg_shape, dtype=torch.float32))
        out["eeg"] = torch.stack(stacked, dim=0)
    else:
        out["eeg"] = None

    presence = torch.tensor(
        [[float(b[f"{m}_present"]) for m in MODALITIES] for b in batch], dtype=torch.float32
    )
    out["presence_mask"] = presence
    out["diagnosis_label"] = torch.tensor([b["diagnosis_label"] for b in batch], dtype=torch.long)
    out["diagnosis_label_valid"] = torch.tensor([b["diagnosis_label_valid"] for b in batch], dtype=torch.bool)
    out["cdr_like"] = torch.stack([b["cdr_like"] for b in batch], dim=0)
    out["cdr_like_valid"] = torch.tensor([b["cdr_like_valid"] for b in batch], dtype=torch.bool)
    out["cohort"] = [b["cohort"] for b in batch]
    return out
