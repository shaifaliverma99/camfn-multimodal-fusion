"""Structural-imaging modality dataset.

Reads the OASIS-1-derived 2D MRI slice JPEGs bundled locally under
`Datasets/Alzheimer/Data/<class>/*.jpg` (see docs/DATA_SOURCES.md for why
this stands in for ADNI/OASIS-3). Yields `[1, H, W]` grayscale tensors.

**Subject-level leakage warning**: filenames look like
`OAS1_0278_MR1_mpr-3_113.jpg` -- `OAS1_0278` is the OASIS-1 subject ID,
and each subject contributes many slices (~249 on average here, since
86,437 slices come from only 347 unique subjects). A plain random or
stratified split over individual slice files lets the same patient's
scans appear in both train and test, which a model can exploit (patient-
specific anatomy/artifacts) instead of learning genuine disease signal.
`subject_id` below lets callers do a *grouped* split instead -- see
`experiments/run_real_experiments.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
_SUBJECT_RE = re.compile(r"^(OAS1_\d+)")


class OASIS1SliceDataset(Dataset):
    """One sample = one MRI slice image + its dementia-severity class index.

    `class_names` gives the CDR-like severity ordering; index 0 = healthy
    control, which lets `MultimodalNeuroDataset` map this onto the shared
    5-way diagnosis label (HC vs AD vs ...).
    """

    def __init__(self, root: str | Path, image_size: int = 128):
        self.root = Path(root)
        self.image_size = image_size
        self.samples: list[tuple[Path, int, str]] = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.glob("*.jpg"):
                m = _SUBJECT_RE.match(img_path.stem)
                subject_id = m.group(1) if m else img_path.stem
                self.samples.append((img_path, class_idx, subject_id))
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),  # -> [1, H, W] in [0, 1]
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, class_idx, subject_id = self.samples[idx]
        img = Image.open(path)
        tensor = self.transform(img)
        return {
            "mri": tensor,
            "severity_class": class_idx,  # 0=none,1=very-mild,2=mild,3=moderate
            "cdr_like": torch.tensor(class_idx / (len(CLASS_NAMES) - 1), dtype=torch.float32),
            "subject_id": subject_id,
            "source_path": str(path),
        }
