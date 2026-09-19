"""Structural-imaging modality dataset.

Reads the OASIS-1-derived 2D MRI slice JPEGs bundled locally under
`Datasets/Alzheimer/Data/<class>/*.jpg` (see docs/DATA_SOURCES.md for why
this stands in for ADNI/OASIS-3). Yields `[1, H, W]` grayscale tensors.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]


class OASIS1SliceDataset(Dataset):
    """One sample = one MRI slice image + its dementia-severity class index.

    `class_names` gives the CDR-like severity ordering; index 0 = healthy
    control, which lets `MultimodalNeuroDataset` map this onto the shared
    5-way diagnosis label (HC vs AD vs ...).
    """

    def __init__(self, root: str | Path, image_size: int = 128):
        self.root = Path(root)
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, class_idx))
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
        path, class_idx = self.samples[idx]
        img = Image.open(path)
        tensor = self.transform(img)
        return {
            "mri": tensor,
            "severity_class": class_idx,  # 0=none,1=very-mild,2=mild,3=moderate
            "cdr_like": torch.tensor(class_idx / (len(CLASS_NAMES) - 1), dtype=torch.float32),
            "source_path": str(path),
        }
