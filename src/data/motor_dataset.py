"""Motor modality dataset — Parkinson's spiral/wave drawings.

Real, open dataset (healthy vs. Parkinson's, spiral and wave drawing
tasks). Stands in for PPMI kinematic/gait streams — see
docs/DATA_SOURCES.md. This is image data, not a raw time series, so it's
routed through `ImageEncoder2D` (a second instance, separate weights from
the MRI encoder).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ParkinsonDrawingDataset(Dataset):
    """One sample = one spiral or wave drawing + healthy/PD label.

    `root` should point at `Datasets/Parkinson`, which contains
    `{spiral,wave}/{training,testing}/{healthy,parkinson}/*.png`.
    """

    def __init__(self, root: str | Path, task: str = "spiral", split: str = "training", image_size: int = 128):
        assert task in ("spiral", "wave")
        self.root = Path(root) / task / split
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []
        for label_name, label_idx in (("healthy", 0), ("parkinson", 1)):
            class_dir = self.root / label_name
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    self.samples.append((img_path, label_idx))
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        img = Image.open(path)
        tensor = self.transform(img)
        return {
            "motor": tensor,
            "pd_motor_label": label,  # 0=healthy, 1=parkinson
            "source_path": str(path),
        }
