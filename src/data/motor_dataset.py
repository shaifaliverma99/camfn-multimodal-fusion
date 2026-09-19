"""Motor modality dataset — Parkinson's spiral/wave drawings.

Real, open dataset (healthy vs. Parkinson's, spiral and wave drawing
tasks). Stands in for PPMI kinematic/gait streams — see
docs/DATA_SOURCES.md. This is image data, not a raw time series, so it's
routed through `ImageEncoder2D` (a second instance, separate weights from
the MRI encoder).

**Subject-level leakage warning**: this dataset's own provided
training/testing folders are NOT subject-disjoint. Filenames look like
`V01HE02.png` / `V01HE03.png` (subject `V01`, healthy class, trials 02
and 03) or `V01HO01.png` (same subject, wave task, trial 01) --
inspection shows the same subject IDs (e.g. V01-V11) appear on both sides
of the provided training/testing split, for both spiral and wave, in
both classes. Treating that split (or a plain random/stratified split
over individual files) as a real train/test boundary lets the model see
the same person's handwriting in both sets. `subject_key` below exposes
a (class, subject-prefix) key so callers can do a *grouped* split
instead -- see `experiments/run_real_experiments.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_SUBJECT_RE = re.compile(r"^(V\d+)")


class ParkinsonDrawingDataset(Dataset):
    """One sample = one spiral or wave drawing + healthy/PD label.

    `root` should point at `Datasets/Parkinson`, which contains
    `{spiral,wave}/{training,testing}/{healthy,parkinson}/*.png`.
    """

    def __init__(self, root: str | Path, task: str = "spiral", split: str = "training", image_size: int = 128):
        assert task in ("spiral", "wave")
        self.root = Path(root) / task / split
        self.image_size = image_size
        self.samples: list[tuple[Path, int, str]] = []
        for label_name, label_idx in (("healthy", 0), ("parkinson", 1)):
            class_dir = self.root / label_name
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    m = _SUBJECT_RE.match(img_path.stem)
                    subject_prefix = m.group(1) if m else img_path.stem
                    # Subject IDs (e.g. "V01") are only unique *within* a
                    # class -- "V01" in healthy and "V01" in parkinson are
                    # different people -- so the group key includes the
                    # class label.
                    subject_key = f"{label_name}_{subject_prefix}"
                    self.samples.append((img_path, label_idx, subject_key))
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
        path, label, subject_key = self.samples[idx]
        img = Image.open(path)
        tensor = self.transform(img)
        return {
            "motor": tensor,
            "pd_motor_label": label,  # 0=healthy, 1=parkinson
            "subject_key": subject_key,
            "source_path": str(path),
        }
