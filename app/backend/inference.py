"""Shared inference module — loads the real checkpoints produced by
experiments/run_real_experiments.py and runs genuine forward passes.

Used by both the FastAPI backend (main.py) and the Streamlit dashboard, so
there's exactly one code path for "what does the model actually predict."

No numbers are fabricated here: if a checkpoint is missing, the
corresponding predictor reports `trained=False` and returns None rather
than a fake probability.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]  # /workspace
PKG_ROOT = Path(__file__).resolve().parents[2]   # camfn_multimodal_system
sys.path.insert(0, str(PKG_ROOT))

from src.data.preprocessing import bandpass_filter, notch_filter, simplified_asr_reject  # noqa: E402
from src.models.camfn import CAMFN  # noqa: E402
from src.models.encoders.eeg_encoder import EEGEncoder  # noqa: E402
from src.models.encoders.image_encoder import ImageEncoder2D  # noqa: E402
from src.models.encoders.tabular_encoder import TabularTemporalEncoder  # noqa: E402
from src.data.multimodal_dataset import MODALITIES, DIAGNOSIS_CLASSES  # noqa: E402
from src.data.acoustic_dataset import FEATURE_COLUMNS  # noqa: E402

CKPT_DIR = PKG_ROOT / "experiments" / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REAL_METRICS_NOTE = (
    "Predictions come from checkpoints trained in experiments/run_real_experiments.py "
    "on real, open data (see docs/DATA_SOURCES.md). See docs/RESULTS.md for honest "
    "accuracy/F1/sensitivity/specificity on each model's held-out test split — these "
    "are research-grade models on substitute open datasets, NOT a validated clinical tool."
)


class SingleModalityClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.encoder(x)
        return self.head(pooled)


def _load_checkpoint(model: nn.Module, name: str) -> tuple[nn.Module, bool]:
    path = CKPT_DIR / name
    if not path.exists():
        return model, False
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, True


class ModelRegistry:
    """Loads every available checkpoint once at import time."""

    def __init__(self):
        self.mri_model, self.mri_trained = _load_checkpoint(
            SingleModalityClassifier(ImageEncoder2D(d_model=64), 64, 2), "alzheimer_mri.pt"
        )
        self.motor_spiral_model, self.motor_spiral_trained = _load_checkpoint(
            SingleModalityClassifier(ImageEncoder2D(d_model=64), 64, 2), "pd_motor_spiral.pt"
        )
        self.motor_wave_model, self.motor_wave_trained = _load_checkpoint(
            SingleModalityClassifier(ImageEncoder2D(d_model=64), 64, 2), "pd_motor_wave.pt"
        )
        self.voice_model, self.voice_trained = _load_checkpoint(
            SingleModalityClassifier(TabularTemporalEncoder(in_features=22, d_model=64), 64, 2), "pd_voice.pt"
        )
        self.eeg_model, self.eeg_trained = _load_checkpoint(
            SingleModalityClassifier(EEGEncoder(in_channels=1, d_model=64), 64, 2), "epilepsy_eeg.pt"
        )
        self.camfn_model, self.camfn_trained = _load_checkpoint(
            CAMFN(d_model=64, n_heads=4, n_fusion_layers=2, eeg_in_channels=1, acoustic_in_features=22, n_diagnosis_classes=5),
            "camfn_joint.pt",
        )


REGISTRY = ModelRegistry()

_MRI_TRANSFORM_SIZE = 64
_MOTOR_TRANSFORM_SIZE = 96


def _pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


def predict_mri(image_bytes: bytes) -> dict:
    if not REGISTRY.mri_trained:
        return {"trained": False, "error": "alzheimer_mri.pt checkpoint not found"}
    img = Image.open(io.BytesIO(image_bytes))
    x = _pil_to_tensor(img, _MRI_TRANSFORM_SIZE).to(DEVICE)
    with torch.no_grad():
        proba = torch.softmax(REGISTRY.mri_model(x), dim=-1)[0].cpu().numpy()
    return {
        "trained": True,
        "labels": ["HC", "AD"],
        "probabilities": {"HC": float(proba[0]), "AD": float(proba[1])},
        "predicted": "AD" if proba[1] > proba[0] else "HC",
        "note": REAL_METRICS_NOTE,
    }


def predict_motor(image_bytes: bytes, task: str = "spiral") -> dict:
    model = REGISTRY.motor_spiral_model if task == "spiral" else REGISTRY.motor_wave_model
    trained = REGISTRY.motor_spiral_trained if task == "spiral" else REGISTRY.motor_wave_trained
    if not trained:
        return {"trained": False, "error": f"pd_motor_{task}.pt checkpoint not found"}
    img = Image.open(io.BytesIO(image_bytes))
    x = _pil_to_tensor(img, _MOTOR_TRANSFORM_SIZE).to(DEVICE)
    with torch.no_grad():
        proba = torch.softmax(model(x), dim=-1)[0].cpu().numpy()
    return {
        "trained": True,
        "labels": ["healthy", "parkinson"],
        "probabilities": {"healthy": float(proba[0]), "parkinson": float(proba[1])},
        "predicted": "parkinson" if proba[1] > proba[0] else "healthy",
        "note": REAL_METRICS_NOTE,
    }


def predict_voice(features: dict[str, float]) -> dict:
    if not REGISTRY.voice_trained:
        return {"trained": False, "error": "pd_voice.pt checkpoint not found"}
    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        return {"trained": False, "error": f"missing required features: {missing}"}
    x = torch.tensor([[features[c] for c in FEATURE_COLUMNS]], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        proba = torch.softmax(REGISTRY.voice_model(x), dim=-1)[0].cpu().numpy()
    return {
        "trained": True,
        "labels": ["healthy", "parkinson"],
        "probabilities": {"healthy": float(proba[0]), "parkinson": float(proba[1])},
        "predicted": "parkinson" if proba[1] > proba[0] else "healthy",
        "note": REAL_METRICS_NOTE
        + " NOTE: this is a weak model -- 5-fold subject-grouped cross-validation over all "
        "195 training recordings gives macro-F1 ~0.58 and AUC ~0.53 (barely above chance; "
        "see docs/RESULTS.md). The specific checkpoint deployed here is one CV fold's model, "
        "whose individual behavior can be better or worse than that aggregate -- treat any "
        "single prediction from it as low-confidence, not a real diagnostic signal.",
    }


def predict_eeg_window(samples: list[float], fs: float = 178.0, already_178: bool = True) -> dict:
    """samples: a single-channel EEG window. If `already_178` is False, the
    signal is bandpass/notch/artifact-cleaned first (for raw uploads)."""
    if not REGISTRY.eeg_trained:
        return {"trained": False, "error": "epilepsy_eeg.pt checkpoint not found"}
    arr = np.asarray(samples, dtype=np.float32)
    if not already_178:
        arr = bandpass_filter(arr[None, :], fs=fs)[0]
        try:
            arr = notch_filter(arr[None, :], fs=fs, freq=min(50.0, fs / 2 - 1))[0]
        except ValueError:
            pass
        arr = simplified_asr_reject(arr[None, :])[0]
    if len(arr) != 178:
        # resample to 178 samples to match the training distribution's window length
        idx = np.linspace(0, len(arr) - 1, 178)
        arr = np.interp(idx, np.arange(len(arr)), arr)
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,178]
    with torch.no_grad():
        proba = torch.softmax(REGISTRY.eeg_model(x), dim=-1)[0].cpu().numpy()
    return {
        "trained": True,
        "labels": ["non_seizure", "seizure"],
        "probabilities": {"non_seizure": float(proba[0]), "seizure": float(proba[1])},
        "predicted": "seizure" if proba[1] > proba[0] else "non_seizure",
        "note": REAL_METRICS_NOTE,
    }


def predict_joint(
    mri_bytes: bytes | None = None,
    motor_bytes: bytes | None = None,
    voice_features: dict[str, float] | None = None,
    eeg_samples: list[float] | None = None,
) -> dict:
    """Runs the full CAMFN model over whichever modalities are supplied,
    masking the rest absent (exactly the missing-modality path the model
    was trained with -- see docs/DATA_SOURCES.md on why no upload here will
    ever have all four modalities from one real patient).
    """
    if not REGISTRY.camfn_trained:
        return {"trained": False, "error": "camfn_joint.pt checkpoint not found"}

    image_size = 64
    presence = [0.0, 0.0, 0.0, 0.0]  # order: MODALITIES = mri, eeg, motor, acoustic
    mri_t = torch.zeros(1, 1, image_size, image_size)
    motor_t = torch.zeros(1, 1, image_size, image_size)
    acoustic_t = torch.zeros(1, 22)
    eeg_t = None

    if mri_bytes is not None:
        img = Image.open(io.BytesIO(mri_bytes))
        mri_t = _pil_to_tensor(img, image_size)
        presence[MODALITIES.index("mri")] = 1.0
    if motor_bytes is not None:
        img = Image.open(io.BytesIO(motor_bytes))
        motor_t = _pil_to_tensor(img, image_size)
        presence[MODALITIES.index("motor")] = 1.0
    if voice_features is not None:
        acoustic_t = torch.tensor([[voice_features[c] for c in FEATURE_COLUMNS]], dtype=torch.float32)
        presence[MODALITIES.index("acoustic")] = 1.0
    if eeg_samples is not None:
        arr = np.asarray(eeg_samples, dtype=np.float32)
        if len(arr) != 178:
            idx = np.linspace(0, len(arr) - 1, 178)
            arr = np.interp(idx, np.arange(len(arr)), arr)
        eeg_t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        presence[MODALITIES.index("eeg")] = 1.0

    if sum(presence) == 0:
        return {"trained": True, "error": "no modality supplied"}

    batch = {
        "mri": mri_t.to(DEVICE),
        "motor": motor_t.to(DEVICE),
        "acoustic": acoustic_t.to(DEVICE),
        "eeg": eeg_t.to(DEVICE) if eeg_t is not None else None,
        "presence_mask": torch.tensor([presence], dtype=torch.float32).to(DEVICE),
    }
    with torch.no_grad():
        out = REGISTRY.camfn_model(batch)
        proba = torch.softmax(out["diagnosis_logits"], dim=-1)[0].cpu().numpy()
        gates = out["fusion_gates"][0].cpu().numpy()

    return {
        "trained": True,
        "labels": DIAGNOSIS_CLASSES,
        "probabilities": {c: float(p) for c, p in zip(DIAGNOSIS_CLASSES, proba)},
        "predicted": DIAGNOSIS_CLASSES[int(proba.argmax())],
        "fusion_gates": {m: float(g) for m, g in zip(MODALITIES, gates)},
        "modalities_present": [m for m, p in zip(MODALITIES, presence) if p > 0],
        "note": REAL_METRICS_NOTE
        + " Comorbid class has zero training examples in this system (see docs/DATA_SOURCES.md) "
        "-- treat any 'Comorbid' probability as uncalibrated.",
    }
