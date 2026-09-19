"""FastAPI backend exposing CAMFN inference over HTTP.

Run: uvicorn app.backend.main:app --reload --port 8000  (from camfn_multimodal_system/)
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from app.backend import inference  # noqa: E402

app = FastAPI(
    title="CAMFN Multimodal Fusion API",
    description=(
        "Research-grade inference API for the CAMFN skeleton. See "
        "docs/DATA_SOURCES.md and docs/RESULTS.md before treating any "
        "output as clinically meaningful."
    ),
    version="0.1.0",
)


class VoiceFeatures(BaseModel):
    features: dict[str, float]


class EEGWindow(BaseModel):
    samples: list[float]
    fs: float = 178.0
    already_178: bool = True


class JointVoiceEEG(BaseModel):
    voice_features: Optional[dict[str, float]] = None
    eeg_samples: Optional[list[float]] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoints_loaded": {
            "mri": inference.REGISTRY.mri_trained,
            "motor_spiral": inference.REGISTRY.motor_spiral_trained,
            "motor_wave": inference.REGISTRY.motor_wave_trained,
            "voice": inference.REGISTRY.voice_trained,
            "eeg": inference.REGISTRY.eeg_trained,
            "camfn_joint": inference.REGISTRY.camfn_trained,
        },
    }


@app.post("/predict/mri")
async def predict_mri(file: UploadFile = File(...)):
    content = await file.read()
    result = inference.predict_mri(content)
    if not result.get("trained", True):
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/predict/motor")
async def predict_motor(file: UploadFile = File(...), task: str = "spiral"):
    if task not in ("spiral", "wave"):
        raise HTTPException(status_code=400, detail="task must be 'spiral' or 'wave'")
    content = await file.read()
    result = inference.predict_motor(content, task=task)
    if not result.get("trained", True):
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/predict/voice")
def predict_voice(payload: VoiceFeatures):
    result = inference.predict_voice(payload.features)
    if not result.get("trained", True):
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/predict/eeg")
def predict_eeg(payload: EEGWindow):
    result = inference.predict_eeg_window(payload.samples, fs=payload.fs, already_178=payload.already_178)
    if not result.get("trained", True):
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/predict/joint")
async def predict_joint(
    mri: Optional[UploadFile] = File(None),
    motor: Optional[UploadFile] = File(None),
    voice_features_json: Optional[str] = None,
    eeg_samples_json: Optional[str] = None,
):
    import json

    mri_bytes = await mri.read() if mri is not None else None
    motor_bytes = await motor.read() if motor is not None else None
    voice_features = json.loads(voice_features_json) if voice_features_json else None
    eeg_samples = json.loads(eeg_samples_json) if eeg_samples_json else None

    result = inference.predict_joint(
        mri_bytes=mri_bytes, motor_bytes=motor_bytes, voice_features=voice_features, eeg_samples=eeg_samples
    )
    if not result.get("trained", True):
        raise HTTPException(status_code=503, detail=result)
    return result
