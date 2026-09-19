"""Signal-processing utilities for the EEG and kinematic/acoustic pipelines.

Everything here operates on plain numpy arrays so it's usable both inside
Dataset classes and in ad-hoc notebooks/scripts.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, stft, welch


def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 60.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass, applied along the last axis.

    Matches spec section 3 ("Bandpass filtering (0.5-60 Hz)").
    """
    nyq = fs / 2.0
    high = min(high, nyq * 0.98)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def notch_filter(signal: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Notch out line noise (50 Hz EU / 60 Hz US), applied along the last axis."""
    nyq = fs / 2.0
    b, a = iirnotch(freq / nyq, q)
    return filtfilt(b, a, signal, axis=-1)


def simplified_asr_reject(signal: np.ndarray, threshold_std: float = 5.0) -> np.ndarray:
    """Simplified stand-in for Artifact Subspace Reconstruction (ASR).

    Real ASR (Mullen et al.) calibrates a clean-data covariance subspace and
    projects out high-variance artifact components. That needs a calibration
    window and a package like `asrpy`, which isn't a dependency here. This
    function instead clips per-channel samples that exceed
    `threshold_std` standard deviations from that channel's mean, which
    removes the worst amplitude artifacts without claiming full ASR.
    Swap in real ASR by replacing this function's body once `asrpy` is
    available — the call site (`eeg_dataset.py`) doesn't need to change.
    """
    out = signal.copy()
    mean = out.mean(axis=-1, keepdims=True)
    std = out.std(axis=-1, keepdims=True) + 1e-8
    z = (out - mean) / std
    mask = np.abs(z) > threshold_std
    clipped = mean + np.clip(z, -threshold_std, threshold_std) * std
    out[mask] = clipped[mask]
    return out


def stft_spectrogram(signal: np.ndarray, fs: float, nperseg: int = 64, noverlap: int = 32) -> np.ndarray:
    """Short-time Fourier transform magnitude spectrogram per channel.

    Input: [..., T]. Output: [..., F, frames] magnitude spectrogram.
    """
    _, _, Z = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)
    return np.abs(Z)


def welch_psd_tremor_power(signal: np.ndarray, fs: float, band: tuple[float, float] = (3.0, 7.0)) -> np.ndarray:
    """Welch PSD, integrated over `band`, as a scalar tremor-power feature.

    Parkinsonian rest tremor is classically ~3-7 Hz; this returns the
    band-limited power per channel/sample. Spec section 3: "tremor frequency
    power density via Welch's method".
    """
    freqs, psd = welch(signal, fs=fs, axis=-1)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return psd[..., band_mask].sum(axis=-1)


def stride_time_variability(step_timestamps: np.ndarray) -> float:
    """Coefficient of variation of stride (step-to-step) intervals.

    Expects a 1D array of heel-strike/step event timestamps (seconds).
    Not exercised by any dataset bundled in this repo (no raw gait event
    logs are present locally) — kept here as the documented extension point
    for when mPower/PhysioNet gait accelerometer data is available; see
    docs/DATA_SOURCES.md.
    """
    intervals = np.diff(np.sort(step_timestamps))
    if len(intervals) < 2:
        return float("nan")
    return float(intervals.std() / (intervals.mean() + 1e-8))
