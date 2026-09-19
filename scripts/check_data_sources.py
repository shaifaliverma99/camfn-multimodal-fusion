#!/usr/bin/env python3
"""Reports which real data this repo actually has for each modality, and is
explicit about which spec-mandated restricted sources are NOT integrated.

Run from anywhere: `python camfn_multimodal_system/scripts/check_data_sources.py`
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # /workspace

CHECKS = [
    ("OASIS-1 MRI slices (Alzheimer stand-in for ADNI/OASIS-3)", REPO_ROOT / "Datasets/Alzheimer/Data"),
    ("Parkinson's spiral/wave drawings (PPMI motor stand-in)", REPO_ROOT / "Datasets/Parkinson"),
    ("UCI Parkinson's voice dataset (PPMI voice stand-in)", REPO_ROOT / "Datasets/Parkinson_voice/parkinsons.data"),
    ("Epileptic Seizure Recognition CSV (TUH EEG stand-in)", REPO_ROOT / "Datasets/Epilepsy/Epileptic Seizure Recognition.csv"),
    ("CHB-MIT raw EDF recordings (TUH EEG stand-in, raw multichannel)", REPO_ROOT / "chbmit"),
]

NOT_INTEGRATED = [
    "ADNI (restricted; needs signed DUA)",
    "OASIS-3 (restricted; needs signed DUA)",
    "PPMI (restricted; needs PPMI registration/approval)",
    "mPower / PhysioNet gait+phonation (restricted; needs Synapse registration)",
    "TUH EEG Seizure Corpus (restricted; needs signed TUH data agreement)",
]


def main() -> None:
    print("=== Data sources actually present ===")
    for name, path in CHECKS:
        status = "FOUND" if path.exists() else "MISSING"
        print(f"[{status}] {name}\n         -> {path}")

    print("\n=== Spec-mandated restricted sources NOT integrated ===")
    for name in NOT_INTEGRATED:
        print(f" - {name}")
    print(
        "\nSee docs/DATA_SOURCES.md for the full mapping and for how to swap "
        "in real credentialed data once you have access."
    )


if __name__ == "__main__":
    main()
