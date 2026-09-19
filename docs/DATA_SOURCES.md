# Data Sources — What the Spec Asked For vs. What This Build Uses

The system prompt names ADNI, OASIS-3, PPMI, mPower/PhysioNet, and TUH EEG as
mandatory sources. Here's the honest status of each, and what this skeleton
actually reads from disk.

| Spec source | Access reality | What this build uses instead | Path |
|---|---|---|---|
| ADNI / OASIS-3 (T1 MRI, PET, CDR) | Requires a signed Data Use Agreement per named individual; cannot be scripted | **OASIS-1** 2D MRI slices (public, no-DUA release from the same lab family) as a structural-imaging stand-in — 4 classes: Non/Very-Mild/Mild/Moderate Dementia | `Datasets/Alzheimer/Data/` |
| PPMI (DaT-SPECT, MRI, kinematics, voice, UPDRS) | Requires PPMI registration + approval | **Parkinson's Drawing dataset** (spiral/wave, healthy vs. PD, open Kaggle release) as a motor-modality stand-in; **UCI Parkinson's voice dataset** (22 acoustic features, open) as the vocal-biomarker stand-in | `Datasets/Parkinson/{spiral,wave}/`, `Datasets/Parkinson_voice/parkinsons.data` |
| mPower / PhysioNet gait+phonation | mPower requires Synapse registration | Not integrated in this pass — the loader interface (`MultimodalNeuroDataset`) has a slot for a raw accelerometer/phonation stream; wire it in once you have mPower access | — |
| TUH EEG Seizure Corpus | Requires a signed TUH data agreement | **CHB-MIT** (open PhysioNet release, already present locally) for raw multi-channel scalp EEG, plus the **Epileptic Seizure Recognition** dataset (open, single-channel 178-sample windows) for a fast tabular-EEG baseline | `chbmit/chb01..chb05/*.edf`, `Datasets/Epilepsy/Epileptic Seizure Recognition.csv` |

None of these substitutions are disguised as the restricted sources anywhere
in code, configs, or docs — dataset classes are named for what they actually
load (`OASIS1SliceDataset`, `CHBMITWindowDataset`, etc.), not for the
restricted corpora they stand in for.

**If/when you get credentialed access**, swap in the real thing by:
1. Writing a new `Dataset` in `src/data/` that yields the same tensor shapes
   the corresponding encoder expects (documented in each encoder's docstring).
2. Pointing `configs/default.yaml` at the new class/path.
No changes to the model or fusion code are needed — that's the point of the
encoder interface.

## What's NOT claimed
- No accuracy/F1/AUC numbers exist yet for this system on any dataset. The
  `scripts/smoke_test.py` script only checks that tensors flow through the
  network with correct shapes, on both real single batches and dummy data —
  it is not a trained model and reports no diagnostic performance.
- The MRI encoder operates on 2D slices because that's what's available
  locally (OASIS-1 release format), not on 3D NIfTI volumes. A
  `Volumetric3DEncoder` stub is included for when real ADNI/OASIS-3 NIfTI
  data is available, but it is untested against real volumetric data here.
- "ASR" (Artifact Subspace Reconstruction) in `src/data/preprocessing.py` is
  a simplified variance/amplitude-threshold artifact rejector, not the full
  PCA-based ASR algorithm (which needs calibration data and a package like
  `asrpy`, not installed here). It's labeled `simplified_asr_reject` in code,
  not `asr`, to avoid overclaiming.
