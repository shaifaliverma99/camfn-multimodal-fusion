# CAMFN — Cross-Attention Multimodal Fusion Network

A standalone implementation of the "End-to-End Multimodal Deep Fusion
System for Neurological Disorders (AD, PD, Epilepsy)" spec — architecture,
real training runs, a dashboard, and an IEEE-style paper — kept
deliberately separate from the existing `NEUROFUSION_RUN` /
`NEURO_FUSION_PHASE2_*` projects elsewhere in this repo.

**Everything here is trained/evaluated on real, open data with real
results — including a model that fails.** Nothing is simulated, and
nothing is presented as more validated than it is. Read
`docs/DATA_SOURCES.md` and `docs/RESULTS.md` before trusting any number.

## What's here

- **`docs/ARCHITECTURE.md`** — full design: encoders, cross-attention
  fusion, dynamic gated fusion, heads, equations, data-flow diagram.
- **`docs/DATA_SOURCES.md`** — **read this first.** ADNI/OASIS-3/PPMI/TUH
  EEG are restricted-access datasets that can't be scripted into this
  repo; documents exactly which real, open datasets already sitting in
  this repo stand in for each one.
- **`docs/RESULTS.md`** — **read this second.** Real accuracy/F1/AUC/
  sensitivity/specificity, averaged over 3 seeds, for every model —
  including the voice classifier's outright failure (majority-class
  collapse) and the joint model's unstable AD recall. See
  `experiments/results/` for the raw per-seed CSVs/logs and
  `aggregate_summary.json`.
- **`src/data/`, `src/models/`, `src/training/`** — the CAMFN network and
  its data pipeline (see previous section of this file's git history for
  the original architecture-only pass).
- **`experiments/run_real_experiments.py`** — trains + evaluates every
  per-modality baseline and the joint 5-way CAMFN model on real data;
  `--seed N [--save-checkpoints]` to reproduce.
- **`experiments/checkpoints/`** — the actual trained weights (seed 42)
  that `app/` loads for inference. **Not committed to git** (this repo's
  `.gitignore` excludes `*.pt` repo-wide) — regenerate them locally in
  ~90 seconds with the command below before running the dashboard/API.
- **`app/backend/`** — FastAPI inference API (`/predict/mri`, `/motor`,
  `/voice`, `/eeg`, `/joint`) wrapping the real checkpoints.
- **`app/dashboard/streamlit_app.py`** — the clinical dashboard: upload
  drop-zones (jpg/png/nii/edf/csv), MRI/EEG viewers, per-class probability
  bars, fusion-gate explainability, and a PDF report download — all
  backed by real inference, with every tab's caption stating that
  model's real held-out accuracy (and, for voice, its real failure).
- **`app/reports/report_generator.py`** — the PDF report generator,
  populated only with fields the model actually returned.
- **`paper/camfn_paper.tex`** (+ `refs.bib`, compiled `camfn_paper.pdf`)
  — the IEEE-style paper: structured abstract, literature survey table of
  10 real, individually-verified studies, methodology with equations and
  an ASCII data-flow diagram, the real results above, and a Discussion
  section that states plainly what the joint experiment does and does not
  prove. All 22 references were checked against a live source (author
  list, venue, and either a DOI or arXiv ID) before being cited — none are
  invented.
- **`scripts/check_data_sources.py`**, **`scripts/smoke_test.py`**,
  **`tests/`** — data-availability report and shape/gradient/missing-
  modality robustness checks.

## Quickstart

```bash
cd /workspace/camfn_multimodal_system
pip install -r requirements.txt        # torch/fastapi/streamlit/etc. already present in this env
python scripts/check_data_sources.py    # see what real data is available
pytest tests/                           # shape/gradient/robustness checks

# Reproduce the real results (docs/RESULTS.md), ~90s/seed on one GPU:
python experiments/run_real_experiments.py --seed 42 --save-checkpoints

# Run the dashboard (uses the checkpoints above):
streamlit run app/dashboard/streamlit_app.py --server.port 8501

# Or the API:
uvicorn app.backend.main:app --port 8000   # docs at /docs

# Build the paper:
cd paper && pdflatex camfn_paper && bibtex camfn_paper && pdflatex camfn_paper && pdflatex camfn_paper
```

## Known limitations (by design, not oversight)

- No true multimodal *patient-level* pairing exists across the four
  cohorts used (see `docs/DATA_SOURCES.md`) — every real sample carries
  exactly one real modality, others masked absent. The joint experiment
  validates missing-modality-robust routing, **not** a multimodal fusion
  benefit — see `docs/RESULTS.md` and the paper's Discussion section.
- MRI/motor encoders run on 2D slices/images, not 3D NIfTI or DaT-SPECT;
  `Volumetric3DEncoder` is an untested extension point.
- `simplified_asr_reject` is a variance-threshold stand-in for full ASR.
- The PD voice classifier collapses to majority-class prediction on its
  real held-out test set (macro-F1 0.44, AUC 0.54) — a genuine
  data-scarcity failure, reported rather than hidden, and surfaced in the
  dashboard's own UI.
- CHB-MIT seizure-interval labels aren't parsed yet; the trained EEG model
  uses the Epileptic Seizure Recognition CSV, which has real labels.
