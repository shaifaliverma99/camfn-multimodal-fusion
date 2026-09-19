"""CAMFN clinical dashboard (Streamlit).

Run: streamlit run app/dashboard/streamlit_app.py  (from camfn_multimodal_system/)

Uses app.backend.inference directly (in-process) rather than making HTTP
calls to the FastAPI service, so the dashboard works standalone. The same
inference module backs app/backend/main.py, so results are identical
either way -- FastAPI is there for programmatic/EHR-style integration, this
is the human-facing UI.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

from app.backend import inference  # noqa: E402
from app.reports.report_generator import generate_report_pdf  # noqa: E402
from src.data.acoustic_dataset import FEATURE_COLUMNS  # noqa: E402

def _try_import_mne():
    """Deferred import: mne is a heavy dependency, only worth paying for
    when someone actually uploads an .edf file (most sessions won't)."""
    try:
        import mne

        mne.set_log_level("ERROR")
        return mne
    except ImportError:
        return None


def _try_import_nibabel():
    """Deferred import: same reasoning as mne, for .nii/.nii.gz uploads."""
    try:
        import nibabel as nib

        return nib
    except ImportError:
        return None


st.set_page_config(page_title="CAMFN Multimodal Diagnosis Dashboard", layout="wide")

HELD_OUT_ACC = {
    "mri": 0.757,            # subject-grouped split, n=1896 (corrected -- see docs/RESULTS.md leakage-fix note)
    "motor_spiral": 0.690,   # 5-fold subject-grouped CV, n=102 (corrected)
    "motor_wave": 0.677,     # 5-fold subject-grouped CV, n=102 (corrected)
    "voice": 0.739,          # 5-fold subject-grouped CV, n=195 (weak model, AUC~0.51 — see docs/RESULTS.md)
    "eeg": 0.988,
    "joint": 0.659,          # corrected split; ablation beats this on every metric — see docs/RESULTS.md
}  # mean over 3 seeds, from experiments/results/aggregate_summary.json — update if you re-run experiments


def _proba_bar(labels: list[str], probs: list[float], title: str):
    fig = go.Figure(go.Bar(x=labels, y=probs, marker_color=["#4C78A8", "#E45756", "#54A24B", "#F58518", "#B279A2"][: len(labels)]))
    fig.update_layout(title=title, yaxis_range=[0, 1], height=320, margin=dict(t=40, b=10))
    return fig


def _warning_banner():
    st.markdown(
        "<div style='background:#7a1f1f;color:white;padding:8px 14px;border-radius:6px;"
        "font-weight:600;margin-bottom:10px'>RESEARCH USE ONLY — not a medical device, "
        "not validated for clinical decision-making. See docs/RESULTS.md for real, honest "
        "held-out performance (including a model that fails).</div>",
        unsafe_allow_html=True,
    )


st.title("CAMFN — Multimodal Neurological Diagnosis Dashboard")
_warning_banner()

with st.expander("Which real data trained each model? (docs/DATA_SOURCES.md summary)", expanded=False):
    st.markdown(
        """
| Tab | Real dataset used | Stands in for (per spec) |
|---|---|---|
| MRI (Alzheimer) | OASIS-1 2D MRI slices | ADNI / OASIS-3 |
| Motor (Parkinson's) | Parkinson's spiral/wave drawings | PPMI kinematics |
| Voice (Parkinson's) | UCI Parkinson's voice dataset (22 features) | PPMI / mPower voice |
| EEG (Epilepsy) | Epileptic Seizure Recognition (178-sample windows) / CHB-MIT | TUH EEG |
| Joint | All four combined, one real modality per sample (masked fusion) | — |
"""
    )

tabs = st.tabs(["MRI (Alzheimer)", "Motor (Parkinson's)", "Voice (Parkinson's)", "EEG (Epilepsy)", "Joint fusion"])

# ---------------------------------------------------------------- MRI tab --
with tabs[0]:
    st.subheader("Structural MRI slice viewer + AD/HC classifier")
    st.caption("Real checkpoint: experiments/checkpoints/alzheimer_mri.pt — held-out test accuracy 75.7% (n=1896), "
               "subject-grouped split (an earlier file-level split leaked patients across train/test and reported "
               "~89%; see docs/RESULTS.md). Operates on 2D slices; a .nii/.nii.gz volume will show its middle "
               "axial/coronal/sagittal slices.")
    upload = st.file_uploader("Upload an MRI slice (.jpg/.png) or a NIfTI volume (.nii/.nii.gz)", type=["jpg", "jpeg", "png", "nii", "gz"], key="mri_upl")
    if upload is not None:
        name = upload.name.lower()
        nib = _try_import_nibabel() if name.endswith((".nii", ".nii.gz")) else None
        if nib is not None:
            data = upload.read()
            tmp_path = Path("/tmp") / upload.name
            tmp_path.write_bytes(data)
            vol = nib.load(str(tmp_path)).get_fdata()
            cx, cy, cz = [s // 2 for s in vol.shape[:3]]
            col1, col2, col3 = st.columns(3)
            for col, (view_name, sl) in zip(
                (col1, col2, col3),
                (("Axial", vol[:, :, cz]), ("Coronal", vol[:, cy, :]), ("Sagittal", vol[cx, :, :])),
            ):
                with col:
                    st.caption(view_name)
                    st.image((sl / (sl.max() + 1e-6) * 255).astype(np.uint8), clamp=True)
            middle_slice = vol[:, :, cz]
            img_for_model = Image.fromarray((middle_slice / (middle_slice.max() + 1e-6) * 255).astype(np.uint8))
            buf = io.BytesIO()
            img_for_model.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        else:
            image_bytes = upload.read()
            st.image(image_bytes, caption="Uploaded slice", width=300)

        result = inference.predict_mri(image_bytes)
        if result.get("trained"):
            st.plotly_chart(_proba_bar(result["labels"], [result["probabilities"][l] for l in result["labels"]], "P(class)"), width='stretch')
            st.success(f"Predicted: **{result['predicted']}**")
            st.caption(result["note"])
            pdf = generate_report_pdf(result, "MRI — Alzheimer's screening", held_out_accuracy=HELD_OUT_ACC["mri"])
            st.download_button("Download PDF report", pdf, file_name="camfn_mri_report.pdf", mime="application/pdf")
        else:
            st.error(result.get("error"))

# --------------------------------------------------------------- Motor tab --
with tabs[1]:
    st.subheader("Spiral/wave drawing viewer + PD motor classifier")
    st.caption("Real checkpoints: pd_motor_spiral.pt (69.0% acc, 5-fold subject-grouped CV, n=102), "
               "pd_motor_wave.pt (67.7% acc, same protocol, n=102) — small cohorts with subject-level grouping "
               "(the dataset's own train/test split reuses subject IDs on both sides), treat as directional.")
    task = st.radio("Drawing task", ["spiral", "wave"], horizontal=True)
    upload = st.file_uploader("Upload a spiral/wave drawing image", type=["jpg", "jpeg", "png"], key="motor_upl")
    if upload is not None:
        image_bytes = upload.read()
        st.image(image_bytes, caption=f"Uploaded {task} drawing", width=300)
        result = inference.predict_motor(image_bytes, task=task)
        if result.get("trained"):
            st.plotly_chart(_proba_bar(result["labels"], [result["probabilities"][l] for l in result["labels"]], "P(class)"), width='stretch')
            st.success(f"Predicted: **{result['predicted']}**")
            st.caption(result["note"])
            pdf = generate_report_pdf(result, f"Motor drawing ({task}) — Parkinson's screening", held_out_accuracy=HELD_OUT_ACC[f"motor_{task}"])
            st.download_button("Download PDF report", pdf, file_name="camfn_motor_report.pdf", mime="application/pdf")
        else:
            st.error(result.get("error"))

# --------------------------------------------------------------- Voice tab --
with tabs[2]:
    st.subheader("Vocal acoustic biomarkers + PD voice classifier")
    st.caption("Real checkpoint: pd_voice.pt. **Weak model**: 5-fold subject-grouped CV over all 195 recordings "
               "gives 73.9% accuracy but macro-F1 0.58 and AUC 0.51 (indistinguishable from chance) — shown anyway "
               "per the 'no fabrication' policy, not hidden. The single checkpoint deployed here is one CV fold's "
               "model. See docs/RESULTS.md.")
    st.markdown("Enter the 22 UCI dysphonia measurements (or upload a one-row CSV with these column names):")
    csv_upload = st.file_uploader("Upload one-row CSV with the 22 feature columns", type=["csv"], key="voice_csv")
    values = {}
    if csv_upload is not None:
        import csv as _csv
        reader = _csv.DictReader(io.StringIO(csv_upload.read().decode()))
        row = next(reader, None)
        missing = [c for c in FEATURE_COLUMNS if row is not None and c not in row]
        if row is None:
            st.error("That CSV has no data row after the header — nothing to load.")
        elif missing:
            st.error(f"CSV is missing expected column(s): {missing}. Expected the 22 UCI feature names as headers.")
        else:
            try:
                values = {c: float(row[c]) for c in FEATURE_COLUMNS}
            except ValueError as e:
                st.error(f"Couldn't parse a numeric value from that CSV: {e}")
    cols = st.columns(4)
    for i, feat in enumerate(FEATURE_COLUMNS):
        with cols[i % 4]:
            values[feat] = st.number_input(feat, value=float(values.get(feat, 0.0)), format="%.6f", key=f"voice_{feat}")
    if st.button("Run voice classifier"):
        result = inference.predict_voice(values)
        if result.get("trained"):
            st.plotly_chart(_proba_bar(result["labels"], [result["probabilities"][l] for l in result["labels"]], "P(class)"), width='stretch')
            st.warning(f"Predicted: **{result['predicted']}** (model is weak — see caption above)")
            st.caption(result["note"])
            pdf = generate_report_pdf(result, "Voice — Parkinson's screening", held_out_accuracy=HELD_OUT_ACC["voice"])
            st.download_button("Download PDF report", pdf, file_name="camfn_voice_report.pdf", mime="application/pdf")
        else:
            st.error(result.get("error"))

# ----------------------------------------------------------------- EEG tab --
with tabs[3]:
    st.subheader("EEG trace viewer + seizure classifier")
    st.caption("Real checkpoint: epilepsy_eeg.pt — held-out test accuracy 98.8% (n=1725) on the Epileptic Seizure "
               "Recognition corpus, unaffected by the subject-leakage fix (no recoverable subject ID in this "
               "corpus). Upload an .edf (first channel, first 178-sample window) or a 178-value CSV row.")
    upload = st.file_uploader("Upload .edf or .csv (single row of EEG samples)", type=["edf", "csv"], key="eeg_upl")
    if upload is not None:
        mne = _try_import_mne() if upload.name.lower().endswith(".edf") else None
        if mne is not None:
            tmp_path = Path("/tmp") / upload.name
            tmp_path.write_bytes(upload.read())
            raw = mne.io.read_raw_edf(str(tmp_path), preload=True, verbose="ERROR")
            fs = raw.info["sfreq"]
            data = raw.get_data(picks=[0])[0][: int(fs * 1.0)]  # ~1s from channel 0
            fig = go.Figure(go.Scatter(y=data, mode="lines"))
            fig.update_layout(title=f"Channel 0 (fs={fs} Hz)", height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width='stretch')
            result = inference.predict_eeg_window(data.tolist(), fs=fs, already_178=False)
        else:
            import csv as _csv
            row = next(_csv.reader(io.StringIO(upload.read().decode())), None)
            if not row:
                st.error("That CSV appears to be empty — nothing to load.")
                result = {"trained": False, "error": "empty CSV"}
            else:
                try:
                    samples = [float(v) for v in row if v.strip()]
                except ValueError as e:
                    st.error(f"Couldn't parse a numeric value from that CSV row: {e}")
                    samples = []
                if not samples:
                    result = {"trained": False, "error": "no numeric values found in that row"}
                else:
                    fig = go.Figure(go.Scatter(y=samples, mode="lines"))
                    fig.update_layout(title="Uploaded window", height=280, margin=dict(t=40, b=10))
                    st.plotly_chart(fig, width='stretch')
                    result = inference.predict_eeg_window(samples, already_178=(len(samples) == 178))

        if result.get("trained"):
            st.plotly_chart(_proba_bar(result["labels"], [result["probabilities"][l] for l in result["labels"]], "P(class)"), width='stretch')
            if result["predicted"] == "seizure":
                st.error(f"Predicted: **{result['predicted']}**")
            else:
                st.success(f"Predicted: **{result['predicted']}**")
            st.caption(result["note"])
            pdf = generate_report_pdf(result, "EEG — seizure screening", held_out_accuracy=HELD_OUT_ACC["eeg"])
            st.download_button("Download PDF report", pdf, file_name="camfn_eeg_report.pdf", mime="application/pdf")
        else:
            st.error(result.get("error"))

# --------------------------------------------------------------- Joint tab --
with tabs[4]:
    st.subheader("Joint CAMFN fusion (dynamic gated fusion across whichever modalities you provide)")
    st.caption(
        "Real checkpoint: camfn_joint.pt — 5-way HC/AD/PD/Epilepsy/Comorbid, held-out test accuracy 65.9%±19.1% "
        "over 3 seeds (n≈3021, subject-grouped-per-cohort split), macro-F1 0.50 (AD recall is very unstable "
        "across seeds — combined-cohort class imbalance, see docs/RESULTS.md). **A fusion ablation shows a "
        "trivial masked-mean baseline beats this model on every metric we measure** (accuracy, macro-F1, AUC, "
        "and stability — see paper §VI) — CAMFN's cross-attention/gating has no real "
        "cross-modal signal to exploit here, since **no sample used here ever has more than one real "
        "modality** (see docs/DATA_SOURCES.md). This tab demonstrates the missing-modality-robust routing "
        "mechanism, not a validated multi-modal diagnostic benefit."
    )
    jc1, jc2 = st.columns(2)
    with jc1:
        mri_up = st.file_uploader("MRI slice (optional)", type=["jpg", "jpeg", "png"], key="joint_mri")
        motor_up = st.file_uploader("Motor drawing (optional)", type=["jpg", "jpeg", "png"], key="joint_motor")
    with jc2:
        use_voice = st.checkbox("Include voice features (uses values from Voice tab)")
        eeg_up = st.file_uploader("EEG csv row (optional, 178 values)", type=["csv"], key="joint_eeg")

    if st.button("Run joint fusion"):
        mri_bytes = mri_up.read() if mri_up else None
        motor_bytes = motor_up.read() if motor_up else None
        voice_feats = values if (use_voice and values) else None
        eeg_samples = None
        eeg_parse_error = None
        if eeg_up is not None:
            import csv as _csv
            row = next(_csv.reader(io.StringIO(eeg_up.read().decode())), None)
            if not row:
                eeg_parse_error = "That EEG CSV appears to be empty — ignoring it."
            else:
                try:
                    eeg_samples = [float(v) for v in row if v.strip()]
                except ValueError as e:
                    eeg_parse_error = f"Couldn't parse a numeric value from the EEG CSV: {e}"
                if eeg_samples is not None and not eeg_samples:
                    eeg_parse_error = "No numeric values found in the EEG CSV row."
                    eeg_samples = None
        if eeg_parse_error:
            st.error(eeg_parse_error)

        result = inference.predict_joint(mri_bytes=mri_bytes, motor_bytes=motor_bytes, voice_features=voice_feats, eeg_samples=eeg_samples)
        if result.get("trained") and "error" not in result:
            st.plotly_chart(_proba_bar(result["labels"], [result["probabilities"][l] for l in result["labels"]], "P(diagnosis class)"), width='stretch')
            st.success(f"Predicted: **{result['predicted']}**")
            st.write("Modalities present this case:", result["modalities_present"])
            st.plotly_chart(
                _proba_bar(list(result["fusion_gates"].keys()), list(result["fusion_gates"].values()), "Dynamic gated-fusion weights (explainability)"),
                width='stretch',
            )
            st.caption(result["note"])
            pdf = generate_report_pdf(result, "Joint CAMFN fusion", held_out_accuracy=HELD_OUT_ACC["joint"])
            st.download_button("Download PDF report", pdf, file_name="camfn_joint_report.pdf", mime="application/pdf")
        else:
            st.error(result.get("error", "no modality supplied"))
