"""Diagnostic PDF report generator (spec section 4: "Automated PDF report
generator detailing class probabilities, confidence intervals, and salient
biomarker features").

Takes a real prediction dict from app.backend.inference and renders it —
no numbers are invented here; if a field wasn't in the prediction, it isn't
in the report. "Confidence interval" here means a simple Wilson-ish
uncertainty band derived from softmax probability + the model's known
held-out test accuracy (see docs/RESULTS.md), explicitly labeled as such
rather than a real per-prediction statistical CI (which would need
calibration data this skeleton doesn't have).
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name="Small", fontSize=8, textColor=colors.grey))


def generate_report_pdf(
    prediction: dict,
    modality_label: str,
    patient_ref: str = "N/A",
    held_out_accuracy: float | None = None,
) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    elements = []

    elements.append(Paragraph("CAMFN Diagnostic Support Report", STYLES["Title"]))
    elements.append(Paragraph(
        "RESEARCH USE ONLY -- NOT A MEDICAL DEVICE. Not validated for clinical decision-making.",
        ParagraphStyle(name="Warn", parent=STYLES["Normal"], textColor=colors.red, fontSize=10, spaceAfter=10),
    ))
    elements.append(Paragraph(f"Generated: {datetime.now(timezone.utc).isoformat()}Z", STYLES["Small"]))
    elements.append(Paragraph(f"Reference / case ID: {patient_ref}", STYLES["Small"]))
    elements.append(Paragraph(f"Modality/task: {modality_label}", STYLES["Small"]))
    elements.append(Spacer(1, 0.2 * inch))

    if not prediction.get("trained", False):
        elements.append(Paragraph(f"No prediction available: {prediction.get('error', 'unknown error')}", STYLES["Normal"]))
        doc.build(elements)
        return buf.getvalue()

    elements.append(Paragraph("Predicted class: " + str(prediction.get("predicted", "N/A")), STYLES["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))

    proba = prediction.get("probabilities", {})
    data = [["Class", "Probability", "Approx. uncertainty band*"]]
    for cls, p in proba.items():
        if held_out_accuracy is not None:
            band = f"± {(1 - held_out_accuracy):.2f} (model-level, not per-sample)"
        else:
            band = "n/a"
        data.append([cls, f"{p:.4f}", band])
    table = Table(data, colWidths=[1.8 * inch, 1.5 * inch, 2.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a55")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(Paragraph(
        "* Uncertainty band is derived from the model's held-out test accuracy "
        "(docs/RESULTS.md), not a calibrated per-prediction confidence interval.",
        STYLES["Small"],
    ))

    if "fusion_gates" in prediction:
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("Cross-modal fusion gate activations (explainability)", STYLES["Heading3"]))
        gate_data = [["Modality", "Gate weight", "Present this case?"]]
        present = set(prediction.get("modalities_present", []))
        for m, g in prediction["fusion_gates"].items():
            gate_data.append([m, f"{g:.3f}", "yes" if m in present else "no (placeholder used)"])
        gate_table = Table(gate_data, colWidths=[1.8 * inch, 1.5 * inch, 2.8 * inch])
        gate_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a55")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        elements.append(gate_table)

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("Notes", STYLES["Heading3"]))
    elements.append(Paragraph(prediction.get("note", ""), STYLES["Normal"]))

    doc.build(elements)
    return buf.getvalue()
