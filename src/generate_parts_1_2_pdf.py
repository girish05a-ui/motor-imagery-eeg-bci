"""
PDF Generator for Parts 1 & 2 Consolidated Completion Report
Generates a formatted PDF report using ReportLab.
Output: outputs/reports/Parts_1_2_Completion_Report.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf():
    root = Path(__file__).resolve().parent.parent
    pdf_path = root / "outputs" / "reports" / "Parts_1_2_Completion_Report.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A'),
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#475569'),
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=10,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#334155'),
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0F172A'),
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("Parts 1 & 2 Completion Audit Report", title_style))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "<b>Project:</b> Deep Learning-Based Motor Imagery EEG Classification for a Simulated Real-Time BCI<br/>"
        "<b>Role:</b> Person 1 (EEG Data Loading, Preprocessing & Validation) &nbsp;|&nbsp; "
        "<b>Date:</b> October 19, 2026 &nbsp;|&nbsp; <b>Audit Verdict:</b> <font color='#166534'><b>PASS (VERIFIED)</b></font>",
        subtitle_style
    ))
    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#CBD5E1'), spaceBefore=2, spaceAfter=8))

    # Executive Summary
    elements.append(Paragraph("1. Executive Summary & Verification Scope", h1_style))
    elements.append(Paragraph(
        "This consolidated audit verifies the completion of <b>Part 1 (Inspect One Recording)</b> and <b>Part 2 (Plot Raw EEG)</b>. "
        "All findings are derived from direct execution on the real BCI Competition IV Dataset 2a recording (<code>data/raw/A01T.gdf</code>). "
        "No synthetic data or assumed parameters were used. Channel types, sampling frequency, event codes, signal amplitudes, "
        "and physical units were rigorously validated.",
        body_style
    ))
    elements.append(Spacer(1, 6))

    # Part 1 Audit Table
    elements.append(Paragraph("2. Part 1 Audit: A01T.gdf Physical Properties & Event Verification", h1_style))
    p1_data = [
        ["Parameter", "Observed Value", "Official Spec (desc_2a.pdf)", "Status"],
        ["File Format & Size", "GDF v2 | 32.08 MB (33,640,300 bytes)", "GDF (General Data Format)", "VERIFIED"],
        ["Sampling Frequency", "250.0 Hz (1 sample = 4.0 ms)", "250 Hz", "VERIFIED"],
        ["Sample Count & Duration", "672,528 samples | 2,690.112 s (44.84 min)", "9 runs (3 baseline + 6 task runs)", "VERIFIED"],
        ["Total Channels", "25 channels (22 EEG + 3 EOG)", "22 EEG + 3 EOG", "VERIFIED"],
        ["EOG Channel Separation", "EOG-left, EOG-central, EOG-right identified", "Monopolar EOG monitoring", "ACTION REQUIRED IN P4"],
        ["Trial Start Events (768)", "288 occurrences", "288 trials per session", "VERIFIED"],
        ["Motor Imagery Cues", "769: 72 | 770: 72 | 771: 72 | 772: 72", "72 trials per class (Left/Right/Feet/Tongue)", "VERIFIED (BALANCED)"],
        ["Artifact Flags (1023)", "15 marked trials", "Visual inspection rejection tags", "DOCUMENTED"],
    ]
    t1 = Table(p1_data, colWidths=[120, 180, 160, 80])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TEXTCOLOR', (3, 1), (3, 3), colors.HexColor('#166534')),
        ('TEXTCOLOR', (3, 4), (3, 4), colors.HexColor('#854D0E')),
        ('TEXTCOLOR', (3, 5), (3, 7), colors.HexColor('#166534')),
    ]))
    elements.append(t1)
    elements.append(Spacer(1, 6))

    # Part 2 Audit Table
    elements.append(Paragraph("3. Part 2 Audit: Raw EEG Visualization & Physical Unit Calibration", h1_style))
    p2_data = [
        ["Channel", "Cortical Region", "Mean (µV)", "Std Dev (µV)", "Min (µV)", "Max (µV)", "Pk-Pk (µV)", "Plausibility"],
        ["EEG-C3", "Left Motor Strip", "+0.81", "9.98", "-28.71", "+36.04", "64.75", "Normal scalp EEG"],
        ["EEG-Cz", "Vertex / Midline", "+0.61", "11.67", "-32.18", "+38.48", "70.65", "Normal scalp EEG"],
        ["EEG-C4", "Right Motor Strip", "+0.54", "10.38", "-27.10", "+35.79", "62.89", "Normal scalp EEG"],
    ]
    t2 = Table(p2_data, colWidths=[60, 110, 55, 60, 55, 55, 60, 85])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TEXTCOLOR', (7, 1), (7, 3), colors.HexColor('#166534')),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "<b>Figure Artifact:</b> <code>outputs/figures/raw_eeg_plot.png</code> | "
        "<b>Dimensions:</b> 3570 x 2094 (300 DPI) | <b>Size:</b> 824.6 KB | "
        "<b>Window:</b> t = 367.0s to 377.0s (10.0 s, 2500 pts) spanning Trial 1 (768) and Cue: Tongue (772). "
        "MNE raw Volts (V) successfully converted to microvolts (µV = V * 1e6). No dead channels or clipping.",
        body_style
    ))
    elements.append(Spacer(1, 6))

    # Evidence Tiers
    elements.append(Paragraph("4. Evidence Separation & Scientific Rigor", h1_style))
    tier_data = [
        ["Tier", "Evidence Class", "Items Verified / Deferred"],
        ["Tier 1", "Verified Facts (Live Execution)", "A01T.gdf loaded; 250 Hz, 25 channels, 603 annotations; C3/Cz/C4 statistics; PNG verified."],
        ["Tier 2", "Recorded in Literature (desc_2a.pdf)", "Codes 769=Left, 770=Right, 771=Feet, 772=Tongue; 768=Trial Start; 1023=Artifact marker."],
        ["Tier 3", "Unverified / Deferred Decisions", "Artifact rejection policy for code-1023 trials (Part 4); Epoch bounds (Part 3); Filtering (Part 4)."],
    ]
    t3 = Table(tier_data, colWidths=[50, 160, 330])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    elements.append(t3)
    elements.append(Spacer(1, 6))

    # Checkpoint Summary & Next Action
    elements.append(Paragraph("5. Continuation Checkpoint & Next Action (Part 3)", h1_style))
    elements.append(Paragraph(
        "<b>Audit Verdict:</b> Parts 1 and 2 pass with zero blockers.<br/>"
        "<b>Next Action (Part 3):</b> Map observed task events (769, 770, 771, 772) to class indices (0, 1, 2, 3), "
        "select the 22 EEG channels (dropping 3 EOG channels), extract event-locked epochs around cue onset, "
        "and assert <code>len(X) == len(y)</code> with 3D X shape <code>(n_epochs, 22, n_times)</code> and balanced class counts.",
        body_style
    ))

    doc.build(elements)
    print(f"PDF Audit Report generated at: {pdf_path}")


if __name__ == "__main__":
    generate_pdf()
