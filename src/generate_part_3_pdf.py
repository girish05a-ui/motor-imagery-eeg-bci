"""
PDF Generator for Part 3 Completion Audit Report
Generates a formatted PDF report using ReportLab.
Output: outputs/reports/Part_3_Completion_Report.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf():
    root = Path(__file__).resolve().parent.parent
    pdf_path = root / "outputs" / "reports" / "Part_3_Completion_Report.pdf"

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

    elements = []

    # Title & Metadata
    elements.append(Paragraph("Part 3 Completion Audit Report", title_style))
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
        "Part 3 evaluated event mapping, channel isolation, and epoch extraction for <code>data/raw/A01T.gdf</code>. "
        "The continuous recording was converted into 288 event-locked 4.0-second epochs. "
        "The 22 scalp EEG channels were isolated and the 3 EOG channels were completely excluded. "
        "Array dimensions, labels, class balance (72 trials per class), zero silent trial loss, "
        "and successful array reloading were independently verified.",
        body_style
    ))
    elements.append(Spacer(1, 6))

    # Event Mapping & Class Balance Table
    elements.append(Paragraph("2. Event Mapping & Class Balance Verification", h1_style))
    map_data = [
        ["Event Code", "Class Index", "Target Class Name", "Cue Meaning in desc_2a.pdf", "Observed Count", "Status"],
        ["769", "0", "Left Hand", "Visual arrow pointing left", "72", "VERIFIED"],
        ["770", "1", "Right Hand", "Visual arrow pointing right", "72", "VERIFIED"],
        ["771", "2", "Both Feet", "Visual arrow pointing downward", "72", "VERIFIED"],
        ["772", "3", "Tongue", "Visual arrow pointing upward", "72", "VERIFIED"],
        ["Total", "—", "4-Class Motor Imagery", "288 task trials (100% balanced)", "288", "VERIFIED"],
    ]
    t1 = Table(map_data, colWidths=[65, 60, 95, 180, 80, 60])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TEXTCOLOR', (5, 1), (5, -1), colors.HexColor('#166534')),
    ]))
    elements.append(t1)
    elements.append(Spacer(1, 6))

    # Array Dimensions & Validation Checks Table
    elements.append(Paragraph("3. Output Validation & Technical Checks", h1_style))
    check_data = [
        ["Check Item", "Target Specification", "Observed / Verified Result", "Status"],
        ["X Dimensions", "3D: (n_epochs, n_channels, n_times)", "(288, 22, 1000)", "PASS"],
        ["y Dimensions", "1D: (n_epochs,)", "(288,)", "PASS"],
        ["Length Match", "len(X) == len(y)", "288 == 288", "PASS"],
        ["Channels Isolated", "22 EEG (Frontal/Central/Parietal/Occipital)", "22 channels (EEG-Fz to EEG-16)", "PASS"],
        ["Channels Excluded", "3 EOG channels dropped", "EOG-left, EOG-central, EOG-right", "PASS"],
        ["Sampling & Time Window", "fs=250 Hz, t=[0.0s, 3.996s], 4.0s duration", "1,000 samples per epoch (250 Hz * 4s)", "PASS"],
        ["Finite Values", "np.isfinite(X).all() == True", "Zero NaN, Zero Inf", "PASS"],
        ["Class Labels", "Subset of {0, 1, 2, 3}", "[0, 1, 2, 3] verified", "PASS"],
        ["Epoch Retention", "No silent trial loss", "Requested: 288, Retained: 288, Dropped: 0", "PASS"],
        ["Artifact File Persistence", "Saved arrays and metadata reload bitwise", "A01T_epochs.npz (11.1 MB) reloads cleanly", "PASS"],
    ]
    t2 = Table(check_data, colWidths=[110, 170, 190, 70])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TEXTCOLOR', (3, 1), (3, -1), colors.HexColor('#166534')),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 6))

    # Artifact Overlap Tracking
    elements.append(Paragraph("4. Artifact Tracking (Pre-Part 4 Audit)", h1_style))
    elements.append(Paragraph(
        "<b>Visual Artifact Overlap:</b> 15 trials coincide with the dataset creators' code-1023 artifact marks: "
        "<code>[21, 45, 52, 56, 67, 94, 97, 98, 104, 191, 214, 229, 254, 276, 287]</code>.<br/>"
        "<b>Retention Policy:</b> Zero trials were removed in Part 3. The 15 indices are formally tracked in "
        "<code>data/processed/A01T_epochs_metadata.json</code> so that artifact handling can be decided, audited, "
        "and documented as an explicit design choice in Part 4.",
        body_style
    ))
    elements.append(Spacer(1, 6))

    # Next Steps
    elements.append(Paragraph("5. Continuation Checkpoint & Next Action (Part 4)", h1_style))
    elements.append(Paragraph(
        "<b>Part 3 Verdict:</b> PASS (Zero Blockers).<br/>"
        "<b>Next Task (Part 4):</b> Implement filtering, artifact handling strategy, and signal quality checks "
        "(bandpass filter 8–30 Hz covering sensorimotor mu and beta rhythms, causality documentation, "
        "before/after PSD comparisons, and formal resolution of the 15 code-1023 trials).",
        body_style
    ))

    doc.build(elements)
    print(f"PDF Audit Report generated at: {pdf_path}")


if __name__ == "__main__":
    generate_pdf()
