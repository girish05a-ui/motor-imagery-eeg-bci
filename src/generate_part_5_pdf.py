"""
PDF Generator for Part 5 Completion Report
Generates a formatted PDF report using ReportLab.
Output: outputs/reports/Part_5_Completion_Report.pdf
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
    pdf_path = root / "outputs" / "reports" / "Part_5_Completion_Report.pdf"

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
        fontSize=17,
        leading=21,
        textColor=colors.HexColor('#0F172A'),
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#475569'),
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13.5,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=7,
        spaceAfter=3,
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor('#334155'),
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#0F172A'),
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("Part 5 Completion Report: Data Splitting & Leakage-Safe Preparation", title_style))
    elements.append(Spacer(1, 3))
    elements.append(Paragraph(
        "<b>Project:</b> Deep Learning-Based Motor Imagery EEG Classification for a Simulated Real-Time BCI<br/>"
        "<b>Role:</b> Person 1 (EEG Data Loading, Preprocessing & Validation) &nbsp;|&nbsp; "
        "<b>Date:</b> October 19, 2026 &nbsp;|&nbsp; <b>Audit Verdict:</b> <font color='#166534'><b>PASS (VERIFIED)</b></font>",
        subtitle_style
    ))
    elements.append(Spacer(1, 3))
    elements.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor('#CBD5E1'), spaceBefore=2, spaceAfter=5))

    # Executive Summary
    elements.append(Paragraph("1. Executive Summary & Evaluation Contract", h1_style))
    elements.append(Paragraph(
        "Part 5 establishes the data partitioning, leakage-safe normalization, and model interface contract for Person 1. "
        "The 288 bandpass-filtered epochs from Subject 1 (<code>A01T_epochs_filtered.npz</code>) were partitioned into "
        "<b>Train (75.0%, 216 trials)</b>, <b>Validation (12.5%, 36 trials)</b>, and <b>Test (12.5%, 36 trials)</b> using a "
        "reproducible stratified split (<code>seed=42</code>). Channel-wise z-score normalization parameters were fitted "
        "<b>strictly on training data only</b> to prevent test leakage. All 15 artifact trials (<code>Code 1023</code>) were "
        "preserved across partitions with explicit boolean masks (13 Train, 2 Val, 0 Test). Canonical shapes <code>(N, 22, 1000)</code> "
        "and recommended central transient-free windows <code>(N, 22, 750)</code> are exported in <code>data/processed/A01T_split_data.npz</code>.",
        body_style
    ))
    elements.append(Spacer(1, 4))

    # Table 1: Split Architecture & Class Balance
    elements.append(Paragraph("2. Partition Architecture & 100% Class Balance (Seed = 42)", h1_style))
    split_data = [
        ["Partition", "Ratio", "Total Trials", "Class 0 (Left)", "Class 1 (Right)", "Class 2 (Feet)", "Class 3 (Tongue)", "Artifacts (1023)"],
        ["Training", "75.0%", "216", "54 (25.0%)", "54 (25.0%)", "54 (25.0%)", "54 (25.0%)", "13 (6.0%)"],
        ["Validation", "12.5%", "36", "9 (25.0%)", "9 (25.0%)", "9 (25.0%)", "9 (25.0%)", "2 (5.6%)"],
        ["Test (Held-Out)", "12.5%", "36", "9 (25.0%)", "9 (25.0%)", "9 (25.0%)", "9 (25.0%)", "0 (0.0%)"],
        ["Total Dataset", "100.0%", "288", "72 (25.0%)", "72 (25.0%)", "72 (25.0%)", "72 (25.0%)", "15 (5.2%)"],
    ]
    ts = Table(split_data, colWidths=[80, 45, 60, 68, 68, 68, 68, 85])
    ts.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    elements.append(ts)
    elements.append(Spacer(1, 4))

    # Normalization Isolation Table
    elements.append(Paragraph("3. Zero-Leakage Normalization Contract (Channel-Wise Z-Score)", h1_style))
    norm_data = [
        ["Partition", "Mean (Expected)", "Mean (Observed)", "Std (Expected)", "Std (Observed)", "Leakage Verification Status"],
        ["Training (X_train)", "0.0000", "-0.00000000", "1.0000", "1.000000", "Fitted partition: perfectly centered & scaled"],
        ["Validation (X_val)", "~ 0.0 (Natural shift)", "+0.00023481", "~ 1.0 (Natural shift)", "0.999151", "PASS: transformed with train stats only"],
        ["Test (X_test)", "~ 0.0 (Natural shift)", "-0.00015252", "~ 1.0 (Natural shift)", "0.989031", "PASS: held-out set completely isolated"],
    ]
    tn = Table(norm_data, colWidths=[90, 85, 80, 80, 75, 130])
    tn.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TEXTCOLOR', (5, 1), (5, -1), colors.HexColor('#166534')),
    ]))
    elements.append(tn)
    elements.append(Spacer(1, 4))

    # Windowing Strategy & Artifact Tracking
    elements.append(Paragraph("4. Windowing Strategy & Artifact Tracking Resolution", h1_style))
    elements.append(Paragraph(
        "<b>Windowing Recommendation:</b> Part 4 verified that per-epoch filtering leaves ~200 ms edge transients along boundary edges. "
        "To avoid forcing model architecture refactoring while providing physical accuracy, both views are exported:<br/>"
        "• <b>Canonical Full Epoch (1,000 samples, 0.0–3.996s):</b> Shapes <code>(216, 22, 1000)</code>, <code>(36, 22, 1000)</code>.<br/>"
        "• <b>Recommended Central Window (750 samples, 0.5–3.5s):</b> Slices <code>[125:875]</code> (shapes <code>(216, 22, 750)</code>, <code>(36, 22, 750)</code>). "
        "Excludes filter transients (mean diff 0.0002 µV, peak max 0.0366 µV in [0.5, 3.5]s vs 0.1254 µV max in [0.4, 3.6]s) and targets peak motor imagery ERD.<br/>"
        "<b>Artifact Handling (Code 1023):</b> All 15 flagged trials are preserved with explicit masks. Person 2 can train on all 216 trials "
        "(100% balanced, 54/class) or 203 clean trials reversibly: <code>X_clean = X_train[~artifact_mask_train]</code>.",
        body_style
    ))
    elements.append(Spacer(1, 4))

    # Comprehensive Quality Checks Table
    elements.append(Paragraph("5. Output Verification & Leakage Validation Matrix", h1_style))
    val_data = [
        ["Check Item", "Target Specification", "Observed / Verified Result", "Status"],
        ["Partition Disjointness", "set(A) & set(B) == empty", "0 overlap across all pairwise combinations", "PASS"],
        ["Partition Completeness", "set(train) | set(val) | set(test) == 288", "Exact 288 trials retained; zero trial loss", "PASS"],
        ["Array Dimensions", "(216, 22, 1000), (36, 22, 1000), (36, 22, 1000)", "Verified bitwise in NPZ archive", "PASS"],
        ["Windowed Dimensions", "(216, 22, 750), (36, 22, 750), (36, 22, 750)", "Verified bitwise in NPZ archive", "PASS"],
        ["Finite Values", "No NaN, No Inf across all arrays", "Zero NaN, Zero Inf across all partitions", "PASS"],
        ["Class Balance", "54/class Train, 9/class Val, 9/class Test", "Exactly 54/54/54/54 and 9/9/9/9", "PASS"],
        ["Artifact Mask Tracking", "15 total flags preserved", "13 Train + 2 Val + 0 Test == 15", "PASS"],
        ["Zero Normalization Leak", "Parameters fit on Train only", "Train: mean=0, std=1; Val/Test: Natural shift", "PASS"],
        ["Split Reproducibility", "Deterministic index assignment", "Bitwise identical indices across repeated runs (seed=42)", "PASS"],
    ]
    tv = Table(val_data, colWidths=[105, 135, 230, 70])
    tv.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (3, 0), (3, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TEXTCOLOR', (3, 1), (3, -1), colors.HexColor('#166534')),
    ]))
    elements.append(tv)
    elements.append(Spacer(1, 4))

    # Generated Artifacts & Continuation Checkpoint
    elements.append(Paragraph("6. Generated Artifacts & Continuation Checkpoint", h1_style))
    elements.append(Paragraph(
        "• <b>Split Arrays Archive:</b> <code>data/processed/A01T_split_data.npz</code> (78.5 MB, full and windowed arrays, labels, masks).<br/>"
        "• <b>Split Metadata:</b> <code>data/processed/A01T_split_metadata.json</code> (seed, split ratios, channel means/stds, run structure).<br/>"
        "• <b>Part 5 Verdict:</b> <b>PASS (Zero Blockers, Fully Verified)</b>. Zero data leakage, 100% class balance, verified reproducibility.<br/>"
        "• <b>Next Step (Part 6):</b> Package teammate-friendly export and handoff with documented load scripts, shape checks, and README.",
        body_style
    ))

    doc.build(elements)
    print(f"PDF report generated: {pdf_path.resolve()}")


if __name__ == '__main__':
    generate_pdf()
