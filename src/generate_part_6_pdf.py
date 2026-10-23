"""
PDF Generator for Part 6 Completion Report
Role: Person 1 (EEG Data Loading, Preprocessing, and Validation)
Project: Deep Learning-Based Motor Imagery EEG Classification for a Simulated Real-Time BCI

Generates a formatted PDF report using ReportLab.
Output: outputs/reports/Part_6_Completion_Report.pdf
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
    pdf_path = root / "outputs" / "reports" / "Part_6_Completion_Report.pdf"

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
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0F172A'),
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#475569'),
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=6,
        spaceAfter=2,
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#334155'),
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor('#0F172A'),
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("Part 6 Completion Report: Packaging & Integration Readiness", title_style))
    elements.append(Spacer(1, 2))
    elements.append(Paragraph(
        "<b>Project:</b> Deep Learning-Based Motor Imagery EEG Classification for a Simulated Real-Time BCI<br/>"
        "<b>Role:</b> Person 1 (EEG Data Loading, Preprocessing & Validation) &nbsp;|&nbsp; "
        "<b>Date:</b> October 19, 2026 &nbsp;|&nbsp; <b>Verdict:</b> <font color='#166534'><b>PASS (10/10 CHECKS VERIFIED)</b></font>",
        subtitle_style
    ))
    elements.append(Spacer(1, 2))
    elements.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor('#CBD5E1'), spaceBefore=2, spaceAfter=4))

    # Executive Summary
    elements.append(Paragraph("1. Executive Summary & Handoff Scope", h1_style))
    elements.append(Paragraph(
        "Part 6 establishes the official data packaging, programmatic data loader interface, teammate documentation, and integration "
        "validation for Person 1. The handoff package delivers the 288 Subject 1 trials partitioned into <b>Train (75.0%, 216)</b>, "
        "<b>Validation (12.5%, 36)</b>, and <b>Test (12.5%, 36)</b> splits with 100% class balance (54/9/9 per class). "
        "Channel-wise z-score normalization was fitted strictly on training data to ensure zero test leakage. "
        "The interface module (<code>src/load_data.py</code>), integration validator (<code>src/validate_handoff.py</code>), and "
        "teammate documentation (<code>data/processed/README.md</code>) have been verified end-to-end with 10/10 integration checks passing.",
        body_style
    ))
    elements.append(Spacer(1, 3))

    # Table 1: Data Specifications & Tensor Contracts
    elements.append(Paragraph("2. Data Specifications & Tensor Contracts", h1_style))
    spec_data = [
        ["Parameter", "Specification / Setting", "Description & Handoff Contract"],
        ["Subject & Session", "Subject 1 (A01), Session T", "288 motor imagery trials across 6 runs (48 trials/run)"],
        ["Sampling Frequency", "250.0 Hz (Delta t = 4.0 ms)", "Nyquist frequency 125.0 Hz; 1000 samples = 4.0s epoch"],
        ["Channels (Axis 1)", "22 scalp EEG electrodes", "Strictly ordered: EEG-Fz to EEG-16 (C3=7, Cz=9, C4=11; 3 EOG excluded)"],
        ["Target Classes", "4 balanced classes (72/class)", "0: Left Hand, 1: Right Hand, 2: Both Feet, 3: Tongue"],
        ["Split Architecture", "75% / 12.5% / 12.5% (Seed 42)", "Train: 216 (54/class) | Val: 36 (9/class) | Test: 36 (9/class)"],
        ["Normalization", "Channel-wise z-score", "Fitted strictly on X_train (mean=0, std=1); frozen params for Val/Test"],
        ["Window Views", "Full (1000) vs Central (750)", "Full: 0.0-4.0s (N,22,1000) | Central: 0.5-3.5s (N,22,750, transient-free)"],
        ["Artifact Flags", "15 trials (Code 1023)", "13 in Train (6.0%), 2 in Val (5.6%), 0 in Test (0.0%); clean mode: 203/34/36"],
    ]
    ts = Table(spec_data, colWidths=[95, 135, 310])
    ts.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 6.8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    elements.append(ts)
    elements.append(Spacer(1, 3))

    # Table 2: 10-Point Integration-Readiness Checklist
    elements.append(Paragraph("3. Integration-Readiness Verification Matrix (All 10/10 Checks Passed)", h1_style))
    val_data = [
        ["#", "Check Item", "Target Specification", "Empirical Result (python src/validate_handoff.py)", "Status"],
        ["1", "Metadata Schema", "Required keys present; counts 216/36/36; sfreq=250", "All 13 keys verified; counts match; sfreq 250.0 confirmed", "PASS"],
        ["2", "Channel Mapping", "22 scalp electrodes; C3, Cz, C4 mapped; zero EOG", "Order verified: Fz=0, C3=7, Cz=9, C4=11; EOG absent", "PASS"],
        ["3", "Full-Epoch Shapes", "(216, 22, 1000), (36, 22, 1000), (36, 22, 1000)", "Shapes and float32 data types verified bitwise", "PASS"],
        ["4", "Central-Window Shapes", "(216, 22, 750), (36, 22, 750), (36, 22, 750)", "Bitwise match to sample slice [125:875] confirmed", "PASS"],
        ["5", "Class Balance", "54/class Train, 9/class Val, 9/class Test", "Exactly 54/54/54/54, 9/9/9/9, and 9/9/9/9 verified", "PASS"],
        ["6", "Artifact Tracking", "15 Code 1023 trials tracked (13/2/0); clean mode", "13/2/0 counts confirmed; clean mode yields 203/34/36", "PASS"],
        ["7", "Zero Leakage", "Train centered at 0 and std=1; Val/Test shift", "Train: mu<1e-6, std=1.000; Val/Test: natural shift", "PASS"],
        ["8", "Numerical Health", "Zero NaN, Zero Inf across all arrays", "100% finite values confirmed in all partitions", "PASS"],
        ["9", "Disjoint & Complete", "Zero partition overlap; union equals 288", "Pairwise intersections empty; union equals 288 trials", "PASS"],
        ["10", "Model Compatibility", "2D Conv (B, 22, T), Spatial (B, 1, 22, T), LSTM (B, T, 22)", "All common DL input transforms execute cleanly", "PASS"],
    ]
    tv = Table(val_data, colWidths=[15, 85, 160, 235, 45])
    tv.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 6.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (4, 0), (4, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6),
        ('TOPPADDING', (0, 0), (-1, -1), 1.6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TEXTCOLOR', (4, 1), (4, -1), colors.HexColor('#166534')),
    ]))
    elements.append(tv)
    elements.append(Spacer(1, 3))

    # Deep Learning Model Compatibility & Quickstart
    elements.append(Paragraph("4. Deep Learning Model Input Formatting Support", h1_style))
    elements.append(Paragraph(
        "<code>src/load_data.py</code> provides direct programmatic access to arrays tailored for downstream deep learning models:<br/>"
        "• <b>2D Temporal Conv:</b> <code>load_split_data('train', normalized=True, window='full')</code> &rarr; <code>(216, 22, 1000)</code><br/>"
        "• <b>Spatial-Temporal Conv (EEGNet / ShallowConvNet):</b> Add spatial dim &rarr; <code>(216, 1, 22, 1000)</code> or <code>(216, 1, 22, 750)</code><br/>"
        "• <b>Recurrent / LSTM:</b> Transpose axes (0, 2, 1) &rarr; <code>(216, 1000, 22)</code> or <code>(216, 750, 22)</code><br/>"
        "• <b>PyTorch Wrapper:</b> Optional <code>EEGDataset(mode='train', spatial_dim=True)</code> returns <code>(Tensor(1, 22, T), Tensor(y))</code>.",
        body_style
    ))
    elements.append(Spacer(1, 3))

    # Critical Technical Caveats
    elements.append(Paragraph("5. Critical Technical Caveats for Modeling & Integration", h1_style))
    elements.append(Paragraph(
        "<b>1. Offline Non-Causal Zero-Phase Filtering:</b> Offline filtering used <code>scipy.signal.filtfilt</code> (8th-order effective "
        "Butterworth, 8-30 Hz, 0&deg; phase distortion). This preserves peak ERD alignment for offline model benchmarking. "
        "<b>Person 3 Warning:</b> This filter requires future samples and cannot be used sample-by-sample in simulated live streaming. "
        "Streaming requires a causal filter, introducing 20-50 ms group delay.<br/>"
        "<b>2. Zero Test Artifacts:</b> Under stratified sampling (seed 42), all 15 <code>Code 1023</code> artifact trials fell into Train (13) "
        "and Validation (2). The held-out test set contains <b>0 artifact trials</b>. Held-out test accuracy evaluates performance on pristine "
        "signals only and cannot benchmark model degradation on corrupted trials. Person 2 can evaluate artifact robustness via clean vs. full training.<br/>"
        "<b>3. Filter Boundary Edge Transients:</b> Canonical full epochs (1000 samples) contain edge transients within the first and last 200 ms "
        "(up to 32.61 &mu;V vs. continuous filtering). The recommended central window (<code>[125:875]</code>, 750 samples, 0.5-3.5s) exhibits "
        "mean diff 0.0002 &mu;V and peak max 0.0366 &mu;V, targeting peak sensorimotor ERD.",
        body_style
    ))
    elements.append(Spacer(1, 3))

    # Deliverables & Sign-Off
    elements.append(Paragraph("6. Deliverables Inventory & Person 1 Final Sign-Off", h1_style))
    elements.append(Paragraph(
        "• <b>Packaged Split Arrays:</b> <code>data/processed/A01T_split_data.npz</code> (82.3 MB, verified bitwise).<br/>"
        "• <b>Split Metadata:</b> <code>data/processed/A01T_split_metadata.json</code> (9.5 KB, verified with sfreq: 250.0).<br/>"
        "• <b>Data Loading Interface:</b> <code>src/load_data.py</code> (10.5 KB, clean self-test pass).<br/>"
        "• <b>Integration Test Runner:</b> <code>src/validate_handoff.py</code> (9.7 KB, 10/10 checks verified).<br/>"
        "• <b>Teammate Documentation:</b> <code>data/processed/README.md</code> &amp; root <code>README.md</code>.<br/>"
        "• <b>Completion Reports:</b> <code>outputs/reports/Part_6_Completion_Report.md</code> &amp; <code>.pdf</code>.<br/>"
        "<b>Person 1 Pipeline Verdict: COMPLETE & VERIFIED.</b> Ready for Person 2 (Modeling) & Person 3 (Simulated Streaming).",
        body_style
    ))

    doc.build(elements)
    print(f"PDF report generated: {pdf_path.resolve()}")


if __name__ == '__main__':
    generate_pdf()
