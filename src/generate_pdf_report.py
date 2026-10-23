"""
PDF Generator for Part 0 Completion Report
Generates a formatted PDF report using ReportLab.
Output: outputs/reports/Part_0_Completion_Report.pdf
"""

import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf():
    root = Path(__file__).resolve().parent.parent
    pdf_path = root / "outputs" / "reports" / "Part_0_Completion_Report.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748B'),
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#0F172A'),
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("Part 0 Completion & Audit Report", title_style))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("<b>Project:</b> Deep Learning-Based Motor Imagery EEG Classification for a Simulated Real-Time BCI<br/>"
                              "<b>Role:</b> Person 1 (EEG Data Loading, Preprocessing & Validation) &nbsp;|&nbsp; "
                              "<b>Date:</b> October 19, 2026 &nbsp;|&nbsp; <b>Status:</b> Environment Ready / Download Pending", subtitle_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#CBD5E1'), spaceBefore=4, spaceAfter=10))

    # 1. Executive Summary
    elements.append(Paragraph("1. Executive Summary", h1_style))
    elements.append(Paragraph(
        "Part 0 establishes the operational baseline and workspace integrity for the EEG preprocessing pipeline. "
        "The project environment was audited on Windows 11 (Python 3.12.10). Required scientific libraries "
        "(MNE-Python, NumPy, SciPy, scikit-learn, Matplotlib, Pandas) were installed and verified. "
        "An inspection CLI and an automated dataset downloader were created and syntax-verified. "
        "<b>No synthetic data, mock signals, or invented event codes have been introduced</b>. "
        "Dataset acquisition from the official BBCI repository is the final gate before entering Part 1.",
        body_style
    ))
    elements.append(Spacer(1, 8))

    # 2. Environment Verification Table
    elements.append(Paragraph("2. Environment Verification", h1_style))
    env_data = [
        ["Component", "Target / Spec", "Observed Version", "Verification Status"],
        ["Python Runtime", ">= 3.10", "3.12.10 (win32)", "VERIFIED"],
        ["MNE-Python", ">= 1.13.0", "1.13.2", "VERIFIED"],
        ["NumPy", ">= 2.0.0", "2.4.4", "VERIFIED"],
        ["SciPy", ">= 1.14.0", "1.17.1", "VERIFIED"],
        ["scikit-learn", ">= 1.6.0", "1.8.0", "VERIFIED"],
        ["Matplotlib", ">= 3.9.0", "3.10.9", "VERIFIED"],
        ["Pandas", ">= 2.2.0", "3.0.3", "VERIFIED"],
        ["PyTorch", "Teammate Scope", "Not installed", "DEFERRED TO PERSON 2"],
    ]
    t = Table(env_data, colWidths=[110, 110, 140, 170])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TEXTCOLOR', (3, 1), (3, 7), colors.HexColor('#166534')),
        ('TEXTCOLOR', (3, 8), (3, 8), colors.HexColor('#854D0E')),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 8))

    # 3. Status Breakdown
    elements.append(Paragraph("3. Workspace & Lifecycle Status", h1_style))
    status_data = [
        ["Phase", "Component", "State", "Details"],
        ["Verified", "Core Python Env", "PASS", "MNE 1.13.2, NumPy 2.4.4, SciPy 1.17.1, sklearn 1.8.0, Matplotlib 3.10.9"],
        ["Verified", "Repo Scaffold", "PASS", "Created data/raw, data/processed, notebooks, outputs, src"],
        ["Verified", "inspect_dataset.py", "PASS", "Executed cleanly; reports exact environment & missing dataset blocker"],
        ["Syntax Checked", "download_dataset.py", "READY", "Validated syntax; awaiting execution to download 420 MB archive"],
        ["Pending", "Raw GDF Files", "PENDING", "0 files in data/raw/; requires running download_dataset.py"],
        ["Blocked", "Part 1 Inspection", "BLOCKED", "Waiting on A01T.gdf; no guessed filenames or mock data allowed"],
    ]
    st = Table(status_data, colWidths=[90, 120, 70, 250])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    elements.append(st)
    elements.append(Spacer(1, 8))

    # 4. Verified Commands & Output
    elements.append(Paragraph("4. Captured Verification Output (python src/inspect_dataset.py)", h1_style))
    raw_terminal = (
        "======================================================================\n"
        "PART 0: ENVIRONMENT INSPECTION\n"
        "======================================================================\n"
        "Python Executable: C:\\Users\\giris\\AppData\\Local\\Programs\\Python\\Python312\\python.exe\n"
        "Python Version:    3.12.10 (win32)\n"
        "  [OK]      MNE-Python                : 1.13.2\n"
        "  [OK]      NumPy                     : 2.4.4\n"
        "  [OK]      SciPy                     : 1.17.1\n"
        "  [OK]      scikit-learn              : 1.8.0\n"
        "  [OK]      Matplotlib                : 3.10.9\n"
        "  [OK]      Pandas                    : 3.0.3\n"
        "======================================================================\n"
        "PART 0: DATASET FILES DISCOVERY\n"
        "Primary raw dataset directory: .../data/raw | GDF files found: 0\n"
        "[BLOCKER] No BCI Competition IV Dataset 2a GDF files found.\n"
        "STATUS: PENDING DATASET DOWNLOAD (Exit Code: 0)\n"
        "======================================================================"
    )
    elements.append(Preformatted(raw_terminal, code_style))
    elements.append(Spacer(1, 8))

    # 5. Exact Next Action
    elements.append(Paragraph("5. Exact Next Steps to Enter Part 1", h1_style))
    elements.append(Paragraph(
        "<b>Step 1:</b> Run <code>python src/download_dataset.py</code> to fetch <code>BCICIV_2a_gdf.zip</code> (420 MB) "
        "and <code>true_labels.zip</code> into <code>data/raw/</code>.<br/>"
        "<b>Step 2:</b> Run <code>python src/inspect_dataset.py</code> to confirm all 18 GDF files are present.<br/>"
        "<b>Step 3:</b> Execute Part 1 to load <code>A01T.gdf</code> using <code>mne.io.read_raw_gdf</code> and inspect channels, sampling rate, and event markers.",
        body_style
    ))

    doc.build(elements)
    print(f"PDF Report generated successfully at: {pdf_path}")


if __name__ == "__main__":
    generate_pdf()
