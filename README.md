# Motor Imagery EEG Classifier — Preprocessing & Modeling Pipeline

**Project:** Deep Learning-Based Motor Imagery EEG Classification for a Simulated Real-Time Brain–Computer Interface  
**Scope:** Person 1 (EEG Data Loading, Preprocessing, Validation & Modeling Handoff)  
**Dataset:** BCI Competition IV Dataset 2a (9 subjects, 22 EEG channels, 250 Hz, 4-class motor imagery)  
**Status:** Parts 0–6 Completed & Fully Verified. Handoff package ready for Person 2 (Modeling).

---

## 1. Quickstart: Loading Data for Model Training (Person 2)

Person 1 provides a clean, validated data loading interface in [`src/load_data.py`](file:///c:/Users/giris/Desktop/Sem5/MINI_Project/Motor%20Imagery%20EEG_Classifier/src/load_data.py):

```python
from src.load_data import load_split_data, get_class_names, get_channel_names

# Load canonical full-epoch arrays (N, 22, 1000) - 0.0s to 4.0s
X_train, y_train, mask_train = load_split_data('train', normalized=True, window='full')
X_val,   y_val,   mask_val   = load_split_data('val',   normalized=True, window='full')
X_test,  y_test,  mask_test  = load_split_data('test',  normalized=True, window='full')

# Load recommended central-window arrays (N, 22, 750) - 0.5s to 3.5s (Transient-Free ERD)
X_train_c, y_train, _ = load_split_data('train', normalized=True, window='central')
X_val_c,   y_val,   _ = load_split_data('val',   normalized=True, window='central')
X_test_c,  y_test,  _ = load_split_data('test',  normalized=True, window='central')

# Load clean-only trials (excluding the 15 Code 1023 visual artifact trials)
X_tr_clean, y_tr_clean, _ = load_split_data('train', clean_only=True)
```

See [data/processed/README.md](file:///c:/Users/giris/Desktop/Sem5/MINI_Project/Motor%20Imagery%20EEG_Classifier/data/processed/README.md) for the detailed handoff guide and API specifications.

---

## 2. Directory Structure

```text
Motor Imagery EEG_Classifier/
├── data/
│   ├── raw/                 # Original BCI IV 2a GDF and MAT files (18 GDF + 18 MAT, untouched)
│   └── processed/           # Processed datasets, split archives, and teammate handoff
│       ├── A01T_epochs.npz                  # Part 3 raw extracted epochs (288, 22, 1000)
│       ├── A01T_epochs_filtered.npz         # Part 4 bandpass-filtered epochs (8-30 Hz)
│       ├── A01T_split_data.npz              # Part 5 & 6 packaged split arrays (Train/Val/Test)
│       ├── A01T_split_metadata.json         # Complete metadata, split indices & normalization params
│       └── README.md                        # Complete Person 2 handoff guide
├── outputs/
│   ├── figures/             # 300 DPI visualizations (raw waveforms, PSD attenuation)
│   └── reports/             # Audit and completion reports (Markdown and PDF)
│       ├── Part_0_Completion_Report.md (.pdf)
│       ├── Parts_1_2_Completion_Report.md (.pdf)
│       ├── Part_3_Completion_Report.md (.pdf)
│       ├── Part_4_Technical_Verification.md
│       ├── Part_5_Completion_Report.md (.pdf)
│       ├── Part_5_Followup_Audit.md
│       └── Part_6_Completion_Report.md (.pdf)
├── src/
│   ├── inspect_dataset.py   # Dataset discovery and environment verification
│   ├── download_dataset.py  # Automated GDF archive downloader
│   ├── plot_raw_eeg.py      # Raw EEG plotting utility
│   ├── extract_epochs.py    # Event mapping and epoch extraction
│   ├── preprocess.py        # 8-30 Hz bandpass filtering and PSD calculation
│   ├── split_data.py        # Stratified splitting and zero-leakage normalization
│   ├── load_data.py         # Teammate data loading interface
│   ├── validate_handoff.py  # 10-point integration-readiness check script
│   └── generate_part_*_pdf.py # ReportLab PDF report generators
├── requirements.txt         # Core dependencies
└── README.md
```

---

## 3. Running Integration & Handoff Checks

To verify data integrity, loading modes, zero-leakage normalization, and model tensor formats:
```bash
python src/validate_handoff.py
```

---

## 4. Pipeline Summary & Verification Status

| Pipeline Stage | Build Target | Output Deliverables | Verified Status |
| :--- | :--- | :--- | :---: |
| **Part 0** | Workspace & Dataset Availability | 18 GDF + 18 MAT files verified in `data/raw/` | **PASS** |
| **Part 1** | Inspect One Recording (`A01T.gdf`) | $f_s = 250\ \text{Hz}$, 22 EEG + 3 EOG channels, 603 annotations | **PASS** |
| **Part 2** | Plot Raw EEG | Multi-channel microvolt plot (`outputs/figures/raw_eeg_plot.png`) | **PASS** |
| **Part 3** | Event Mapping & Epoching | 288 epochs, shape `(288, 22, 1000)`, 4 balanced classes (72/class) | **PASS** |
| **Part 4** | 8–30 Hz Filtering & Artifact Audit | Butterworth zero-phase filter, 99.8% low-drift attenuation, 15 mask flags | **PASS WITH CAVEATS** |
| **Part 5** | Split Strategy & Normalization | 75% Train (216), 12.5% Val (36), 12.5% Test (36), zero data leakage | **PASS** |
| **Part 6** | Packaging & Integration Readiness | `load_data.py`, `validate_handoff.py`, `data/processed/README.md` | **PASS (VERIFIED)** |
