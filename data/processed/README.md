# Person 1 Data Handoff & Interface Guide

**Project:** Deep Learning-Based Motor Imagery EEG Classification for a Simulated Real-Time BCI  
**From:** Person 1 (EEG Data Loading, Preprocessing, and Validation)  
**To:** Person 2 (Model Architecture & Deep Learning Training) & Person 3 (Simulated Streaming & 3D BCI Frontend)  
**Dataset:** BCI Competition IV Dataset 2a — Subject `A01`, Session `T` (Training)  
**Status:** Preprocessing and Data Splitting Validated (Parts 0–6 Complete)

---

## 1. Quickstart: Loading Preprocessed Arrays

Person 1 provides a clean Python module [`src/load_data.py`](file:///c:/Users/giris/Desktop/Sem5/MINI_Project/Motor%20Imagery%20EEG_Classifier/src/load_data.py) that handles loading, windowing, normalization, and artifact filtering:

```python
from src.load_data import load_split_data, get_channel_names, get_class_names

# 1. Canonical Full-Epoch Loading (N, 22, 1000), 0.0s to 4.0s relative to visual cue
X_train, y_train, mask_train = load_split_data('train', normalized=True, window='full')
X_val,   y_val,   mask_val   = load_split_data('val',   normalized=True, window='full')
X_test,  y_test,  mask_test  = load_split_data('test',  normalized=True, window='full')

print("Train shape:", X_train.shape)  # (216, 22, 1000) - 54 per class (100% balanced)
print("Val shape:  ", X_val.shape)    # (36, 22, 1000)  - 9 per class (100% balanced)
print("Test shape: ", X_test.shape)   # (36, 22, 1000)  - 9 per class (100% balanced)

# 2. Recommended Central-Window Loading (N, 22, 750), 0.5s to 3.5s (Transient-Free ERD)
X_train_c, y_train, _ = load_split_data('train', normalized=True, window='central')
X_val_c,   y_val,   _ = load_split_data('val',   normalized=True, window='central')
X_test_c,  y_test,  _ = load_split_data('test',  normalized=True, window='central')

print("Windowed Train shape:", X_train_c.shape)  # (216, 22, 750)

# 3. Clean-Only Mode (Excludes the 15 Code 1023 visual artifact trials)
X_tr_clean, y_tr_clean, _ = load_split_data('train', clean_only=True)
print("Clean Train shape:", X_tr_clean.shape)  # (203, 22, 1000)
```

Alternatively, you can load the arrays directly via standard `numpy`:

```python
import numpy as np

data = np.load('data/processed/A01T_split_data.npz')
X_train = data['X_train_norm']  # (216, 22, 1000)
y_train = data['y_train']       # (216,)
mask_train = data['artifact_mask_train']  # (216,) boolean
```

---

## 2. Dataset & Signal Specifications

| Parameter | Specification | Value / Description |
| :--- | :--- | :--- |
| **Subject & Session** | Subject 1 (`A01`), Session `T` | 288 total event-locked motor imagery trials |
| **Sampling Rate ($f_s$)** | $250.0\ \text{Hz}$ | Nyquist frequency $f_N = 125.0\ \text{Hz}$, $\Delta t = 4.0\ \text{ms}$ |
| **Channel Count** | 22 scalp EEG electrodes | Ordered according to standard 10–20 international system |
| **Channel Names** | `EEG-Fz` to `EEG-16` | Frontal, central, parietal, and occipital coverage (see Section 3) |
| **Excluded Channels** | 3 EOG channels | `EOG-left`, `EOG-central`, `EOG-right` (completely excluded from tensors) |
| **Data Types** | `float32` (X), `int64` (y), `bool` (mask) | Native compatibility with PyTorch `torch.float32` and `torch.int64` |

---

## 3. Channel Mapping & Electrode Order

The 22 scalp EEG channels are strictly ordered across axis 1 (`shape: (N, 22, T)`):

```text
Index  Channel    Cortical Region       Index  Channel    Cortical Region
-------------------------------------------------------------------------
 0     EEG-Fz     Frontal Midline        11    EEG-C4     Right Motor Cortex
 1     EEG-0      Frontal                12    EEG-8      Central-Parietal
 2     EEG-1      Frontal                13    EEG-9      Central-Parietal
 3     EEG-2      Frontal                14    EEG-10     Central-Parietal
 4     EEG-3      Frontal                15    EEG-11     Central-Parietal
 5     EEG-4      Frontal-Central        16    EEG-12     Central-Parietal
 6     EEG-5      Frontal-Central        17    EEG-13     Parietal
 7     EEG-C3     Left Motor Cortex      18    EEG-14     Parietal
 8     EEG-6      Central                19    EEG-Pz     Parietal Midline
 9     EEG-Cz     Vertex / Midline       20    EEG-15     Occipital
10     EEG-7      Central                21    EEG-16     Occipital
```

> [!TIP]
> Electrodes **`EEG-C3` (index 7)**, **`EEG-Cz` (index 9)**, and **`EEG-C4` (index 11)** overlay the primary motor cortex ($M_1$) and are the most informative features for lateralized motor imagery decoding.

---

## 4. Event-to-Class Mapping & Label Conventions

| GDF Event Code | Integer Label ($y$) | Target Class Name | Visual Cue Meaning (`desc_2a.pdf`) | Total Trials |
| :---: | :---: | :--- | :--- | :---: |
| `769` | `0` | **Left Hand** | Visual arrow pointing left | 72 |
| `770` | `1` | **Right Hand** | Visual arrow pointing right | 72 |
| `771` | `2` | **Both Feet** | Visual arrow pointing downward | 72 |
| `772` | `3` | **Tongue** | Visual arrow pointing upward | 72 |

---

## 5. Partition Architecture & Class Distribution

The dataset is partitioned using a reproducible stratified split (`seed=42`) with a **75% / 12.5% / 12.5% (6:1:1)** ratio:

| Partition | Split Ratio | Total Trials | Class 0 (Left) | Class 1 (Right) | Class 2 (Feet) | Class 3 (Tongue) | Artifact Flags (`1023`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training** | 75.0% | **216** | 54 (25.0%) | 54 (25.0%) | 54 (25.0%) | 54 (25.0%) | 13 (6.0%) |
| **Validation** | 12.5% | **36** | 9 (25.0%) | 9 (25.0%) | 9 (25.0%) | 9 (25.0%) | 2 (5.6%) |
| **Test (Held-out)** | 12.5% | **36** | 9 (25.0%) | 9 (25.0%) | 9 (25.0%) | 9 (25.0%) | 0 (0.0%) |
| **Total** | 100.0% | **288** | **72** | **72** | **72** | **72** | **15** |

- **Zero Overlap:** The train, validation, and test index sets are mutually disjoint ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **Completeness:** The union of partitions equals exactly the 288 trials without any silent loss.

---

## 6. Leakage-Free Normalization Procedure

- **Method:** Channel-wise standard z-score normalization ($z = \frac{x - \mu_c}{\sigma_c}$).
- **Leakage Protection Contract:** Parameters $\mu_c$ and $\sigma_c$ were computed **strictly on `X_train`** across all training epochs and time points.
- **Validation & Test Transformation:** `X_val` and `X_test` were transformed using the frozen training statistics. No test-set statistics were accessed.
- **Inference Parameter Reuse:** If new streaming data is processed, the saved parameters in `A01T_split_metadata.json` (or via `get_normalization_params()`) must be applied. **Never refit normalization parameters on test or live streaming data!**
- **Unnormalized Signals Available:** Unnormalized bandpassed arrays (`X_train`, `X_val`, `X_test` in SI Volts) are also included in the NPZ archive for CSP baseline covariance calculation.

---

## 7. Windowing Options: Full Epoch vs. Recommended Central Window

Person 1 provides two window representations:

1. **Canonical Full-Epoch View (`window='full'`):**
   - Shape: `(N, 22, 1000)`
   - Time Interval: $t \in [0.000, 3.996]\ \text{s}$ (4.0 seconds relative to visual cue onset).
   - Use Case: Preserves the standard 4-second model input format without altering model interfaces.
2. **Recommended Central Window (`window='central'`):**
   - Shape: `(N, 22, 750)`
   - Time Interval: $t \in [0.500, 3.500]\ \text{s}$ (samples 125 to 875, duration 3.0 seconds).
   - Use Case: **Recommended for Deep Learning.** Completely avoids the first and last 200 ms where per-epoch filter edge transients occur (overall mean difference $<0.0003\ \mu\text{V}$, sample-mean profile $<0.005\ \mu\text{V}$, peak max $0.0366\ \mu\text{V}$). Focuses on the physiological window where motor imagery ERD is strongest.

---

## 8. Artifact Handling Policy (`Code 1023`)

- **Dataset Authors' Mark:** In `desc_2a.pdf`, code `1023` marks trials containing visual artifacts (e.g. eye blinks, ocular movements, muscle tension).
- **Exact Coincidence:** 15 trials coincide with code `1023` event tags. All 15 start samples coincide with event `768` (trial start).
- **Reversible Dual-Mode Selection:**
  - **Option A (Default — Recommended for Training):** Train on all 216 trials (`clean_only=False`). Maintains 100% perfect class balance (54 per class). 8–30 Hz bandpass filtering attenuates low-frequency EOG drifts by >99.8%.
  - **Option B (Sensitivity Testing):** Train on 203 clean trials (`clean_only=True`). Excludes the 13 training artifacts. Note: produces slight class asymmetry (52, 52, 50, 49).

---

## 9. Critical Technical Caveats for Modeling & Integration

1. **Non-Causal Zero-Phase Filtering (Offline Benchmark Only):**
   - Filtering was performed using `scipy.signal.filtfilt` (4th-order Butterworth, effective 8th-order, odd-reflection padding `padlen=27`).
   - This achieves zero phase distortion ($0^\circ$), preserving peak ERD latencies for offline model training.
   - **Crucial Caution for Person 3:** Zero-phase filtering is **strictly non-causal** because it requires samples from the future. It cannot be used sample-by-sample in Phase 6 simulated live streaming. Live streaming requires a causal forward-only filter, which will introduce a group delay of 20–50 ms.
2. **Zero Artifacts in Held-Out Test Set:**
   - Under stratified sampling (`seed=42`), all 15 artifact trials fell into Train (13) and Validation (2). The Test partition contains **0 artifact trials**.
   - **Implication:** The held-out test evaluation measures performance on clean motor imagery trials only. The modeling teammate cannot use `X_test` to measure model degradation under artifact contamination.
3. **Boundary Edge Transients:**
   - In full epochs `(N, 22, 1000)`, the first and last 50 samples (200 ms) contain edge transients (up to $32.61\ \mu\text{V}$ vs. continuous filtering). Use `window='central'` or slice `X[:, :, 125:875]` to eliminate them.

---

## 10. File Index in `data/processed/`

- **`A01T_split_data.npz` (82.3 MB):** Primary compressed archive containing all split arrays, windowed arrays, normalized/unnormalized signals, labels, artifact masks, and channel names.
- **`A01T_split_metadata.json` (9.4 KB):** Complete split metadata, class distributions, normalization parameters ($\mu_c, \sigma_c$), run breakdown, and reproducibility settings.
- **`A01T_epochs_filtered.npz` (23.5 MB):** Pre-split filtered epochs (288, 22, 1000).
- **`A01T_epochs.npz` (11.6 MB):** Raw extracted epochs (288, 22, 1000).
- **`README.md`:** This documentation file.
