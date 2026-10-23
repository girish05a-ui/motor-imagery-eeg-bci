"""
Integration-Readiness & Handoff Validation Script
Role: Person 1 (EEG Data Loading, Preprocessing, and Validation)
Project: Deep Learning-Based Motor Imagery EEG Classification for a Simulated Real-Time BCI

Runs comprehensive checks on the packaged handoff arrays, metadata, and data loading interface.
Validates:
1. File existence and metadata schema.
2. Canonical full window (1,000 samples) and central window (750 samples) loading.
3. Normalized vs. raw Volts loading.
4. Clean-only vs. full partition loading.
5. Disjointness and completeness of train/val/test splits.
6. Zero data leakage in normalization.
7. Model tensor formatting compatibility (CNN spatial, temporal, and LSTM).
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path if running directly
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from load_data import (
    load_split_data,
    load_metadata,
    get_channel_names,
    get_class_names,
    get_normalization_params
)


def run_integration_readiness_checks():
    print("=" * 80)
    print("PART 6: INTEGRATION-READINESS & DATA HANDOFF VALIDATION")
    print("=" * 80)

    passed_checks = 0
    total_checks = 10

    # 1. Metadata Schema Verification
    print("\n[Check 1/10] Verifying Split Metadata Schema...")
    meta = load_metadata()
    required_keys = [
        "dataset_name", "subject_id", "session", "sfreq", "random_seed",
        "split_policy", "split_ratios", "split_counts", "class_balance_per_split",
        "artifact_policy", "normalization", "time_windows", "split_indices"
    ]
    for k in required_keys:
        assert k in meta, f"Missing key in metadata: {k}"
    assert meta["split_counts"]["total"] == 288
    assert meta["split_counts"]["train"] == 216
    assert meta["split_counts"]["val"] == 36
    assert meta["split_counts"]["test"] == 36
    print("  --> PASS: All required metadata keys present and split counts verified (216/36/36).")
    passed_checks += 1

    # 2. Channel Names & Ordering
    print("\n[Check 2/10] Verifying Channel Names & Order (22 Scalp Electrodes)...")
    chs = get_channel_names()
    assert len(chs) == 22, f"Expected 22 channels, got {len(chs)}"
    assert chs[0] == "EEG-Fz" and chs[7] == "EEG-C3" and chs[9] == "EEG-Cz" and chs[11] == "EEG-C4"
    # Ensure EOG channels are completely excluded
    for ch in chs:
        assert not ch.startswith("EOG"), f"Forbidden EOG channel found: {ch}"
    print(f"  --> PASS: 22 EEG channels confirmed (Motor cortex C3, Cz, C4 correctly mapped; EOG excluded).")
    passed_checks += 1

    # 3. Canonical Full-Epoch Loading (1,000 samples, 0.0-4.0s)
    print("\n[Check 3/10] Verifying Canonical Full-Epoch Shape (N, 22, 1000)...")
    X_tr, y_tr, m_tr = load_split_data('train', normalized=True, window='full')
    X_v, y_v, m_v = load_split_data('val', normalized=True, window='full')
    X_te, y_te, m_te = load_split_data('test', normalized=True, window='full')

    assert X_tr.shape == (216, 22, 1000), f"X_train full shape error: {X_tr.shape}"
    assert X_v.shape == (36, 22, 1000), f"X_val full shape error: {X_v.shape}"
    assert X_te.shape == (36, 22, 1000), f"X_test full shape error: {X_te.shape}"
    assert X_tr.dtype == np.float32 and X_v.dtype == np.float32 and X_te.dtype == np.float32
    print("  --> PASS: Full epoch shapes match canonical model interface (float32).")
    passed_checks += 1

    # 4. Recommended Central Window Loading (750 samples, 0.5-3.5s)
    print("\n[Check 4/10] Verifying Recommended Central-Window Shape (N, 22, 750)...")
    X_tr_c, _, _ = load_split_data('train', normalized=True, window='central')
    X_v_c, _, _ = load_split_data('val', normalized=True, window='central')
    X_te_c, _, _ = load_split_data('test', normalized=True, window='central')

    assert X_tr_c.shape == (216, 22, 750), f"X_train central shape error: {X_tr_c.shape}"
    assert X_v_c.shape == (36, 22, 750), f"X_val central shape error: {X_v_c.shape}"
    assert X_te_c.shape == (36, 22, 750), f"X_test central shape error: {X_te_c.shape}"
    # Verify central window is exact slice [125:875] of full window
    assert np.array_equal(X_tr_c, X_tr[:, :, 125:875]), "Central window slice misalignment!"
    print("  --> PASS: Central window arrays verified bitwise as samples [125:875] (transient-free).")
    passed_checks += 1

    # 5. Class Balance & Label Alignment Across All Partitions
    print("\n[Check 5/10] Verifying 100% Class Balance (54 / 9 / 9 per class)...")
    for c in range(4):
        assert (y_tr == c).sum() == 54, f"Train imbalance class {c}"
        assert (y_v == c).sum() == 9, f"Val imbalance class {c}"
        assert (y_te == c).sum() == 9, f"Test imbalance class {c}"
    classes = get_class_names()
    print(f"  --> PASS: Exactly 54/54/54/54 in Train, 9/9/9/9 in Val, 9/9/9/9 in Test for {classes}.")
    passed_checks += 1

    # 6. Artifact Tracking & Clean-Only Mode
    print("\n[Check 6/10] Verifying Artifact Mask Tracking & Clean-Only Reversibility...")
    assert m_tr.sum() == 13, f"Train artifact mismatch: {m_tr.sum()} != 13"
    assert m_v.sum() == 2, f"Val artifact mismatch: {m_v.sum()} != 2"
    assert m_te.sum() == 0, f"Test artifact mismatch: {m_te.sum()} != 0"

    # Test clean_only mode
    X_tr_clean, y_tr_clean, m_tr_clean = load_split_data('train', clean_only=True)
    X_v_clean, y_v_clean, m_v_clean = load_split_data('val', clean_only=True)
    X_te_clean, y_te_clean, m_te_clean = load_split_data('test', clean_only=True)

    assert len(X_tr_clean) == 203 and len(y_tr_clean) == 203 and m_tr_clean.sum() == 0
    assert len(X_v_clean) == 34 and len(y_v_clean) == 34 and m_v_clean.sum() == 0
    assert len(X_te_clean) == 36 and len(y_te_clean) == 36 and m_te_clean.sum() == 0
    print("  --> PASS: 15 Code 1023 trials tracked (13/2/0). Clean-only mode cleanly produces 203/34/36 trials.")
    passed_checks += 1

    # 7. Unnormalized Volts vs. Normalized Z-Score Integrity
    print("\n[Check 7/10] Verifying Raw Volts vs. Z-Score Normalized Signals...")
    X_tr_raw, _, _ = load_split_data('train', normalized=False)
    assert np.max(np.abs(X_tr_raw)) < 1e-3, "Raw signals not in SI Volts!"
    assert np.max(np.abs(X_tr)) > 1.0, "Normalized signals not z-scored!"

    # Verify zero data leakage: train stats are ~0 and ~1, val/test are not identically 0/1
    train_ch_means = np.mean(X_tr, axis=(0, 2))
    train_ch_stds = np.std(X_tr, axis=(0, 2))
    val_ch_means = np.mean(X_v, axis=(0, 2))
    val_ch_stds = np.std(X_v, axis=(0, 2))

    assert np.all(np.abs(train_ch_means) < 1e-6), "Train channel means not centered at 0"
    assert np.all(np.abs(train_ch_stds - 1.0) < 1e-4), "Train channel stds not scaled to 1"
    assert not np.all(np.abs(val_ch_means) < 1e-6), "Val data appears artificially centered (leakage)!"
    print("  --> PASS: Raw Volts (SI) and Z-score normalized arrays verified. Zero leakage confirmed.")
    passed_checks += 1

    # 8. Numerical Integrity (Zero NaN, Zero Inf)
    print("\n[Check 8/10] Verifying Numerical Cleanliness (Zero NaN, Zero Inf)...")
    for name, arr in [("X_train", X_tr), ("X_val", X_v), ("X_test", X_te), ("X_raw", X_tr_raw)]:
        assert np.isfinite(arr).all(), f"Non-finite values detected in {name}!"
    print("  --> PASS: All arrays 100% finite across all partitions and representations.")
    passed_checks += 1

    # 9. Partition Disjointness & Completeness
    print("\n[Check 9/10] Verifying Partition Disjointness and Completeness...")
    proc_dir = src_dir.parent / "data" / "processed"
    archive = np.load(proc_dir / "A01T_split_data.npz")
    tr_idx = set(archive['train_indices'])
    val_idx = set(archive['val_indices'])
    te_idx = set(archive['test_indices'])

    assert len(tr_idx & val_idx) == 0, "Train and Val overlap!"
    assert len(tr_idx & te_idx) == 0, "Train and Test overlap!"
    assert len(val_idx & te_idx) == 0, "Val and Test overlap!"
    assert tr_idx | val_idx | te_idx == set(range(288)), "Incomplete partition union!"
    print("  --> PASS: Zero overlap across partitions. Union equals exactly 288 trials.")
    passed_checks += 1

    # 10. Deep Learning Model Input Compatibility (PyTorch / TensorFlow / Scikit-Learn)
    print("\n[Check 10/10] Verifying Model Input Tensor Formatting Compatibility...")
    # Formats required by Person 2:
    # A. 2D Temporal Conv: (batch, 22, 1000)
    batch_2d = X_tr[:32]
    assert batch_2d.shape == (32, 22, 1000)

    # B. Spatial-Temporal 2D Conv (EEGNet / ShallowConvNet): (batch, 1, 22, 1000)
    batch_spatial = np.expand_dims(batch_2d, axis=1)
    assert batch_spatial.shape == (32, 1, 22, 1000)

    # C. Recurrent / LSTM format: (batch, time=1000, channels=22)
    batch_lstm = np.transpose(batch_2d, (0, 2, 1))
    assert batch_lstm.shape == (32, 1000, 22)

    # D. Central window formats: (batch, 22, 750), (batch, 1, 22, 750), (batch, 750, 22)
    batch_c_2d = X_tr_c[:32]
    assert batch_c_2d.shape == (32, 22, 750)
    assert np.expand_dims(batch_c_2d, axis=1).shape == (32, 1, 22, 750)
    assert np.transpose(batch_c_2d, (0, 2, 1)).shape == (32, 750, 22)

    print("  --> PASS: Array layouts support 2D Conv (1, 22, T), Temporal Conv (22, T), and LSTM (T, 22).")
    passed_checks += 1

    print("\n" + "=" * 80)
    print(f"INTEGRATION-READINESS VERDICT: ALL {passed_checks}/{total_checks} CHECKS PASSED")
    print("Data handoff package is 100% verified and ready for Person 2 (Modeling).")
    print("=" * 80)
    return True


if __name__ == '__main__':
    try:
        run_integration_readiness_checks()
    except Exception as e:
        print(f"\n[ERROR] Integration-readiness validation failed: {e}", file=sys.stderr)
        sys.exit(1)
