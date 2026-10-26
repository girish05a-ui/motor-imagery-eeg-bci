"""
Batch preprocessing for BCI Competition IV Dataset 2a, training sessions A01T-A09T.

Place this file in the project's src/ folder and run:
    python src/process_all_subjects.py

Safety:
- Existing A01T epochs/filtered/split files are preserved and never overwritten.
- A02T-A09T outputs are created using the same event mapping, 8-30 Hz filter,
  artifact-mask policy, and 75/12.5/12.5 split approach.
- Evaluation recordings A01E-A09E are intentionally NOT used here.
- Creates a pooled A01-A09 training-session dataset for the teammate.
"""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedShuffleSplit

EVENT_MAPPING = {"769": 0, "770": 1, "771": 2, "772": 3}
CLASS_NAMES = {0: "Left Hand", 1: "Right Hand", 2: "Both Feet", 3: "Tongue"}
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
SFREQ_EXPECTED = 250.0
LOW_HZ, HIGH_HZ, FILTER_ORDER = 8.0, 30.0, 4
SEED = 42


def json_write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def validate_xy(X, y, subject):
    assert X.ndim == 3, f"{subject}: X must be 3D, got {X.shape}"
    assert y.ndim == 1 and len(X) == len(y), f"{subject}: X/y mismatch"
    assert X.shape[1:] == (22, 1000), f"{subject}: expected (N,22,1000), got {X.shape}"
    assert len(X) == 288, f"{subject}: expected 288 training trials, got {len(X)}"
    assert np.isfinite(X).all(), f"{subject}: NaN/Inf in EEG"
    assert set(np.unique(y).tolist()) == {0, 1, 2, 3}, f"{subject}: missing/invalid class"
    counts = {int(c): int((y == c).sum()) for c in range(4)}
    assert all(counts[c] == 72 for c in range(4)), f"{subject}: unexpected class counts {counts}"
    return counts


def artifact_indices_from_raw(raw, cue_samples, sfreq):
    """Match each cue to a nearby 1023 artifact annotation, preserving all trials."""
    artifact_onsets = []
    for ann in raw.annotations:
        if str(ann["description"]).strip() == "1023":
            # Annotation onset is in seconds in the recording time base.
            artifact_onsets.append(float(ann["onset"]))
    cue_seconds = np.asarray(cue_samples, dtype=float) / float(sfreq)
    # Retain the project's existing +/-5-second coincidence convention.
    return [
        int(i) for i, cue_t in enumerate(cue_seconds)
        if any(abs(cue_t - art_t) < 5.0 for art_t in artifact_onsets)
    ]


def extract_one(gdf_path: Path, subject: str, processed: Path):
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - SFREQ_EXPECTED) > 1e-6:
        raise ValueError(f"{gdf_path.name}: expected 250 Hz, got {sfreq}")
    all_channels = raw.ch_names
    eog = [c for c in all_channels if "EOG" in c.upper()]
    eeg = [c for c in all_channels if "EOG" not in c.upper()]
    if len(eeg) != 22 or len(eog) != 3:
        raise ValueError(f"{gdf_path.name}: expected 22 EEG + 3 EOG; found {len(eeg)} + {len(eog)}")

    target_events = {code: int(code) for code in EVENT_MAPPING}
    events, event_dict = mne.events_from_annotations(raw, event_id=target_events, verbose="ERROR")
    if len(events) != 288:
        raise ValueError(f"{gdf_path.name}: expected 288 motor imagery cues, got {len(events)}")
    tmin, tmax = 0.0, 4.0 - 1.0 / sfreq
    epochs = mne.Epochs(
        raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, picks=eeg,
        baseline=None, preload=True, reject=None, flat=None, verbose="ERROR"
    )
    X = epochs.get_data().astype(np.float32)
    codes = epochs.events[:, 2]
    y = np.asarray([EVENT_MAPPING[str(int(code))] for code in codes], dtype=np.int64)
    counts = validate_xy(X, y, subject)
    artifact_indices = artifact_indices_from_raw(raw, epochs.events[:, 0], sfreq)
    artifact_mask = np.zeros(len(y), dtype=bool)
    artifact_mask[artifact_indices] = True

    out_npz = processed / f"{subject}T_epochs.npz"
    out_json = processed / f"{subject}T_epochs_metadata.json"
    np.savez_compressed(out_npz, X=X, y=y, channel_names=np.asarray(eeg))
    meta = {
        "dataset_name": "BCI Competition IV Dataset 2a",
        "subject_id": subject, "session": "T (Training)", "source_file": gdf_path.name,
        "sfreq": sfreq, "tmin": tmin, "tmax": tmax, "duration_seconds": 4.0,
        "n_samples_per_epoch": int(X.shape[-1]), "n_epochs": int(len(X)),
        "n_channels": int(X.shape[1]), "channel_names": eeg, "excluded_channels": eog,
        "event_mapping": EVENT_MAPPING, "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "class_counts": {str(k): v for k, v in counts.items()},
        "coincident_code_1023_trials": artifact_indices,
        "artifact_policy_note": "All trials retained; artifact flags are supplied separately in filtered NPZ.",
        "data_units": "Volts (SI)", "dtype": "float32"
    }
    json_write(out_json, meta)
    with np.load(out_npz) as saved:
        assert saved["X"].shape == X.shape and np.array_equal(saved["y"], y)
    return X, y, eeg, artifact_mask, sfreq, meta


def filter_one(X, y, channel_names, artifact_mask, sfreq, subject, processed):
    b, a = butter(FILTER_ORDER, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=sfreq)
    Xf = filtfilt(b, a, X, axis=-1).astype(np.float32)
    assert Xf.shape == X.shape and np.isfinite(Xf).all()
    out_npz = processed / f"{subject}T_epochs_filtered.npz"
    out_json = processed / f"{subject}T_epochs_filtered_metadata.json"
    np.savez_compressed(
        out_npz, X=Xf, y=y, channel_names=np.asarray(channel_names),
        artifact_mask=artifact_mask.astype(bool)
    )
    meta = {
        "dataset_name": "BCI Competition IV Dataset 2a", "subject_id": subject,
        "session": "T (Training)", "source_filtered_npz": f"{subject}T_epochs.npz",
        "sfreq": float(sfreq), "n_epochs": int(len(y)), "n_channels": int(Xf.shape[1]),
        "n_times": int(Xf.shape[2]), "channel_names": list(channel_names),
        "filter": {"type": "Butterworth IIR bandpass", "order": FILTER_ORDER,
                   "l_freq": LOW_HZ, "h_freq": HIGH_HZ, "phase": "zero-phase (scipy filtfilt)",
                   "causality": "non-causal; offline preprocessing only"},
        "artifact_policy": {
            "policy": "Preserve all trials and provide boolean artifact_mask",
            "artifact_count": int(artifact_mask.sum()),
            "coincident_code_1023_trials": np.flatnonzero(artifact_mask).tolist(),
            "class_counts_all": {str(c): int((y == c).sum()) for c in range(4)}
        },
        "dtype": "float32", "data_units": "Volts (SI)"
    }
    json_write(out_json, meta)
    with np.load(out_npz) as saved:
        assert saved["X"].shape == (288, 22, 1000)
        assert saved["y"].shape == (288,)
        assert np.array_equal(saved["y"], y)
        assert np.array_equal(saved["artifact_mask"], artifact_mask)
        assert np.isfinite(saved["X"]).all()
    return Xf


def stratified_indices(y, strata, seed=SEED):
    """75/12.5/12.5 split, stratified by subject AND class."""
    n = len(y)
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, temp_idx = next(s1.split(np.zeros(n), strata))
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_sub, test_sub = next(s2.split(np.zeros(len(temp_idx)), strata[temp_idx]))
    return np.sort(train_idx), np.sort(temp_idx[val_sub]), np.sort(temp_idx[test_sub])


def main():
    root = Path(__file__).resolve().parent.parent
    raw_dir, processed = root / "data" / "raw", root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # Load the existing A01T result rather than regenerating or overwriting it.
    a01_raw = processed / "A01T_epochs.npz"
    a01_meta_path = processed / "A01T_epochs_metadata.json"
    a01_filt = processed / "A01T_epochs_filtered.npz"
    a01_filt_meta_path = processed / "A01T_epochs_filtered_metadata.json"
    required = [a01_raw, a01_meta_path, a01_filt, a01_filt_meta_path]
    if not all(p.exists() for p in required):
        missing = [p.name for p in required if not p.exists()]
        raise FileNotFoundError("Existing A01T outputs are required and must be preserved. Missing: " + ", ".join(missing))

    with np.load(a01_raw) as z:
        X01_raw, y01, ch01 = z["X"].copy(), z["y"].copy(), z["channel_names"].tolist()
    with open(a01_meta_path, encoding="utf-8") as f:
        meta01 = json.load(f)
    with np.load(a01_filt) as z:
        X01_filt, y01f = z["X"].copy(), z["y"].copy()
        ch01f = z["channel_names"].tolist()
        art01 = z["artifact_mask"].astype(bool).copy()
    with open(a01_filt_meta_path, encoding="utf-8") as f:
        fmeta01 = json.load(f)
    if not np.array_equal(y01, y01f):
        raise ValueError("Existing A01T raw/filtered labels do not match.")
    validate_xy(X01_raw, y01, "A01")
    if X01_filt.shape != (288, 22, 1000) or not np.isfinite(X01_filt).all():
        raise ValueError("Existing A01T filtered data failed shape/finiteness checks.")

    records = [{
        "subject": "A01", "X": X01_filt, "y": y01f, "channels": ch01f,
        "artifact_mask": art01, "sfreq": float(fmeta01.get("sfreq", 250.0))
    }]
    print("A01T: using existing verified files; not overwriting them.")

    # Process only A02T-A09T; E (evaluation) sessions remain separate.
    for subject in SUBJECTS[1:]:
        gdf = raw_dir / f"{subject}T.gdf"
        if not gdf.exists():
            raise FileNotFoundError(f"Missing expected training recording: {gdf}")
        print(f"\nProcessing {gdf.name} ...")
        X, y, channels, artifact_mask, sfreq, _ = extract_one(gdf, subject, processed)
        Xf = filter_one(X, y, channels, artifact_mask, sfreq, subject, processed)
        records.append({
            "subject": subject, "X": Xf, "y": y, "channels": channels,
            "artifact_mask": artifact_mask, "sfreq": sfreq
        })
        print(f"  OK: {Xf.shape}; classes={np.bincount(y, minlength=4).tolist()}; artifacts={int(artifact_mask.sum())}")

    # Verify common channel ordering before pooling.
    reference_channels = records[0]["channels"]
    for r in records:
        if r["channels"] != reference_channels:
            raise ValueError(f"{r['subject']}: EEG channel order differs from A01.")

    X_all = np.concatenate([r["X"] for r in records], axis=0).astype(np.float32)
    y_all = np.concatenate([r["y"] for r in records]).astype(np.int64)
    artifact_all = np.concatenate([r["artifact_mask"] for r in records]).astype(bool)
    subject_names = np.concatenate([
        np.full(len(r["y"]), r["subject"], dtype="U3") for r in records
    ])
    subject_class_strata = np.asarray([
        f"{r['subject']}_class{int(label)}"
        for r in records for label in r["y"]
    ])
    assert X_all.shape == (2592, 22, 1000)
    assert y_all.shape == (2592,) and len(subject_names) == len(y_all)
    assert np.isfinite(X_all).all()
    assert all(int((y_all == c).sum()) == 648 for c in range(4))

    train_idx, val_idx, test_idx = stratified_indices(y_all, subject_class_strata)
    # Train-only normalization, globally fitted across the training trials.
    Xtr, Xv, Xte = X_all[train_idx], X_all[val_idx], X_all[test_idx]
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sigma = Xtr.std(axis=(0, 2), keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    Xtrn = ((Xtr - mu) / sigma).astype(np.float32)
    Xvn = ((Xv - mu) / sigma).astype(np.float32)
    Xten = ((Xte - mu) / sigma).astype(np.float32)
    s0, s1 = int(round(0.5 * records[0]["sfreq"])), int(round(3.5 * records[0]["sfreq"]))

    def cls_counts(idx):
        return {str(c): int((y_all[idx] == c).sum()) for c in range(4)}
    def subject_counts(idx):
        return {s: int((subject_names[idx] == s).sum()) for s in SUBJECTS}

    pooled_path = processed / "A01-A09T_pooled_split_data.npz"
    pooled_meta_path = processed / "A01-A09T_pooled_split_metadata.json"
    np.savez_compressed(
        pooled_path,
        X_train=Xtr, X_val=Xv, X_test=Xte,
        X_train_norm=Xtrn, X_val_norm=Xvn, X_test_norm=Xten,
        X_train_windowed=Xtr[:, :, s0:s1], X_val_windowed=Xv[:, :, s0:s1],
        X_test_windowed=Xte[:, :, s0:s1],
        X_train_norm_windowed=Xtrn[:, :, s0:s1],
        X_val_norm_windowed=Xvn[:, :, s0:s1],
        X_test_norm_windowed=Xten[:, :, s0:s1],
        y_train=y_all[train_idx], y_val=y_all[val_idx], y_test=y_all[test_idx],
        artifact_mask_train=artifact_all[train_idx],
        artifact_mask_val=artifact_all[val_idx], artifact_mask_test=artifact_all[test_idx],
        subject_ids_train=subject_names[train_idx], subject_ids_val=subject_names[val_idx],
        subject_ids_test=subject_names[test_idx],
        train_indices=train_idx, val_indices=val_idx, test_indices=test_idx,
        channel_names=np.asarray(reference_channels),
        normalization_mean=mu.reshape(-1), normalization_std=sigma.reshape(-1)
    )
    json_write(pooled_meta_path, {
        "dataset_name": "BCI Competition IV Dataset 2a",
        "subjects": SUBJECTS, "sessions_included": "Training sessions A01T-A09T only",
        "evaluation_sessions_included": False,
        "total_epochs": int(len(y_all)), "shape_all": list(X_all.shape),
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "class_counts_all": {str(c): int((y_all == c).sum()) for c in range(4)},
        "split_policy": "Stratified 75/12.5/12.5 by subject-and-class; random_state=42",
        "split_counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "class_counts": {"train": cls_counts(train_idx), "val": cls_counts(val_idx), "test": cls_counts(test_idx)},
        "subject_counts": {"train": subject_counts(train_idx), "val": subject_counts(val_idx), "test": subject_counts(test_idx)},
        "artifact_counts": {"all": int(artifact_all.sum()), "train": int(artifact_all[train_idx].sum()),
                            "val": int(artifact_all[val_idx].sum()), "test": int(artifact_all[test_idx].sum())},
        "normalization": "Per-channel z-score fitted only on pooled training trials; same frozen parameters applied to val/test",
        "central_window_samples": [s0, s1], "central_window_seconds": [0.5, 3.5],
        "channels": reference_channels, "dtype": "float32", "units": "Volts (SI)",
        "note": "This is within-subject trial-level evaluation, not leave-one-subject-out evaluation."
    })

    # Reload and integrity checks for the pooled handoff.
    with np.load(pooled_path) as z:
        assert z["X_train"].shape == (1944, 22, 1000)
        assert z["X_val"].shape == (324, 22, 1000)
        assert z["X_test"].shape == (324, 22, 1000)
        assert z["X_train_windowed"].shape == (1944, 22, 750)
        for name in ["X_train", "X_val", "X_test", "X_train_norm", "X_val_norm", "X_test_norm"]:
            assert np.isfinite(z[name]).all(), f"Non-finite values in {name}"
        assert len(z["subject_ids_train"]) == len(z["y_train"])
        assert len(z["subject_ids_val"]) == len(z["y_val"])
        assert len(z["subject_ids_test"]) == len(z["y_test"])
        assert len(set(z["train_indices"]) & set(z["val_indices"])) == 0
        assert len(set(z["train_indices"]) & set(z["test_indices"])) == 0
        assert len(set(z["val_indices"]) & set(z["test_indices"])) == 0

    print("\nALL DONE")
    print("A01T existing files: preserved")
    print("A02T-A09T: epochs + filtered epochs + metadata created")
    print(f"Pooled handoff: {pooled_path.name}")
    print(f"Pooled metadata: {pooled_meta_path.name}")
    print("Expected pooled shapes: train (1944,22,1000), val (324,22,1000), test (324,22,1000)")
    print("Evaluation E sessions were not processed or mixed into training.")


if __name__ == "__main__":
    main()
