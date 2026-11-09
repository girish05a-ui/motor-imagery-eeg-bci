"""Create the approved A01–A07-development / A08–A09-test EEG archive.

This script derives a new archive from the verified pooled artifact.  It never
modifies that source archive or its within-subject split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


SOURCE_ARCHIVE = "A01-A09T_pooled_split_data.npz"
OUTPUT_ARCHIVE = "A01-A07T_dev_A08-A09T_test_cross_subject_data.npz"
OUTPUT_METADATA = "A01-A07T_dev_A08-A09T_test_cross_subject_metadata.json"
DEVELOPMENT_SUBJECTS = ("A01", "A02", "A03", "A04", "A05", "A06", "A07")
TEST_SUBJECTS = ("A08", "A09")
RANDOM_SEED = 42
WINDOW = slice(125, 875)


def _processed_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "processed"


def _restore_pooled_trials(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Restore complete trial order from the source archive's disjoint indices."""
    with np.load(path, allow_pickle=False) as source:
        all_indices = np.concatenate([source[f"{part}_indices"] for part in ("train", "val", "test")])
        total = len(all_indices)
        if len(np.unique(all_indices)) != total or not np.array_equal(np.sort(all_indices), np.arange(total)):
            raise ValueError("Source pooled indices are not a complete, disjoint trial provenance map")

        channels = source["channel_names"].copy()
        X = np.empty((total, channels.size, 1000), dtype=np.float32)
        y = np.empty(total, dtype=np.int64)
        artifacts = np.empty(total, dtype=bool)
        subjects = np.empty(total, dtype="U3")
        for part in ("train", "val", "test"):
            indices = source[f"{part}_indices"]
            X[indices] = source[f"X_{part}"]
            y[indices] = source[f"y_{part}"]
            artifacts[indices] = source[f"artifact_mask_{part}"]
            subjects[indices] = source[f"subject_ids_{part}"]
    return X, y, artifacts, subjects, channels


def create_cross_subject_archive(processed_dir: Path | None = None) -> tuple[Path, Path]:
    """Create the separate approved cross-subject archive and manifest."""
    processed_dir = processed_dir or _processed_dir()
    source_path = processed_dir / SOURCE_ARCHIVE
    output_path = processed_dir / OUTPUT_ARCHIVE
    metadata_path = processed_dir / OUTPUT_METADATA
    if not source_path.is_file():
        raise FileNotFoundError(f"Required pooled source archive not found: {source_path}")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("Cross-subject artifacts already exist; refusing to overwrite them")

    X_all, y_all, artifact_all, subject_all, channels = _restore_pooled_trials(source_path)
    if X_all.shape != (2592, 22, 1000) or not np.isfinite(X_all).all():
        raise ValueError(f"Unexpected source signal contract: {X_all.shape}")
    if set(np.unique(y_all)) != {0, 1, 2, 3}:
        raise ValueError("Source archive labels are not exactly {0, 1, 2, 3}")
    if set(np.unique(subject_all)) != set(DEVELOPMENT_SUBJECTS + TEST_SUBJECTS):
        raise ValueError("Source archive subject identifiers do not match A01–A09")

    development_indices = np.flatnonzero(np.isin(subject_all, DEVELOPMENT_SUBJECTS))
    held_out_indices = np.flatnonzero(np.isin(subject_all, TEST_SUBJECTS))
    strata = np.asarray([f"{subject}_class{label}" for subject, label in zip(subject_all[development_indices], y_all[development_indices])])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    train_local, val_local = next(splitter.split(np.zeros(len(development_indices)), strata))
    train_indices = np.sort(development_indices[train_local])
    val_indices = np.sort(development_indices[val_local])
    test_indices = np.sort(held_out_indices)

    if set(subject_all[train_indices]) != set(DEVELOPMENT_SUBJECTS):
        raise AssertionError("Every development participant must occur in training")
    if set(subject_all[val_indices]) != set(DEVELOPMENT_SUBJECTS):
        raise AssertionError("Every development participant must occur in validation")
    if set(subject_all[test_indices]) != set(TEST_SUBJECTS):
        raise AssertionError("Held-out test must contain A08 and A09 only")
    if any(np.intersect1d(left, right).size for left, right in ((train_indices, val_indices), (train_indices, test_indices), (val_indices, test_indices))):
        raise AssertionError("Cross-subject split indices overlap")
    if len(train_indices) != 1512 or len(val_indices) != 504 or len(test_indices) != 576:
        raise AssertionError("Unexpected approved cross-subject split sizes")

    X_train, X_val, X_test = X_all[train_indices], X_all[val_indices], X_all[test_indices]
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    X_train_norm = ((X_train - mean) / std).astype(np.float32)
    X_val_norm = ((X_val - mean) / std).astype(np.float32)
    X_test_norm = ((X_test - mean) / std).astype(np.float32)

    np.savez_compressed(
        output_path,
        X_train=X_train, X_val=X_val, X_test=X_test,
        X_train_norm=X_train_norm, X_val_norm=X_val_norm, X_test_norm=X_test_norm,
        X_train_windowed=X_train[:, :, WINDOW], X_val_windowed=X_val[:, :, WINDOW], X_test_windowed=X_test[:, :, WINDOW],
        X_train_norm_windowed=X_train_norm[:, :, WINDOW], X_val_norm_windowed=X_val_norm[:, :, WINDOW], X_test_norm_windowed=X_test_norm[:, :, WINDOW],
        y_train=y_all[train_indices], y_val=y_all[val_indices], y_test=y_all[test_indices],
        artifact_mask_train=artifact_all[train_indices], artifact_mask_val=artifact_all[val_indices], artifact_mask_test=artifact_all[test_indices],
        subject_ids_train=subject_all[train_indices], subject_ids_val=subject_all[val_indices], subject_ids_test=subject_all[test_indices],
        source_indices_train=train_indices, source_indices_val=val_indices, source_indices_test=test_indices,
        channel_names=channels, normalization_mean=mean.reshape(-1).astype(np.float32), normalization_std=std.reshape(-1).astype(np.float32),
    )

    def counts(indices: np.ndarray) -> dict[str, int]:
        return {str(label): int((y_all[indices] == label).sum()) for label in range(4)}

    def subject_counts(indices: np.ndarray) -> dict[str, int]:
        return {subject: int((subject_all[indices] == subject).sum()) for subject in np.unique(subject_all[indices])}

    metadata = {
        "dataset_name": "BCI Competition IV Dataset 2a",
        "source_archive": SOURCE_ARCHIVE,
        "source_archive_bytes": source_path.stat().st_size,
        "protocol": "Cross-subject holdout: A01-A07 development; A08-A09 final test only",
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "held_out_test_subjects": list(TEST_SUBJECTS),
        "sessions_included": "Training sessions A01T-A09T only",
        "random_seed": RANDOM_SEED,
        "split_policy": "A01-A07 subject-and-class stratified 75/25 train/validation; all A08-A09 trials held out for final test",
        "split_counts": {"train": len(train_indices), "val": len(val_indices), "test": len(test_indices)},
        "class_counts": {"train": counts(train_indices), "val": counts(val_indices), "test": counts(test_indices)},
        "subject_counts": {"train": subject_counts(train_indices), "val": subject_counts(val_indices), "test": subject_counts(test_indices)},
        "shape_full": [22, 1000],
        "shape_central": [22, 750],
        "central_window_samples": [125, 875],
        "channels": channels.tolist(),
        "labels": {"0": "Left Hand", "1": "Right Hand", "2": "Both Feet", "3": "Tongue"},
        "normalization": "Per-channel z-score fitted only on A01-A07 training trials; frozen parameters applied to validation and A08-A09 test",
        "units": "Volts (SI)",
        "dtype": "float32",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    archive, metadata = create_cross_subject_archive()
    print(f"Created: {archive}")
    print(f"Created: {metadata}")


if __name__ == "__main__":
    main()
