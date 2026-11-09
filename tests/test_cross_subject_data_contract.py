"""Regression checks for the approved cross-subject EEG data contract."""

from pathlib import Path
import zipfile

import numpy as np

from src.load_data import get_channel_names, load_split_data


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
CROSS_ARCHIVE = PROCESSED / "A01-A07T_dev_A08-A09T_test_cross_subject_data.npz"


def _npz_member_shape(path: Path, key: str) -> tuple[int, ...]:
    """Read a compressed NPZ member's NPY header without materializing its array."""
    with zipfile.ZipFile(path) as archive, archive.open(f"{key}.npy") as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(member)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(member)
    return shape


def test_legacy_a01_loader_remains_backward_compatible() -> None:
    X, y, mask = load_split_data("train", dataset="a01")
    assert X.shape == (216, 22, 1000)
    assert X.dtype == np.float32
    assert y.shape == (216,) and y.dtype == np.int64
    assert mask.shape == (216,) and mask.dtype == bool


def test_pooled_loader_is_available_without_changing_legacy_default() -> None:
    X, y, mask = load_split_data("val", window="central", dataset="pooled")
    assert X.shape == (324, 22, 750)
    assert np.bincount(y, minlength=4).tolist() == [81, 81, 81, 81]
    assert mask.shape == (324,)
    assert get_channel_names(dataset="pooled") == get_channel_names(dataset="a01")


def test_cross_subject_participant_separation_shapes_labels_and_normalization() -> None:
    assert CROSS_ARCHIVE.is_file(), "Run src/create_cross_subject_split.py before this test"
    with np.load(CROSS_ARCHIVE, allow_pickle=False) as data:
        train_subjects = set(data["subject_ids_train"].tolist())
        val_subjects = set(data["subject_ids_val"].tolist())
        test_subjects = set(data["subject_ids_test"].tolist())
        assert train_subjects == val_subjects == {"A01", "A02", "A03", "A04", "A05", "A06", "A07"}
        assert test_subjects == {"A08", "A09"}
        assert not (train_subjects | val_subjects) & test_subjects

        assert _npz_member_shape(CROSS_ARCHIVE, "X_train") == (1512, 22, 1000)
        assert _npz_member_shape(CROSS_ARCHIVE, "X_val") == (504, 22, 1000)
        assert _npz_member_shape(CROSS_ARCHIVE, "X_test") == (576, 22, 1000)
        assert _npz_member_shape(CROSS_ARCHIVE, "X_train_norm_windowed") == (1512, 22, 750)
        assert _npz_member_shape(CROSS_ARCHIVE, "X_val_norm_windowed") == (504, 22, 750)
        assert _npz_member_shape(CROSS_ARCHIVE, "X_test_norm_windowed") == (576, 22, 750)
        assert data["channel_names"].tolist() == get_channel_names(dataset="a01")

        for part, expected in (("train", [378, 378, 378, 378]), ("val", [126, 126, 126, 126]), ("test", [144, 144, 144, 144])):
            labels = data[f"y_{part}"]
            assert labels.dtype == np.int64
            assert set(np.unique(labels)) == {0, 1, 2, 3}
            assert np.bincount(labels, minlength=4).tolist() == expected

        source_indices = [data[f"source_indices_{part}"] for part in ("train", "val", "test")]
        assert len(np.unique(np.concatenate(source_indices))) == 2592
        assert not np.intersect1d(source_indices[0], source_indices[1]).size
        assert not np.intersect1d(source_indices[0], source_indices[2]).size
        assert not np.intersect1d(source_indices[1], source_indices[2]).size

        train_normalized = data["X_train_norm"]
        assert np.isfinite(train_normalized).all()
        assert np.abs(train_normalized.mean(axis=(0, 2))).max() < 2e-6
        assert np.abs(train_normalized.std(axis=(0, 2)) - 1.0).max() < 2e-6


def test_cross_subject_loader_uses_the_separate_artifact() -> None:
    X_train, y_train, _ = load_split_data("train", window="central", dataset="cross_subject")
    X_test, y_test, _ = load_split_data("test", window="central", dataset="cross_subject")
    assert X_train.shape == (1512, 22, 750)
    assert X_test.shape == (576, 22, 750)
    assert np.bincount(y_train, minlength=4).tolist() == [378, 378, 378, 378]
    assert np.bincount(y_test, minlength=4).tolist() == [144, 144, 144, 144]


def test_cross_subject_test_uses_frozen_training_statistics() -> None:
    """Verify held-out data was transformed with stored train-only parameters."""
    with np.load(CROSS_ARCHIVE, allow_pickle=False) as data:
        expected = (
            (data["X_test"] - data["normalization_mean"][None, :, None])
            / data["normalization_std"][None, :, None]
        ).astype(np.float32)
        assert np.array_equal(expected, data["X_test_norm"])
