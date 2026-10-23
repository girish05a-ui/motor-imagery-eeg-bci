"""
Data Loading Interface for Person 2 (Modeling Teammate)
Role: Person 1 (EEG Data Loading, Preprocessing, and Validation)
Project: Deep Learning-Based Motor Imagery EEG Classification for a Simulated Real-Time BCI

Provides a clean, tested Python interface to load preprocessed, leakage-free EEG arrays.
Supports full 4.0s (1,000 samples) and recommended central 3.0s (750 samples) windows,
z-score normalized or raw filtered Volts, and full or clean-only trial subsets.
"""

from pathlib import Path
import json
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Optional PyTorch support if installed in environment
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


CLASS_MAPPING = {
    0: "Left Hand",
    1: "Right Hand",
    2: "Both Feet",
    3: "Tongue"
}

EVENT_CODE_MAPPING = {
    769: 0,
    770: 1,
    771: 2,
    772: 3
}


def _get_processed_dir(data_dir: Optional[str] = None) -> Path:
    if data_dir is not None:
        p = Path(data_dir)
        if p.exists():
            return p
    # Auto-resolve relative to this script
    script_dir = Path(__file__).resolve().parent
    default_dir = script_dir.parent / "data" / "processed"
    if default_dir.exists():
        return default_dir
    # Fallback to current working directory
    cwd_dir = Path.cwd() / "data" / "processed"
    if cwd_dir.exists():
        return cwd_dir
    raise FileNotFoundError("Could not locate 'data/processed' directory. Please specify data_dir explicitly.")


def load_metadata(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load the Part 5 split metadata JSON dictionary."""
    proc_dir = _get_processed_dir(data_dir)
    meta_path = proc_dir / "A01T_split_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_split_data(
    mode: str = 'train',
    normalized: bool = True,
    window: str = 'full',
    clean_only: bool = False,
    return_mask: bool = True,
    data_dir: Optional[str] = None
) -> Tuple[np.ndarray, ...]:
    """
    Load preprocessed EEG trials, labels, and artifact masks for Person 2 modeling.

    Parameters
    ----------
    mode : str
        Partition to load:
          - 'train': 216 trials (75.0% of dataset, 54/class)
          - 'val':    36 trials (12.5% of dataset, 9/class)
          - 'test':   36 trials (12.5% of dataset, 9/class)
          - 'all':   288 trials (Full dataset concatenated across train, val, test)
    normalized : bool, default=True
        - True: Channel-wise z-score normalized using mean/std fitted strictly on X_train.
        - False: Raw 8-30 Hz bandpass-filtered signals in SI Volts (V).
    window : str, default='full'
        - 'full': Full 4.0-second epoch (1,000 samples, t in [0.0, 4.0) s at 250 Hz).
                  Shape: (N, 22, 1000). Preserves canonical model interface.
        - 'central': Recommended central window (750 samples, t in [0.5, 3.5] s at 250 Hz).
                     Shape: (N, 22, 750). Completely excludes filter edge transients.
    clean_only : bool, default=False
        - False: Returns all trials in partition (100% class balance guaranteed).
        - True: Excludes trials flagged with Code 1023 visual artifacts.
                Note: Train has 13 artifacts (203 clean), Val has 2 artifacts (34 clean),
                Test has 0 artifacts (36 clean).
    return_mask : bool, default=True
        - True: Returns (X, y, artifact_mask)
        - False: Returns (X, y)
    data_dir : str, optional
        Custom path to data/processed directory. Auto-resolved if None.

    Returns
    -------
    X : np.ndarray, dtype=float32
        EEG signal array of shape (N, 22, 1000) or (N, 22, 750).
    y : np.ndarray, dtype=int64
        Class labels in {0, 1, 2, 3}.
    artifact_mask : np.ndarray, dtype=bool (optional)
        Boolean mask where True indicates Code 1023 visual artifact.
    """
    mode = mode.lower()
    window = window.lower()

    if mode not in ['train', 'val', 'test', 'all']:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'train', 'val', 'test', or 'all'.")
    if window not in ['full', 'central']:
        raise ValueError(f"Invalid window '{window}'. Expected 'full' (1000 samples) or 'central' (750 samples).")

    proc_dir = _get_processed_dir(data_dir)
    npz_path = proc_dir / "A01T_split_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Processed split archive not found: {npz_path}")

    archive = np.load(npz_path)

    if mode == 'all':
        # Concatenate train, val, test
        parts = ['train', 'val', 'test']
    else:
        parts = [mode]

    X_list, y_list, mask_list = [], [], []

    for part in parts:
        # Construct key name based on normalization and window
        if normalized:
            key_x = f"X_{part}_norm_windowed" if window == 'central' else f"X_{part}_norm"
        else:
            key_x = f"X_{part}_windowed" if window == 'central' else f"X_{part}"

        key_y = f"y_{part}"
        key_mask = f"artifact_mask_{part}"

        X_part = archive[key_x]
        y_part = archive[key_y]
        mask_part = archive[key_mask]

        if clean_only:
            keep = ~mask_part
            X_part = X_part[keep]
            y_part = y_part[keep]
            mask_part = mask_part[keep]

        X_list.append(X_part)
        y_list.append(y_part)
        mask_list.append(mask_part)

    X_out = np.concatenate(X_list, axis=0) if len(X_list) > 1 else X_list[0]
    y_out = np.concatenate(y_list, axis=0) if len(y_list) > 1 else y_list[0]
    mask_out = np.concatenate(mask_list, axis=0) if len(mask_list) > 1 else mask_list[0]

    # Validate output integrity
    assert X_out.ndim == 3 and X_out.shape[1] == 22, f"Unexpected X shape: {X_out.shape}"
    assert y_out.ndim == 1 and len(y_out) == len(X_out), f"Length mismatch: X={len(X_out)}, y={len(y_out)}"
    assert np.isfinite(X_out).all(), "Non-finite values found in output array X"

    if return_mask:
        return X_out, y_out, mask_out
    return X_out, y_out


def get_channel_names(data_dir: Optional[str] = None) -> list:
    """Return ordered list of 22 scalp EEG channel names."""
    proc_dir = _get_processed_dir(data_dir)
    npz_path = proc_dir / "A01T_split_data.npz"
    archive = np.load(npz_path)
    return archive['channel_names'].tolist()


def get_class_names() -> Dict[int, str]:
    """Return integer-to-class-name mapping dictionary."""
    return CLASS_MAPPING.copy()


def get_normalization_params(data_dir: Optional[str] = None) -> Dict[str, list]:
    """
    Return frozen channel-wise normalization parameters fitted on X_train.
    These parameters must be used at inference time; never refit on test data!
    """
    meta = load_metadata(data_dir)
    norm_meta = meta['normalization']
    return {
        "channel_means_uV": norm_meta['channel_means_uV'],
        "channel_stds_uV": norm_meta['channel_stds_uV'],
        "fit_scope": norm_meta['fit_scope'],
        "method": norm_meta['method']
    }


# Optional PyTorch Dataset Implementation
if TORCH_AVAILABLE:
    class EEGDataset(Dataset):
        """PyTorch Dataset wrapper for Person 1 preprocessed EEG trials."""
        def __init__(
            self,
            mode: str = 'train',
            normalized: bool = True,
            window: str = 'full',
            clean_only: bool = False,
            spatial_dim: bool = True,
            data_dir: Optional[str] = None
        ):
            X, y, mask = load_split_data(
                mode=mode,
                normalized=normalized,
                window=window,
                clean_only=clean_only,
                return_mask=True,
                data_dir=data_dir
            )
            # If spatial_dim=True, reshape (N, 22, T) -> (N, 1, 22, T) for 2D Conv
            if spatial_dim:
                X = np.expand_dims(X, axis=1)

            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).long()
            self.mask = torch.from_numpy(mask).bool()

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.X[idx], self.y[idx]


if __name__ == '__main__':
    print("=" * 70)
    print("PERSON 1 DATA LOADER SELF-TEST & VERIFICATION")
    print("=" * 70)

    chs = get_channel_names()
    print(f"Channels ({len(chs)}): {chs[:4]} ... {chs[-3:]}")
    print(f"Classes: {get_class_names()}")

    # Test full window
    X_tr, y_tr, m_tr = load_split_data('train', normalized=True, window='full')
    X_val, y_val, m_val = load_split_data('val', normalized=True, window='full')
    X_te, y_te, m_te = load_split_data('test', normalized=True, window='full')

    print(f"\n[Canonical Full Window (1,000 samples, 0.0-4.0s)]")
    print(f"  Train: X={X_tr.shape}, y={y_tr.shape}, artifacts={m_tr.sum()} | classes: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}, artifacts={m_val.sum()} | classes: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"  Test:  X={X_te.shape}, y={y_te.shape}, artifacts={m_te.sum()} | classes: {dict(zip(*np.unique(y_te, return_counts=True)))}")

    # Test central window
    X_tr_c, _, _ = load_split_data('train', normalized=True, window='central')
    X_val_c, _, _ = load_split_data('val', normalized=True, window='central')
    X_te_c, _, _ = load_split_data('test', normalized=True, window='central')

    print(f"\n[Recommended Central Window (750 samples, 0.5-3.5s)]")
    print(f"  Train: X={X_tr_c.shape}")
    print(f"  Val:   X={X_val_c.shape}")
    print(f"  Test:  X={X_te_c.shape}")

    # Test clean_only mode
    X_tr_clean, y_tr_clean, _ = load_split_data('train', clean_only=True)
    print(f"\n[Clean-Only Mode (Excluding Code 1023)]")
    print(f"  Train Clean: X={X_tr_clean.shape}, y={y_tr_clean.shape} | classes: {dict(zip(*np.unique(y_tr_clean, return_counts=True)))}")

    norm_params = get_normalization_params()
    print(f"\n[Normalization Parameters Fitted on Train]")
    print(f"  Mean across channels: {np.mean(norm_params['channel_means_uV']):.4f} uV")
    print(f"  Std  across channels: {np.mean(norm_params['channel_stds_uV']):.4f} uV")

    print("\nSelf-test PASSED successfully.")
