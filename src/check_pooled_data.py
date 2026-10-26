
import numpy as np
import json
from pathlib import Path

root = Path("data/processed")
data_file = root / "A01-A09T_pooled_split_data.npz"
meta_file = root / "A01-A09T_pooled_split_metadata.json"

print("DATA FILE EXISTS:", data_file.exists())
print("METADATA EXISTS:", meta_file.exists())

with np.load(data_file, allow_pickle=False) as data:
    print("\nARRAYS FOUND:")
    for key in data.files:
        arr = data[key]
        print(
            f"{key}: shape={arr.shape}, dtype={arr.dtype}, "
            f"finite={np.isfinite(arr).all() if np.issubdtype(arr.dtype, np.number) else 'N/A'}"
        )

    print("\nLABEL COUNTS:")
    for key in data.files:
        if "y_" in key.lower() or "label" in key.lower():
            arr = data[key]
            print(key, np.unique(arr, return_counts=True))

with open(meta_file, encoding="utf-8") as f:
    meta = json.load(f)

print("\nMETADATA COUNTS:")
print("Total epochs:", meta.get("total_epochs"))
print("Split counts:", meta.get("split_counts"))
print("Artifact counts:", meta.get("artifact_counts"))
with np.load(data_file, allow_pickle=False) as d:
    # Ensure no trial index appears in multiple splits
    tr, va, te = d["train_indices"], d["val_indices"], d["test_indices"]
    print("\nSplit indices disjoint:",
          len(set(tr) & set(va)) == 0 and
          len(set(tr) & set(te)) == 0 and
          len(set(va) & set(te)) == 0)

    # Check normalized training data is approximately standardized
    print("Normalized train mean (overall):",
          round(float(d["X_train_norm"].mean()), 5))
    print("Normalized train std (overall):",
          round(float(d["X_train_norm"].std()), 5))

print("\nCHECK COMPLETE")

