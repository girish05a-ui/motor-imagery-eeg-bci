"""
PART 0 — Workspace and Dataset Availability Inspection
Role: Person 1 (EEG Data Loading and Preprocessing)
Project: Deep Learning-Based Motor Imagery EEG Classification for a Simulated Real-Time BCI

Checks:
1. Python environment and required libraries (MNE, NumPy, SciPy, scikit-learn, Matplotlib, Pandas, PyTorch).
2. Detection and validation of raw BCI Competition IV Dataset 2a files (*.gdf / *.mat).
3. Reporting of exact file paths, sizes, formats, or missing dataset instructions.
"""

import os
import sys
from pathlib import Path


def inspect_environment():
    print("=" * 70)
    print("PART 0: ENVIRONMENT INSPECTION")
    print("=" * 70)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version.split()[0]} ({sys.platform})")
    print("-" * 70)

    packages = [
        ("mne", "MNE-Python"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("torch", "PyTorch (DL teammate)"),
    ]

    all_core_ok = True
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  [OK]      {display_name:<25} : {version}")
        except ImportError:
            if module_name in ["mne", "numpy", "scipy", "sklearn", "matplotlib"]:
                all_core_ok = False
                print(f"  [MISSING] {display_name:<25} : NOT installed (REQUIRED for Person 1)")
            else:
                print(f"  [OPTIONAL]{display_name:<25} : NOT installed (Model teammate scope)")

    return all_core_ok


def inspect_dataset(workspace_root: Path):
    print("\n" + "=" * 70)
    print("PART 0: DATASET FILES DISCOVERY")
    print("=" * 70)

    raw_dir = workspace_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    search_dirs = [
        raw_dir,
        workspace_root,
        workspace_root.parent,
    ]

    gdf_files = []
    mat_files = []
    zip_files = []

    print(f"Primary raw dataset directory: {raw_dir}")

    # Inspect data/raw first
    for f in raw_dir.glob("*"):
        if f.suffix.lower() == ".gdf":
            gdf_files.append(f)
        elif f.suffix.lower() == ".mat":
            mat_files.append(f)
        elif f.suffix.lower() in [".zip", ".tar", ".gz"]:
            zip_files.append(f)

    # If empty in data/raw, look in immediate parents for convenience
    if not gdf_files and not mat_files and not zip_files:
        for s_dir in search_dirs[1:]:
            if s_dir.exists():
                for f in s_dir.glob("*.gdf"):
                    gdf_files.append(f)
                for f in s_dir.glob("*2a*.zip"):
                    zip_files.append(f)

    print(f"GDF files found:  {len(gdf_files)}")
    print(f"MAT files found:  {len(mat_files)}")
    print(f"ZIP files found:  {len(zip_files)}")

    if gdf_files:
        print("\nDiscovered GDF recordings in data/raw:")
        for idx, gf in enumerate(sorted(gdf_files), 1):
            size_mb = gf.stat().st_size / (1024 * 1024)
            print(f"  {idx:2d}. {gf.name} ({size_mb:.2f} MB) -> {gf}")
        return True, gdf_files
    else:
        print("\n[BLOCKER] No BCI Competition IV Dataset 2a GDF files found.")
        print("-" * 70)
        print("To acquire the official dataset:")
        print("  1. Official GDF Archive URL (420 MB):")
        print("     https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip")
        print("  2. Official True Evaluation Labels URL:")
        print("     https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip")
        print("  3. Official Dataset Description:")
        print("     https://www.bbci.de/competition/iv/desc_2a.pdf")
        print("  4. Placement destination:")
        print(f"     Extract files into: {raw_dir}")
        print("     Expected files: A01T.gdf ... A09T.gdf and A01E.gdf ... A09E.gdf (18 GDF files total)")
        print("-" * 70)
        return False, []


def main():
    workspace_root = Path(__file__).resolve().parent.parent
    core_ok = inspect_environment()
    dataset_found, files = inspect_dataset(workspace_root)

    print("\n" + "=" * 70)
    print("PART 0 COMPLETION STATUS")
    print("=" * 70)
    if core_ok and dataset_found:
        print("STATUS: PASS (Environment ready and dataset files located)")
    elif core_ok and not dataset_found:
        print("STATUS: PENDING DATASET DOWNLOAD")
        print("Environment passes, but dataset files must be placed in data/raw before proceeding to Part 1.")
    else:
        print("STATUS: FAIL (Core environment requirements missing)")
    print("=" * 70)


if __name__ == "__main__":
    main()
