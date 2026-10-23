"""
Dataset Downloader for BCI Competition IV Dataset 2a
Downloads the official GDF recordings and true labels from the Berlin BCI repository.
Target directory: data/raw/
"""

import os
import sys
import zipfile
from pathlib import Path
import requests

DATASET_GDF_URL = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
TRUE_LABELS_URL = "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip"


def download_with_progress(url: str, output_path: Path):
    print(f"Downloading from {url} ...")
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        mb_down = downloaded / (1024 * 1024)
                        mb_tot = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  Progress: {mb_down:.1f} MB / {mb_tot:.1f} MB ({percent:.1f}%)")
                    else:
                        sys.stdout.write(f"\r  Progress: {downloaded / (1024 * 1024):.1f} MB")
                    sys.stdout.flush()

        if temp_path.exists():
            temp_path.replace(output_path)
        print("\nDownload complete.")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download from {url}: {e}") from e


def extract_zip(zip_path: Path, target_dir: Path):
    print(f"Extracting {zip_path.name} to {target_dir} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction complete.")


def main():
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download GDF archive (420 MB)
    gdf_zip = raw_dir / "BCICIV_2a_gdf.zip"
    if not gdf_zip.exists():
        download_with_progress(DATASET_GDF_URL, gdf_zip)
    else:
        print(f"Found existing {gdf_zip.name}, skipping download.")

    extract_zip(gdf_zip, raw_dir)

    # 2. Download True evaluation labels
    labels_zip = raw_dir / "true_labels.zip"
    if not labels_zip.exists():
        download_with_progress(TRUE_LABELS_URL, labels_zip)
    else:
        print(f"Found existing {labels_zip.name}, skipping download.")

    extract_zip(labels_zip, raw_dir)

    print("\nDataset is successfully downloaded and unpacked in data/raw.")
    print("Run `python src/inspect_dataset.py` to verify.")


if __name__ == "__main__":
    main()
