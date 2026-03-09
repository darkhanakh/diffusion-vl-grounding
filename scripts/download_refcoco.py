"""Download RefCOCO/+/g datasets and COCO train2014 images."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path("data")

REFCOCO_URLS = {
    "refcoco": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
    "refcoco+": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
    "refcocog": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
}

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/train2014.zip"


def download_and_extract(url: str, target_dir: Path) -> None:
    """Download a zip file and extract it."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_name = url.split("/")[-1]
    zip_path = target_dir / zip_name

    if not zip_path.exists():
        print(f"Downloading {url}...")
        subprocess.run(
            ["curl", "-L", "-o", str(zip_path), url],
            check=True,
        )

    print(f"Extracting {zip_name}...")
    subprocess.run(
        ["unzip", "-q", "-o", str(zip_path), "-d", str(target_dir)],
        check=True,
    )


def main() -> None:
    print("=== Downloading RefCOCO datasets ===")
    for name, url in REFCOCO_URLS.items():
        dataset_dir = DATA_ROOT / name
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"  {name} already exists, skipping.")
            continue
        download_and_extract(url, DATA_ROOT)

    print("\n=== Downloading COCO train2014 images ===")
    images_dir = DATA_ROOT / "images" / "train2014"
    if images_dir.exists() and any(images_dir.iterdir()):
        print("  COCO images already exist, skipping.")
    else:
        print("  WARNING: COCO train2014 is ~13GB. Download manually if needed:")
        print(f"  curl -L -o data/train2014.zip {COCO_IMAGES_URL}")
        print("  unzip data/train2014.zip -d data/images/")

    print("\nDone! Dataset structure:")
    for p in sorted(DATA_ROOT.rglob("*")):
        if p.is_dir():
            print(f"  {p}/")


if __name__ == "__main__":
    main()
