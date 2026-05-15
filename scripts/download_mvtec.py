#!/usr/bin/env python3
"""Download MVTec-AD (Hugging Face mirror) or create toy data for smoke tests."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# Hugging Face dataset mirror (community repack)
HF_REPO = "Voxel51/MVTec"
HF_CATEGORY_FILES = {
    "bottle": "bottle.zip",
}

# Zenodo full dataset (~5GB) — optional
ZENODO_MVTEC = "https://zenodo.org/record/4735652/files/mvtec_anomaly_detection.tar.xz?download=1"


def _download_url(url: str, dest: Path) -> None:
  dest.parent.mkdir(parents=True, exist_ok=True)
  req = urllib.request.Request(url, headers={"User-Agent": "VisionAnomaly/0.1"})
  with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
    shutil.copyfileobj(resp, out)


def download_hf_category(category: str, root: Path) -> bool:
  try:
    from huggingface_hub import hf_hub_download
  except ImportError:
    return False
  dest_dir = root / category
  if (dest_dir / "train" / "good").is_dir():
    print(f"[skip] {category} already at {dest_dir}")
    return True
  fname = HF_CATEGORY_FILES.get(category)
  if not fname:
    print(f"No HF file mapping for {category}; try --toy or --zenodo")
    return False
  print(f"Downloading {category} from Hugging Face ({HF_REPO})...")
  archive = hf_hub_download(
      repo_id=HF_REPO,
      repo_type="dataset",
      filename=fname,
      local_dir=str(root / "_cache"),
  )
  archive = Path(archive)
  if archive.suffix == ".zip":
    with zipfile.ZipFile(archive, "r") as zf:
      zf.extractall(root)
  else:
    with tarfile.open(archive, "r:*") as tar:
      tar.extractall(path=root)
  # Normalize layout: some zips extract to category/ or bottle/
  if not (dest_dir / "train").exists():
    for p in root.iterdir():
      if p.is_dir() and p.name == category and (p / "train").exists():
        if dest_dir.exists():
          shutil.rmtree(dest_dir)
        shutil.move(str(p), str(dest_dir))
        break
  print(f"Done: {dest_dir}")
  return True


def download_zenodo_full(root: Path) -> None:
  archive = root / "mvtec_anomaly_detection.tar.xz"
  print("Downloading full MVTec-AD from Zenodo (~5GB)...")
  _download_url(ZENODO_MVTEC, archive)
  print("Extracting...")
  with tarfile.open(archive, "r:xz") as tar:
    tar.extractall(path=root)
  archive.unlink(missing_ok=True)
  print(f"Done: {root}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Download MVTec-AD")
  parser.add_argument("--root", type=Path, default=Path("data/mvtec"))
  parser.add_argument("--category", default="bottle")
  parser.add_argument("--toy", action="store_true", help="Generate synthetic mini dataset")
  parser.add_argument("--zenodo", action="store_true", help="Download full dataset from Zenodo")
  args = parser.parse_args()

  if args.toy:
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "create_toy_mvtec.py"), "--root", str(args.root), "--category", args.category],
        check=True,
    )
    return

  if args.zenodo:
    download_zenodo_full(args.root)
    return

  if download_hf_category(args.category, args.root):
    return

  print(
      "\nAutomatic download failed. Options:\n"
      "  1) python scripts/download_mvtec.py --toy\n"
      "  2) python scripts/download_mvtec.py --zenodo\n"
      "  3) Manual: https://www.mvtec.com/company/research/datasets/mvtec-ad\n"
      "     Extract to data/mvtec/<category>/train|test|ground_truth\n"
  )
  raise SystemExit(1)


if __name__ == "__main__":
  main()
