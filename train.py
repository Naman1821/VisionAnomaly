#!/usr/bin/env python3
"""
VisionAnomaly — MVTec-AD bottle: extract → train PatchCore → demo samples.

Place official category archive at: data/bottle.tar.xz
Uses visionanomaly (src/) — does not modify existing package code.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms as T

from visionanomaly.config import load_config, resolve_device  # noqa: E402
from visionanomaly.data.mvtec import build_dataloader  # noqa: E402
from visionanomaly.models.factory import build_model  # noqa: E402
from visionanomaly.utils.seed import set_seed  # noqa: E402

_infer_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

MODEL_DIR = ROOT / "model"
WEIGHTS_FILE = MODEL_DIR / "patchcore_weights.bin"
SAMPLE_GOOD = ROOT / "sample_images" / "good"
SAMPLE_DEFECT = ROOT / "sample_images" / "defective"
DATA_ROOT = ROOT / "data" / "mvtec"
BOTTLE_TAR = ROOT / "data" / "bottle.tar.xz"
CATEGORY = "bottle"
CONFIG = ROOT / "configs" / "default.yaml"

DEFECT_FOLDERS = ("broken_large", "broken_small", "contamination")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _is_real_bottle() -> bool:
  train_good = DATA_ROOT / CATEGORY / "train" / "good"
  if not train_good.is_dir():
    return False
  n_train = sum(1 for p in train_good.iterdir() if p.suffix.lower() in IMG_EXTS)
  test_root = DATA_ROOT / CATEGORY / "test"
  return n_train >= 50 and all((test_root / d).is_dir() for d in DEFECT_FOLDERS)


def _extract_bottle_tar() -> None:
  """Extract data/bottle.tar.xz → data/mvtec/bottle/."""
  if not BOTTLE_TAR.is_file():
    raise SystemExit(
        f"Missing {BOTTLE_TAR}\n"
        "Download MVTec-AD bottle category and save as data/bottle.tar.xz"
    )

  cat_dir = DATA_ROOT / CATEGORY
  if _is_real_bottle():
    print(f"[skip] Bottle data already at {cat_dir}")
    return

  if cat_dir.is_dir():
    shutil.rmtree(cat_dir)

  DATA_ROOT.mkdir(parents=True, exist_ok=True)
  print(f"Extracting {BOTTLE_TAR.name} → {DATA_ROOT}/...")
  with tarfile.open(BOTTLE_TAR, "r:xz") as tar:
    tar.extractall(path=DATA_ROOT)

  if not _is_real_bottle():
    raise SystemExit("Extract failed — expected data/mvtec/bottle/train/good/ (+ test defects)")


def _calibrate_threshold(detector) -> float:
  """Threshold for k-NN image scores (typically ~3.5–6 on MVTec bottle, not 0–1)."""
  good_dir = DATA_ROOT / CATEGORY / "test" / "good"
  good_scores: list[float] = []
  defect_scores: list[float] = []
  for p in sorted(good_dir.iterdir()):
    if p.suffix.lower() in IMG_EXTS:
      good_scores.append(detector.predict(_infer_transform(Image.open(p).convert("RGB")))[0])
  test_root = DATA_ROOT / CATEGORY / "test"
  for folder in DEFECT_FOLDERS:
    sub = test_root / folder
    if sub.is_dir():
      for p in sorted(sub.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
          defect_scores.append(detector.predict(_infer_transform(Image.open(p).convert("RGB")))[0])
  if not good_scores:
    return 4.1
  # Just above the highest normal test score (k-NN distances ~4 on MVTec bottle)
  return float(max(good_scores) + 0.01)


def _train_patchcore() -> None:
  cfg = load_config(CONFIG)
  cfg["data"]["root"] = str(DATA_ROOT)
  cfg["data"]["category"] = CATEGORY
  cfg["data"]["num_workers"] = 0
  cfg["model"]["name"] = "patchcore"
  set_seed(cfg.get("seed", 42))
  device = resolve_device(cfg.get("device", "auto"))
  print(f"Training PatchCore on real {CATEGORY} ({device})...")

  loader = build_dataloader(
      root=DATA_ROOT,
      category=CATEGORY,
      split="train",
      image_size=cfg["data"]["image_size"],
      batch_size=cfg["data"]["train_batch_size"],
      num_workers=0,
  )
  detector = build_model(cfg, device)
  detector.fit(loader)
  threshold = _calibrate_threshold(detector)
  print(f"Score threshold (normal vs defect): {threshold:.4f}")

  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  ckpt_dir = MODEL_DIR / "patchcore"
  detector.save(ckpt_dir)

  state = torch.load(ckpt_dir / "patchcore.pt", map_location="cpu", weights_only=False)
  state["score_threshold"] = threshold
  torch.save(state, ckpt_dir / "patchcore.pt")
  torch.save(state, WEIGHTS_FILE)
  (MODEL_DIR / "score_threshold.txt").write_text(f"{threshold:.6f}\n", encoding="utf-8")
  print(f"Saved → {WEIGHTS_FILE} (threshold={threshold:.4f})")


def _copy_samples(n_good: int = 6, n_defect: int = 6) -> None:
  """Copy test images for Streamlit demo (real bottle only)."""
  test_root = DATA_ROOT / CATEGORY / "test"
  SAMPLE_GOOD.mkdir(parents=True, exist_ok=True)
  SAMPLE_DEFECT.mkdir(parents=True, exist_ok=True)
  for d in (SAMPLE_GOOD, SAMPLE_DEFECT):
    for old in d.glob("*"):
      if old.is_file():
        old.unlink()

  good_dir = test_root / "good"
  good_files = sorted(p for p in good_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
  random.seed(42)
  for src in random.sample(good_files, min(n_good, len(good_files))):
    shutil.copy2(src, SAMPLE_GOOD / src.name)

  per_type = 2
  random.seed(43)
  picked: list[Path] = []
  for folder in DEFECT_FOLDERS:
    sub = test_root / folder
    pool = sorted(p for p in sub.iterdir() if p.suffix.lower() in IMG_EXTS)
    picked.extend(random.sample(pool, min(per_type, len(pool))))
  for src in picked[:n_defect]:
    shutil.copy2(src, SAMPLE_DEFECT / f"{src.parent.name}_{src.stem}.png")

  print(f"Demo samples: {len(list(SAMPLE_GOOD.glob('*')))} good, {len(list(SAMPLE_DEFECT.glob('*')))} defective")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--extract-only", action="store_true")
  args = parser.parse_args()

  _extract_bottle_tar()
  if args.extract_only:
    print("Extract done.")
    return

  _train_patchcore()
  _copy_samples()
  print("\nDone. Run: streamlit run streamlit_app.py")


if __name__ == "__main__":
  main()
