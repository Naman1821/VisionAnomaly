#!/usr/bin/env python3
"""
VisionAnomaly — MVTec-AD: extract → train PatchCore → demo samples.

Place category archives at: data/bottle.tar.xz, data/capsule.tar.xz, …
Uses visionanomaly (src/) — does not modify existing package code.
"""

from __future__ import annotations

import argparse
import json
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
from visionanomaly.engine.evaluator import evaluate_model  # noqa: E402
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
DATA_ROOT = ROOT / "data" / "mvtec"
SAMPLE_ROOT = ROOT / "sample_images"
CONFIG = ROOT / "configs" / "default.yaml"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# Per-category MVTec test defect folder names
CATEGORY_DEFECTS: dict[str, tuple[str, ...]] = {
    "bottle": ("broken_large", "broken_small", "contamination"),
    "capsule": ("crack", "faulty_imprint", "poke", "scratch", "squeeze"),
}

DEFAULT_CATEGORIES = ("bottle", "capsule")


def _tar_path(category: str) -> Path:
  return ROOT / "data" / f"{category}.tar.xz"


def _weights_path(category: str) -> Path:
  return MODEL_DIR / f"patchcore_{category}_weights.bin"


def _threshold_path(category: str) -> Path:
  return MODEL_DIR / f"score_threshold_{category}.txt"


def _is_real_category(category: str) -> bool:
  train_good = DATA_ROOT / category / "train" / "good"
  if not train_good.is_dir():
    return False
  n_train = sum(1 for p in train_good.iterdir() if p.suffix.lower() in IMG_EXTS)
  test_root = DATA_ROOT / category / "test"
  defects = CATEGORY_DEFECTS.get(category, ())
  return n_train >= 50 and all((test_root / d).is_dir() for d in defects)


def _extract_category_tar(category: str) -> None:
  tar_path = _tar_path(category)
  if not tar_path.is_file():
    raise SystemExit(f"Missing {tar_path}")

  cat_dir = DATA_ROOT / category
  if _is_real_category(category):
    print(f"[skip] {category} data already at {cat_dir}")
    return

  if cat_dir.is_dir():
    shutil.rmtree(cat_dir)

  DATA_ROOT.mkdir(parents=True, exist_ok=True)
  print(f"Extracting {tar_path.name} → {DATA_ROOT}/...")
  with tarfile.open(tar_path, "r:xz") as tar:
    tar.extractall(path=DATA_ROOT)

  if not _is_real_category(category):
    raise SystemExit(f"Extract failed for {category} — check archive layout.")


def _calibrate_threshold(detector, category: str) -> float:
  """Threshold from normal test images (k-NN distances, category-specific scale)."""
  good_dir = DATA_ROOT / category / "test" / "good"
  good_scores: list[float] = []
  for p in sorted(good_dir.iterdir()):
    if p.suffix.lower() in IMG_EXTS:
      good_scores.append(detector.predict(_infer_transform(Image.open(p).convert("RGB")))[0])
  if not good_scores:
    return 4.1
  return float(max(good_scores) + 0.01)


def _train_category(category: str) -> dict:
  cfg = load_config(CONFIG)
  cfg["data"]["root"] = str(DATA_ROOT)
  cfg["data"]["category"] = category
  cfg["data"]["num_workers"] = 0
  cfg["model"]["name"] = "patchcore"
  set_seed(cfg.get("seed", 42))
  device = resolve_device(cfg.get("device", "auto"))
  print(f"\nTraining PatchCore on {category} ({device})...")

  loader = build_dataloader(
      root=DATA_ROOT,
      category=category,
      split="train",
      image_size=cfg["data"]["image_size"],
      batch_size=cfg["data"]["train_batch_size"],
      num_workers=0,
  )
  detector = build_model(cfg, device)
  detector.fit(loader)
  threshold = _calibrate_threshold(detector, category)
  print(f"  threshold ({category}): {threshold:.4f}")

  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  ckpt_dir = MODEL_DIR / f"patchcore_{category}"
  detector.save(ckpt_dir)

  state = torch.load(ckpt_dir / "patchcore.pt", map_location="cpu", weights_only=False)
  state["score_threshold"] = threshold
  state["category"] = category
  torch.save(state, ckpt_dir / "patchcore.pt")
  weights = _weights_path(category)
  torch.save(state, weights)
  _threshold_path(category).write_text(f"{threshold:.6f}\n", encoding="utf-8")

  # Legacy single-file name for bottle (Streamlit cloud / older links)
  if category == "bottle":
    torch.save(state, MODEL_DIR / "patchcore_weights.bin")
    (MODEL_DIR / "score_threshold.txt").write_text(f"{threshold:.6f}\n", encoding="utf-8")

  print(f"  saved → {weights}")

  test_loader = build_dataloader(
      root=DATA_ROOT,
      category=category,
      split="test",
      image_size=cfg["data"]["image_size"],
      batch_size=1,
      num_workers=0,
  )
  metrics = evaluate_model(detector, test_loader, device, MODEL_DIR / f"eval_{category}", save_heatmaps=False)
  print(f"  eval: image_auroc={metrics['image_auroc']:.4f} pixel_auroc={metrics.get('pixel_auroc', 0):.4f}")
  return metrics


def _copy_samples(category: str, n_good: int = 3, n_defect: int = 3) -> None:
  """Copy test images into sample_images/{category}/good|defective/."""
  test_root = DATA_ROOT / category / "test"
  good_out = SAMPLE_ROOT / category / "good"
  defect_out = SAMPLE_ROOT / category / "defective"
  good_out.mkdir(parents=True, exist_ok=True)
  defect_out.mkdir(parents=True, exist_ok=True)
  for d in (good_out, defect_out):
    for old in d.glob("*"):
      if old.is_file():
        old.unlink()

  good_files = sorted(
      p for p in (test_root / "good").iterdir() if p.suffix.lower() in IMG_EXTS
  )
  random.seed(42 if category == "bottle" else 142)
  for src in random.sample(good_files, min(n_good, len(good_files))):
    shutil.copy2(src, good_out / src.name)

  defects = CATEGORY_DEFECTS.get(category, ())
  per_type = max(1, n_defect // max(len(defects), 1))
  random.seed(43 if category == "bottle" else 143)
  picked: list[Path] = []
  for folder in defects:
    sub = test_root / folder
    if not sub.is_dir():
      continue
    pool = sorted(p for p in sub.iterdir() if p.suffix.lower() in IMG_EXTS)
    picked.extend(random.sample(pool, min(per_type, len(pool))))
  for src in picked[:n_defect]:
    shutil.copy2(src, defect_out / f"{src.parent.name}_{src.stem}.png")

  print(f"  samples/{category}: {len(list(good_out.glob('*')))} good, {len(list(defect_out.glob('*')))} defective")


def _save_metrics_table(all_metrics: dict[str, dict]) -> None:
  out = MODEL_DIR / "metrics_summary.json"
  out.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
  print(f"\nMetrics summary → {out}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--category",
      default="all",
      help="bottle | capsule | all (default: all)",
  )
  parser.add_argument("--extract-only", action="store_true")
  parser.add_argument("--skip-train", action="store_true", help="Only extract + samples")
  args = parser.parse_args()

  if args.category == "all":
    categories = list(DEFAULT_CATEGORIES)
  else:
    categories = [args.category]

  for cat in categories:
    _extract_category_tar(cat)

  if args.extract_only:
    print("Extract done.")
    return

  all_metrics: dict[str, dict] = {}
  if not args.skip_train:
    for cat in categories:
      m = _train_category(cat)
      all_metrics[cat] = {
          "image_auroc": round(float(m["image_auroc"]), 4),
          "pixel_auroc": round(float(m.get("pixel_auroc", 0)), 4),
          "num_test": int(m["num_test"]),
      }

  for cat in categories:
    _copy_samples(cat, n_good=3, n_defect=3)

  if all_metrics:
    _save_metrics_table(all_metrics)

  print("\nDone. Run: streamlit run streamlit_app.py")


if __name__ == "__main__":
  main()
