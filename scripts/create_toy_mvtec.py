#!/usr/bin/env python3
"""Create a tiny synthetic MVTec-style dataset for local smoke tests (no download)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _random_texture(size: int, seed: int) -> Image.Image:
  rng = np.random.default_rng(seed)
  base = rng.integers(180, 220, (size, size), dtype=np.uint8)
  noise = rng.integers(-15, 15, (size, size))
  img = np.clip(base + noise, 0, 255).astype(np.uint8)
  return Image.fromarray(img, mode="L").convert("RGB")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path("data/mvtec"))
  parser.add_argument("--category", default="bottle")
  parser.add_argument("--size", type=int, default=256)
  args = parser.parse_args()

  root = args.root / args.category
  train_good = root / "train" / "good"
  test_good = root / "test" / "good"
  test_defect = root / "test" / "broken_large"
  gt_dir = root / "ground_truth" / "broken_large"
  for d in (train_good, test_good, test_defect, gt_dir):
    d.mkdir(parents=True, exist_ok=True)

  for i in range(20):
    _random_texture(args.size, seed=i).save(train_good / f"train_{i:03d}.png")
  for i in range(5):
    _random_texture(args.size, seed=100 + i).save(test_good / f"good_{i:03d}.png")
  for i in range(8):
    img = _random_texture(args.size, seed=200 + i)
    draw = ImageDraw.Draw(img)
    draw.ellipse((80, 60, 160, 140), fill=(40, 40, 120))
    img.save(test_defect / f"broken_{i:03d}.png")
    mask = Image.new("L", (args.size, args.size), 0)
    ImageDraw.Draw(mask).ellipse((80, 60, 160, 140), fill=255)
    mask.save(gt_dir / f"broken_{i:03d}_mask.png")

  print(f"Toy MVTec-AD created at {root}")
  print("  train/good: 20 | test/good: 5 | test/broken_large: 8")


if __name__ == "__main__":
  main()
