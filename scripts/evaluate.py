#!/usr/bin/env python3
"""Evaluate a trained model on MVTec-AD test split."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import os

os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

from visionanomaly.config import load_config, resolve_device  # noqa: E402
from visionanomaly.data.mvtec import build_dataloader  # noqa: E402
from visionanomaly.engine.evaluator import evaluate_model  # noqa: E402
from visionanomaly.models.factory import build_model  # noqa: E402

app = typer.Typer()
console = Console()


@app.command()
def main(
    config: Path = typer.Option(ROOT / "configs/default.yaml", "--config", "-c"),
    category: str | None = None,
    model: str | None = None,
    checkpoint: Path | None = typer.Option(None, help="Override checkpoint dir"),
) -> None:
  cfg = load_config(config)
  if category:
    cfg["data"]["category"] = category
  if model:
    cfg["model"]["name"] = model
  device = resolve_device(cfg.get("device", "auto"))
  cat = cfg["data"]["category"]
  ckpt = checkpoint or Path(cfg["eval"]["output_dir"]) / cat / cfg["model"]["name"]
  if not ckpt.exists():
    raise typer.BadParameter(f"Checkpoint not found: {ckpt}. Run train.py first.")

  detector = build_model(cfg, device)
  detector.load(ckpt)

  test_loader = build_dataloader(
      root=cfg["data"]["root"],
      category=cat,
      split="test",
      image_size=cfg["data"]["image_size"],
      batch_size=cfg["data"]["test_batch_size"],
      num_workers=cfg["data"]["num_workers"],
  )
  out_dir = Path(cfg["eval"]["output_dir"]) / cat / cfg["model"]["name"] / "eval"
  metrics = evaluate_model(
      detector,
      test_loader,
      device,
      out_dir,
      save_heatmaps=cfg["eval"].get("save_heatmaps", True),
      max_heatmaps=cfg["eval"].get("max_heatmap_samples", 32),
  )

  table = Table(title=f"Results — {cat} / {cfg['model']['name']}")
  table.add_column("Metric")
  table.add_column("Value")
  for k, v in metrics.items():
    table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
  console.print(table)
  console.print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
  app()
