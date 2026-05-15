#!/usr/bin/env python3
"""Train PatchCore or PaDiM on MVTec-AD normal images."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import os

os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

from visionanomaly.config import load_config, resolve_device  # noqa: E402
from visionanomaly.data.mvtec import build_dataloader  # noqa: E402
from visionanomaly.models.factory import build_model  # noqa: E402
from visionanomaly.utils.seed import set_seed  # noqa: E402

app = typer.Typer(help="Train VisionAnomaly model")
console = Console()


@app.command()
def main(
    config: Path = typer.Option(ROOT / "configs/default.yaml", "--config", "-c"),
    category: str | None = typer.Option(None, help="Override data.category"),
    model: str | None = typer.Option(None, help="Override model.name"),
) -> None:
  cfg = load_config(config)
  if category:
    cfg["data"]["category"] = category
  if model:
    cfg["model"]["name"] = model
  set_seed(cfg.get("seed", 42))
  device = resolve_device(cfg.get("device", "auto"))
  cat = cfg["data"]["category"]
  console.print(Panel(f"Training [bold]{cfg['model']['name']}[/] on [cyan]{cat}[/] ({device})"))

  train_loader = build_dataloader(
      root=cfg["data"]["root"],
      category=cat,
      split="train",
      image_size=cfg["data"]["image_size"],
      batch_size=cfg["data"]["train_batch_size"],
      num_workers=cfg["data"]["num_workers"],
      max_samples=cfg["train"].get("max_train_samples"),
  )
  detector = build_model(cfg, device)
  detector.fit(train_loader)

  ckpt_dir = Path(cfg["eval"]["output_dir"]) / cat / cfg["model"]["name"]
  detector.save(ckpt_dir)
  console.print(f"[green]Saved checkpoint → {ckpt_dir}[/]")


if __name__ == "__main__":
  app()
