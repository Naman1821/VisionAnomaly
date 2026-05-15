#!/usr/bin/env python3
"""Train & evaluate PatchCore vs PaDiM and write comparison table."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
app = typer.Typer()
console = Console()


@app.command()
def main(
    config: Path = typer.Option(ROOT / "configs/default.yaml", "--config", "-c"),
    category: str = typer.Option("bottle"),
    models: str = typer.Option("patchcore,padim", help="Comma-separated"),
) -> None:
  names = [m.strip() for m in models.split(",")]
  rows = []
  for name in names:
    console.print(f"\n[bold]=== {name} ===[/]")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/train.py"), "-c", str(config), "--category", category, "--model", name],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/evaluate.py"), "-c", str(config), "--category", category, "--model", name],
        check=True,
    )
    metrics_path = ROOT / "outputs" / category / name / "eval" / "metrics.json"
    with open(metrics_path, encoding="utf-8") as f:
      m = json.load(f)
    rows.append({"model": name, **m})

  df = pd.DataFrame(rows)
  out = ROOT / "outputs" / category / "comparison.csv"
  out.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(out, index=False)
  console.print(df.to_string(index=False))
  console.print(f"\n[green]Saved {out}[/]")


if __name__ == "__main__":
  app()
