from __future__ import annotations

from visionanomaly.models.padim import PaDiM
from visionanomaly.models.patchcore import PatchCore


def build_model(cfg: dict, device: str):
  name = cfg["model"]["name"].lower()
  if name == "patchcore":
    return PatchCore(
        backbone=cfg["model"]["backbone"],
        layers=cfg["model"]["layers"],
        coreset_ratio=cfg["model"]["coreset_sampling_ratio"],
        num_neighbors=cfg["model"]["num_neighbors"],
        device=device,
    )
  if name == "padim":
    return PaDiM(
        backbone=cfg["model"]["backbone"],
        layers=cfg["model"]["layers"],
        n_components=cfg["model"]["padim_dim"],
        device=device,
    )
  raise ValueError(f"Unknown model: {name}")
