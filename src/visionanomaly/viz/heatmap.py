from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def normalize_map(score_map: np.ndarray) -> np.ndarray:
  smin, smax = score_map.min(), score_map.max()
  if smax - smin < 1e-8:
    return np.zeros_like(score_map)
  return (score_map - smin) / (smax - smin)


def overlay_heatmap(
    image: np.ndarray,
    score_map: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
  """image: HWC uint8 RGB, score_map: HW float."""
  heat = (plt.cm.jet(normalize_map(score_map))[:, :, :3] * 255).astype(np.uint8)
  if image.shape[:2] != heat.shape[:2]:
    heat = cv2.resize(heat, (image.shape[1], image.shape[0]))
  return cv2.addWeighted(image, 1 - alpha, heat, alpha, 0)


def denormalize_image(t: torch.Tensor) -> np.ndarray:
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
  img = (t.cpu() * std + mean).clamp(0, 1)
  return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_triptych(
    path: Path,
    image: np.ndarray,
    score_map: np.ndarray,
    gt_mask: np.ndarray | None = None,
    title: str = "",
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  ncols = 3 if gt_mask is not None else 2
  fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
  if ncols == 2:
    axes = list(axes)
  axes[0].imshow(image)
  axes[0].set_title("Input")
  axes[0].axis("off")
  axes[1].imshow(overlay_heatmap(image, score_map))
  axes[1].set_title("Anomaly heatmap")
  axes[1].axis("off")
  if gt_mask is not None:
    axes[2].imshow(gt_mask, cmap="gray")
    axes[2].set_title("GT mask")
    axes[2].axis("off")
  if title:
    fig.suptitle(title)
  plt.tight_layout()
  fig.savefig(path, dpi=120, bbox_inches="tight")
  plt.close(fig)
