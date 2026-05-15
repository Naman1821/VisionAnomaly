from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from visionanomaly.metrics.auroc import pixel_auroc, safe_auroc
from visionanomaly.viz.heatmap import denormalize_image, save_triptych


def evaluate_model(model, dataloader, device: str, output_dir: Path, save_heatmaps: bool = True, max_heatmaps: int = 32):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  heatmap_dir = output_dir / "heatmaps"
  if save_heatmaps:
    heatmap_dir.mkdir(exist_ok=True)

  image_scores, image_labels = [], []
  pixel_scores_list, pixel_masks_list = [], []
  saved = 0

  for batch in tqdm(dataloader, desc="Evaluate"):
    images = batch["image"]
    labels = batch["label"].numpy()
    masks = batch["mask"]
    paths = batch["path"]

    for i in range(images.shape[0]):
      score, smap = model.predict(images[i])
      image_scores.append(score)
      image_labels.append(labels[i])

      if masks is not None and labels[i] == 1:
        gt = masks[i, 0].numpy()
        pixel_scores_list.append(smap)
        pixel_masks_list.append(gt)

      if save_heatmaps and saved < max_heatmaps:
        img_np = denormalize_image(images[i])
        gt_mask = masks[i, 0].numpy() if masks is not None and masks[i] is not None else None
        name = Path(paths[i]).stem
        save_triptych(
            heatmap_dir / f"{saved:03d}_{name}.png",
            img_np,
            smap,
            gt_mask,
            title=f"score={score:.3f} label={'defect' if labels[i] else 'ok'}",
        )
        saved += 1

  metrics = {
      "image_auroc": safe_auroc(np.array(image_labels), np.array(image_scores)),
      "num_test": len(image_labels),
      "num_anomaly": int(np.sum(image_labels)),
  }
  if pixel_scores_list:
    metrics["pixel_auroc"] = pixel_auroc(
        np.stack(pixel_masks_list), np.stack(pixel_scores_list)
    )
  with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
  return metrics
