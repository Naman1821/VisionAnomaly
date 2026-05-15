#!/usr/bin/env python3
"""Gradio demo — upload image, get anomaly score + heatmap."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionanomaly.config import resolve_device  # noqa: E402
from visionanomaly.models.factory import build_model  # noqa: E402
from visionanomaly.viz.heatmap import overlay_heatmap  # noqa: E402

_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_detector = None
_threshold = 0.5


def _load_detector(checkpoint: Path, config_path: Path):
  global _detector, _threshold
  with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
  device = resolve_device(cfg.get("device", "auto"))
  _detector = build_model(cfg, device)
  _detector.load(checkpoint)
  metrics_path = checkpoint / "eval" / "metrics.json"
  if metrics_path.exists():
    import json

    with open(metrics_path, encoding="utf-8") as f:
      m = json.load(f)
    # Heuristic threshold from normal scores would need calibration; use fixed for demo
    _threshold = 0.5
  return _detector


def predict(image: np.ndarray):
  if _detector is None:
    return "Model not loaded. Set CHECKPOINT env or use CLI.", None, None
  if image is None:
    return "Upload an image.", None, None
  pil = Image.fromarray(image).convert("RGB")
  tensor = _transform(pil)
  score, smap = _detector.predict(tensor)
  label = "DEFECT" if score > _threshold else "OK"
  overlay = overlay_heatmap(np.array(pil.resize((256, 256))), smap)
  summary = (
      f"**Score:** {score:.4f}\n\n"
      f"**Prediction:** {label}\n\n"
      f"*(Higher score = more anomalous. Train on your category for best results.)*"
  )
  return summary, overlay, smap


def build_ui(checkpoint: Path, config_path: Path) -> gr.Blocks:
  _load_detector(checkpoint, config_path)
  with gr.Blocks(title="VisionAnomaly", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # VisionAnomaly
        **Industrial defect detection** — PatchCore / PaDiM on MVTec-AD.
        Upload a product image; get anomaly score + heatmap (red = suspicious region).
        """
    )
    with gr.Row():
      inp = gr.Image(type="numpy", label="Input image")
      out_overlay = gr.Image(label="Heatmap overlay")
    out_text = gr.Markdown()
    out_raw = gr.Image(label="Raw score map", visible=False)
    inp.change(predict, inputs=inp, outputs=[out_text, out_overlay, out_raw])
  return demo


if __name__ == "__main__":
  import os

  ckpt = Path(os.environ.get("CHECKPOINT", ROOT / "outputs/bottle/patchcore"))
  cfg = Path(os.environ.get("CONFIG", ROOT / "configs/default.yaml"))
  demo = build_ui(ckpt, cfg)
  demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
