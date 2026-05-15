"""
VisionAnomaly — Hugging Face Spaces entry (Gradio).

Upload a product image → anomaly score + defect heatmap (PatchCore).
First cold start trains a small demo model on synthetic MVTec-style data (~1–3 min on CPU).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import yaml
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

from visionanomaly.config import load_config, resolve_device  # noqa: E402
from visionanomaly.data.mvtec import build_dataloader  # noqa: E402
from visionanomaly.models.factory import build_model  # noqa: E402
from visionanomaly.utils.seed import set_seed  # noqa: E402
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
_CKPT = ROOT / "checkpoints" / "bottle" / "patchcore"
_CFG = ROOT / "configs" / "default.yaml"


def _bootstrap_checkpoint() -> Path:
  """Train PatchCore on toy data if no checkpoint (HF cold start)."""
  if (_CKPT / "patchcore.pt").exists():
    return _CKPT

  import subprocess

  subprocess.run(
      [
          sys.executable,
          str(ROOT / "scripts" / "create_toy_mvtec.py"),
          "--root",
          str(ROOT / "data" / "mvtec"),
          "--category",
          "bottle",
      ],
      check=True,
  )

  cfg = load_config(_CFG)
  cfg["eval"]["output_dir"] = str(ROOT / "checkpoints")
  cfg["data"]["root"] = str(ROOT / "data" / "mvtec")
  cfg["data"]["category"] = "bottle"
  cfg["data"]["num_workers"] = 0
  cfg["model"]["name"] = "patchcore"
  set_seed(cfg.get("seed", 42))
  device = resolve_device(cfg.get("device", "auto"))

  train_loader = build_dataloader(
      root=cfg["data"]["root"],
      category=cfg["data"]["category"],
      split="train",
      image_size=cfg["data"]["image_size"],
      batch_size=cfg["data"]["train_batch_size"],
      num_workers=0,
  )
  model = build_model(cfg, device)
  model.fit(train_loader)
  model.save(_CKPT)
  return _CKPT


def _load_detector() -> None:
  global _detector
  if _detector is not None:
    return
  ckpt = _bootstrap_checkpoint()
  with open(_CFG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
  device = resolve_device(cfg.get("device", "auto"))
  _detector = build_model(cfg, device)
  _detector.load(ckpt)


def predict(image: np.ndarray):
  if image is None:
    return "Upload an image to analyze.", None
  _load_detector()
  pil = Image.fromarray(image).convert("RGB")
  tensor = _transform(pil)
  score, smap = _detector.predict(tensor)
  label = "**DEFECT**" if score > _threshold else "**OK**"
  overlay = overlay_heatmap(np.array(pil.resize((256, 256))), smap)
  text = (
      f"### VisionAnomaly\n\n"
      f"**Score:** `{score:.4f}` (higher = more anomalous)\n\n"
      f"**Result:** {label}\n\n"
      f"Red regions = model suspects a defect.\n\n"
      f"[GitHub](https://github.com/Naman1821/VisionAnomaly)"
  )
  return text, overlay


with gr.Blocks(title="VisionAnomaly", theme=gr.themes.Soft()) as demo:
  gr.Markdown(
      """
      # VisionAnomaly
      **Industrial defect detection** — PatchCore on MVTec-style data.
      Upload a product image; get an anomaly score and heatmap.
      """
  )
  with gr.Row():
    inp = gr.Image(type="numpy", label="Product image")
    out_img = gr.Image(label="Anomaly heatmap")
  out_md = gr.Markdown()
  gr.Examples(
      examples=[
          [str(ROOT / "docs/samples/normal_example.png")],
          [str(ROOT / "docs/samples/defect_example.png")],
      ],
      inputs=inp,
      label="Sample images",
  )
  btn = gr.Button("Analyze", variant="primary")
  btn.click(predict, inputs=inp, outputs=[out_md, out_img])
  inp.change(predict, inputs=inp, outputs=[out_md, out_img])

if __name__ == "__main__":
  demo.launch()
