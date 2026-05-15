#!/usr/bin/env python3
"""FastAPI inference server for VisionAnomaly."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
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

app = FastAPI(title="VisionAnomaly API", version="0.1.0")
_detector = None


@app.on_event("startup")
def load_model() -> None:
  global _detector
  cfg_path = Path(os.environ.get("CONFIG", ROOT / "configs/default.yaml"))
  ckpt = Path(os.environ.get("CHECKPOINT", ROOT / "outputs/bottle/patchcore"))
  with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
  device = resolve_device(cfg.get("device", "auto"))
  _detector = build_model(cfg, device)
  _detector.load(ckpt)


@app.get("/health")
def health():
  return {"status": "ok", "model_loaded": _detector is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  if _detector is None:
    return JSONResponse({"error": "model not loaded"}, status_code=503)
  raw = await file.read()
  pil = Image.open(io.BytesIO(raw)).convert("RGB")
  tensor = _transform(pil)
  score, smap = _detector.predict(tensor)
  overlay = overlay_heatmap(np.array(pil.resize((256, 256))), smap)
  import numpy as np

  buf = io.BytesIO()
  Image.fromarray(overlay).save(buf, format="PNG")
  buf.seek(0)
  return JSONResponse(
      {
          "score": score,
          "label": "defect" if score > 0.5 else "ok",
          "heatmap_png": "see multipart — use /predict/json for score only",
      }
  )


@app.post("/predict/json")
async def predict_json(file: UploadFile = File(...)):
  if _detector is None:
    return JSONResponse({"error": "model not loaded"}, status_code=503)
  raw = await file.read()
  pil = Image.open(io.BytesIO(raw)).convert("RGB")
  tensor = _transform(pil)
  score, _ = _detector.predict(tensor)
  return {"score": score, "label": "defect" if score > 0.5 else "ok"}
