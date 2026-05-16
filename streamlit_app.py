#!/usr/bin/env python3
"""
VisionAnomaly — Streamlit demo (MVTec-AD bottle / PatchCore).

Uses existing visionanomaly.models.PatchCore and viz helpers — no changes to src/.
Load weights from model/patchcore_weights.bin (created by train.py).

Note: root app.py is the existing Hugging Face Gradio entry and is not modified.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
if Path("/mount/src").is_dir():
  os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _on_streamlit_cloud() -> bool:
  """Streamlit Community Cloud mounts repo at /mount/src."""
  return Path("/mount/src").is_dir()

from visionanomaly.config import resolve_device  # noqa: E402
from visionanomaly.models.patchcore import PatchCore  # noqa: E402
from visionanomaly.viz.heatmap import normalize_map, overlay_heatmap  # noqa: E402

WEIGHTS = ROOT / "model" / "patchcore_weights.bin"
CONFIG = ROOT / "configs" / "default.yaml"
SAMPLE_GOOD = ROOT / "sample_images" / "good"
SAMPLE_DEFECT = ROOT / "sample_images" / "defective"
THRESHOLD_FILE = ROOT / "model" / "score_threshold.txt"
THUMB_W = 64

_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_COMPACT_CSS = """
<style>
  div[data-testid="stSidebar"] + div .sample-list-row {
    padding: 2px 0;
    border-bottom: 1px solid rgba(128,128,128,0.2);
  }
  .sample-name {
    font-size: 0.85rem;
    line-height: 1.2;
    margin: 0;
    word-break: break-all;
  }
</style>
"""


def _load_cfg() -> dict:
  with open(CONFIG, encoding="utf-8") as f:
    return yaml.safe_load(f)


@st.cache_resource
def _load_model(_weights_version: float) -> tuple[PatchCore, float]:
  """Load PatchCore memory bank from model/patchcore_weights.bin."""
  if not WEIGHTS.is_file():
    raise FileNotFoundError(f"Missing {WEIGHTS}. Run: python train.py")
  cfg = _load_cfg()
  # Cloud free tier: CPU only (no CUDA/MPS)
  device = "cpu" if _on_streamlit_cloud() else resolve_device(cfg.get("device", "auto"))
  detector = PatchCore(
      backbone=cfg["model"]["backbone"],
      layers=cfg["model"]["layers"],
      coreset_ratio=cfg["model"]["coreset_sampling_ratio"],
      num_neighbors=cfg["model"]["num_neighbors"],
      device=device,
  )
  state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
  detector.memory_bank = state["memory_bank"].to(device)
  detector.coreset_ratio = state["coreset_ratio"]
  detector.num_neighbors = state["num_neighbors"]
  detector._rp = state.get("rp")
  threshold = float(state.get("score_threshold", 0.0))
  if threshold <= 0 and THRESHOLD_FILE.is_file():
    threshold = float(THRESHOLD_FILE.read_text().strip())
  if threshold <= 0:
    threshold = 4.09  # fallback for real MVTec bottle k-NN scores
  return detector, threshold


def _list_samples(folder: Path) -> list[Path]:
  if not folder.is_dir():
    return []
  exts = {".png", ".jpg", ".jpeg", ".bmp"}
  return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def _sample_list_row(path: Path, key: str) -> None:
  """Compact row: small thumbnail, filename, Run button."""
  c_img, c_name, c_btn = st.columns([1, 5, 1], gap="small")
  with c_img:
    st.image(str(path), width=THUMB_W)
  with c_name:
    st.markdown(f'<p class="sample-name">{path.name}</p>', unsafe_allow_html=True)
  with c_btn:
    if st.button("Run", key=key, use_container_width=True):
      st.session_state.selected_path = str(path)
      st.rerun()


def _run_inference(
    detector: PatchCore, pil: Image.Image, threshold: float
) -> tuple[float, str, np.ndarray, np.ndarray, np.ndarray]:
  """Return score, label, RGB, jet heatmap, overlay (all uint8 RGB)."""
  tensor = _transform(pil.convert("RGB"))
  score, smap = detector.predict(tensor)
  label = "DEFECT" if score > threshold else "OK"
  rgb = np.array(pil.resize((256, 256)).convert("RGB"))
  norm = (normalize_map(smap) * 255).astype(np.uint8)
  heat = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
  overlay = np.asarray(overlay_heatmap(rgb, smap), dtype=np.uint8)
  return score, label, rgb, heat, overlay


def _show_result(
    score: float,
    label: str,
    rgb: np.ndarray,
    heat: np.ndarray,
    overlay: np.ndarray,
    threshold: float,
) -> None:
  c1, c2, c3 = st.columns(3)
  c1.image(rgb, caption="Original", use_container_width=True)
  c2.image(heat, caption="Anomaly heatmap", use_container_width=True)
  c3.image(overlay, caption="Overlay", use_container_width=True)

  st.metric(
      "Anomaly score",
      f"{score:.4f}",
      help="PatchCore k-NN distance (higher = more anomalous; threshold calibrated on normal bottles)",
  )
  if label == "OK":
    st.success("✅ NORMAL")
  else:
    st.error("🚨 DEFECT DETECTED")
  st.caption(f"Threshold: {threshold:.3f} (calibrated on normal bottles). Score above → defect.")


def main() -> None:
  st.set_page_config(page_title="VisionAnomaly", page_icon="🔬", layout="wide")
  st.markdown(_COMPACT_CSS, unsafe_allow_html=True)

  if "selected_path" not in st.session_state:
    st.session_state.selected_path = None

  left, right = st.columns([1, 2])

  with left:
    st.title("VisionAnomaly")
    st.subheader("Industrial Defect Detection")
    st.markdown("**Trained on MVTec-AD Bottle (PatchCore)**")
    st.warning(
        "⚠️ Model trained on real MVTec-AD bottle images. "
        "Pick a sample below — only bundled demo images are supported."
    )

    st.markdown("#### Samples")
    good_files = _list_samples(SAMPLE_GOOD)
    defect_files = _list_samples(SAMPLE_DEFECT)

    if not good_files and not defect_files:
      st.info("No samples found. Run `python train.py` first.")
    else:
      if good_files:
        st.markdown("**Normal**")
        for p in good_files:
          _sample_list_row(p, key=f"g_{p.name}")

      if defect_files:
        st.markdown("**Defective**")
        for p in defect_files:
          _sample_list_row(p, key=f"d_{p.name}")

      if st.session_state.selected_path is None and good_files:
        st.session_state.selected_path = str(good_files[0])

  with right:
    st.subheader("Results")
    try:
      weights_ver = WEIGHTS.stat().st_mtime if WEIGHTS.is_file() else 0.0
      detector, threshold = _load_model(weights_ver)
    except FileNotFoundError as e:
      st.error(str(e))
      st.stop()
    except Exception as e:
      st.error(f"Failed to load model: {e}")
      st.stop()

    sel = st.session_state.selected_path
    if not sel or not Path(sel).is_file():
      st.info("Select a sample on the left and click **Run**.")
      st.stop()

    try:
      pil = Image.open(sel).convert("RGB")
    except Exception:
      st.error("Could not open selected image.")
      st.stop()

    st.caption(f"**{Path(sel).name}**")
    with st.spinner("Running PatchCore…"):
      score, label, rgb, heat, overlay = _run_inference(detector, pil, threshold)
    _show_result(score, label, rgb, heat, overlay, threshold)


if __name__ == "__main__":
  main()
