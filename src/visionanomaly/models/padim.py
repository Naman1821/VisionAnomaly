from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from visionanomaly.features.extractor import FeatureExtractor


def _patchify(feat: torch.Tensor) -> torch.Tensor:
  b, c, h, w = feat.shape
  return feat.permute(0, 2, 3, 1).reshape(b * h * w, c)


class PaDiM:
  """PaDiM: per-patch multivariate Gaussian on reduced features."""

  def __init__(
      self,
      backbone: str = "wide_resnet50_2",
      layers: list[str] | None = None,
      n_components: int = 100,
      device: str = "cpu",
  ) -> None:
    self.device = device
    self.n_components = n_components
    self.extractor = FeatureExtractor(backbone, layers).to(device).eval()
    self.mean: torch.Tensor | None = None
    self.cov_inv: torch.Tensor | None = None
    self.h = self.w = 0
    self._idx: torch.Tensor | None = None

  @torch.no_grad()
  def fit(self, dataloader) -> None:
    feats_list = []
    for batch in tqdm(dataloader, desc="PaDiM fit"):
      x = batch["image"].to(self.device)
      feat = self.extractor.embed(x)
      feats_list.append(feat.cpu())
    all_feat = torch.cat(feats_list, dim=0)
    _, c, h, w = all_feat.shape
    self.h, self.w = h, w
    patches = all_feat.permute(0, 2, 3, 1).numpy()  # N,H,W,C
    rng = np.random.default_rng(42)
    if c > self.n_components:
      self._idx = torch.from_numpy(
          rng.choice(c, size=self.n_components, replace=False)
      )
    else:
      self._idx = torch.arange(c)
    d = len(self._idx)
    mean = np.zeros((h, w, d), dtype=np.float32)
    cov = np.zeros((h, w, d, d), dtype=np.float32)
    for i in range(h):
      for j in range(w):
        vectors = patches[:, i, j, self._idx.numpy()]
        mean[i, j] = vectors.mean(axis=0)
        cov[i, j] = np.cov(vectors.T) + np.eye(d) * 1e-6
    self.mean = torch.from_numpy(mean).float().to(self.device)
    self.cov_inv = torch.from_numpy(np.linalg.inv(cov)).float().to(self.device)

  @torch.no_grad()
  def predict(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
    x = x.to(self.device)
    if x.dim() == 3:
      x = x.unsqueeze(0)
    feat = self.extractor.embed(x)
    _, c, h, w = feat.shape
    selected = feat[0, self._idx.to(self.device)]  # d,H,W
    score_map = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
      for j in range(w):
        v = selected[:, i, j]
        delta = v - self.mean[i, j]
        maha = torch.einsum("i,ij,j->", delta, self.cov_inv[i, j], delta)
        score_map[i, j] = float(maha.sqrt())
    score_map = F.interpolate(
        torch.from_numpy(score_map).view(1, 1, h, w).float(),
        size=(x.shape[2], x.shape[3]),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    return float(score_map.max()), score_map

  def save(self, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": self.mean.cpu(),
            "cov_inv": self.cov_inv.cpu(),
            "idx": self._idx.cpu(),
            "h": self.h,
            "w": self.w,
            "n_components": self.n_components,
        },
        path / "padim.pt",
    )

  def load(self, path: str | Path) -> None:
    path = Path(path)
    state = torch.load(path / "padim.pt", map_location=self.device, weights_only=False)
    self.mean = state["mean"].to(self.device)
    self.cov_inv = state["cov_inv"].to(self.device)
    self._idx = state["idx"].to(self.device)
    self.h = state["h"]
    self.w = state["w"]
    self.n_components = state["n_components"]
