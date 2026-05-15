from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.random_projection import SparseRandomProjection
from tqdm import tqdm

from visionanomaly.features.extractor import FeatureExtractor


def _patchify(feat: torch.Tensor) -> torch.Tensor:
  """B,C,H,W -> (B*H*W), C"""
  b, c, h, w = feat.shape
  return feat.permute(0, 2, 3, 1).reshape(b * h * w, c)


class PatchCore:
  """PatchCore anomaly detector with coreset memory bank."""

  def __init__(
      self,
      backbone: str = "wide_resnet50_2",
      layers: list[str] | None = None,
      coreset_ratio: float = 0.1,
      num_neighbors: int = 9,
      device: str = "cpu",
  ) -> None:
    self.device = device
    self.coreset_ratio = coreset_ratio
    self.num_neighbors = num_neighbors
    self.extractor = FeatureExtractor(backbone, layers).to(device).eval()
    self.memory_bank: torch.Tensor | None = None
    self._rp: SparseRandomProjection | None = None

  @torch.no_grad()
  def fit(self, dataloader) -> None:
    embeddings = []
    for batch in tqdm(dataloader, desc="PatchCore fit"):
      x = batch["image"].to(self.device)
      feat = self.extractor.embed(x)
      embeddings.append(_patchify(feat).cpu())
    all_emb = torch.cat(embeddings, dim=0).numpy()
    n = len(all_emb)
    k = max(int(n * self.coreset_ratio), 1)
    # Greedy coreset (k-center) on random projection for speed
    self._rp = SparseRandomProjection(n_components=min(128, all_emb.shape[1]), random_state=42)
    projected = self._rp.fit_transform(all_emb)
    selected = self._greedy_coreset(projected, k)
    self.memory_bank = torch.from_numpy(all_emb[selected]).float().to(self.device)

  def _greedy_coreset(self, features: np.ndarray, k: int) -> np.ndarray:
    n = len(features)
    k = min(k, n)
    idx = [np.random.randint(0, n)]
    min_dist = np.linalg.norm(features - features[idx[0]], axis=1)
    for _ in range(k - 1):
      next_idx = int(np.argmax(min_dist))
      idx.append(next_idx)
      min_dist = np.minimum(
          min_dist, np.linalg.norm(features - features[next_idx], axis=1)
      )
    return np.array(idx)

  @torch.no_grad()
  def predict(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
    """Returns image-level score and pixel-level map."""
    x = x.to(self.device)
    if x.dim() == 3:
      x = x.unsqueeze(0)
    feat = self.extractor.embed(x)
    b, c, h, w = feat.shape
    patches = _patchify(feat)  # N, C
    # k-NN distance to memory bank
    dists = torch.cdist(patches, self.memory_bank, p=2)
    knn, _ = torch.topk(dists, k=min(self.num_neighbors, len(self.memory_bank)), largest=False, dim=1)
    patch_scores = knn.mean(dim=1).cpu().numpy()
    score_map = patch_scores.reshape(h, w)
    score_map = F.interpolate(
        torch.from_numpy(score_map).view(1, 1, h, w).float(),
        size=(x.shape[2], x.shape[3]),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    image_score = float(patch_scores.max())
    return image_score, score_map

  def save(self, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "memory_bank": self.memory_bank.cpu(),
        "coreset_ratio": self.coreset_ratio,
        "num_neighbors": self.num_neighbors,
        "rp": self._rp,
    }
    torch.save(state, path / "patchcore.pt")
    meta = {"model": "patchcore"}
    (path / "meta.pkl").write_bytes(pickle.dumps(meta))

  def load(self, path: str | Path) -> None:
    path = Path(path)
    state = torch.load(path / "patchcore.pt", map_location=self.device, weights_only=False)
    self.memory_bank = state["memory_bank"].to(self.device)
    self.coreset_ratio = state["coreset_ratio"]
    self.num_neighbors = state["num_neighbors"]
    self._rp = state.get("rp")
