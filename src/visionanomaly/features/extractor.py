from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Wide_ResNet50_2_Weights


class FeatureExtractor(nn.Module):
  """Multi-scale patch features from frozen Wide ResNet-50-2 (torchvision)."""

  def __init__(
      self,
      backbone: str = "wide_resnet50_2",
      layers: list[str] | None = None,
  ) -> None:
    super().__init__()
    if backbone != "wide_resnet50_2":
      raise ValueError(f"Only wide_resnet50_2 supported; got {backbone}")
    layers = layers or ["layer2", "layer3"]
    self.layer_names = layers
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    self.body = models.wide_resnet50_2(weights=weights)
    self.body.fc = nn.Identity()
    self._cache: dict[str, torch.Tensor] = {}

    def _hook(name: str):
      def fn(_module, _inp, out):
        self._cache[name] = out

      return fn

    for name in layers:
      layer = getattr(self.body, name)
      layer.register_forward_hook(_hook(name))
    for p in self.body.parameters():
      p.requires_grad = False
    self.eval()

  @torch.no_grad()
  def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    self._cache.clear()
    self.body(x)
    return [self._cache[n] for n in self.layer_names]

  def embed(self, x: torch.Tensor) -> torch.Tensor:
    feats = self.forward(x)
    h, w = feats[-1].shape[-2:]
    resized = [
        torch.nn.functional.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
        for f in feats
    ]
    return torch.cat(resized, dim=1)
