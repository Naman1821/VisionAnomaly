import numpy as np

from visionanomaly.metrics.auroc import pixel_auroc, safe_auroc


def test_safe_auroc_perfect():
  y = np.array([0, 0, 1, 1])
  s = np.array([0.1, 0.2, 0.8, 0.9])
  assert safe_auroc(y, s) == 1.0


def test_pixel_auroc():
  masks = np.array([[[0, 1], [0, 0]], [[1, 1], [0, 0]]])
  scores = np.array([[[0.1, 0.9], [0.2, 0.3]], [[0.8, 0.7], [0.1, 0.2]]])
  auc = pixel_auroc(masks, scores)
  assert auc >= 0.5
