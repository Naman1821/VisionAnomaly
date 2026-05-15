from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
  y_true = np.asarray(y_true).astype(int)
  y_score = np.asarray(y_score).astype(float)
  if len(np.unique(y_true)) < 2:
    return float("nan")
  return float(roc_auc_score(y_true, y_score))


def pixel_auroc(masks: np.ndarray, scores: np.ndarray) -> float:
  """masks, scores: [N, H, W]"""
  return safe_auroc(masks.ravel(), scores.ravel())
