from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


@dataclass
class MVTecSample:
    image: torch.Tensor
    label: int  # 0 normal, 1 anomaly
    mask: torch.Tensor | None
    path: str
    split: str


class MVTecDataset(Dataset):
  """MVTec-AD layout: root/category/{train,test}/{good,defect}/"""

  def __init__(
      self,
      root: str | Path,
      category: str,
      split: str,
      image_size: int = 256,
      augment: bool = False,
  ) -> None:
    self.root = Path(root) / category
    self.category = category
    self.split = split
    self.samples: list[tuple[Path, int, Path | None]] = []

    if split == "train":
      good_dir = self.root / "train" / "good"
      if not good_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {good_dir}. Run: python scripts/download_mvtec.py"
        )
      for p in sorted(good_dir.glob("*")):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
          self.samples.append((p, 0, None))
    elif split == "test":
      test_root = self.root / "test"
      if not test_root.is_dir():
        raise FileNotFoundError(f"Missing {test_root}")
      for defect_dir in sorted(test_root.iterdir()):
        if not defect_dir.is_dir():
          continue
        label = 0 if defect_dir.name == "good" else 1
        for p in sorted(defect_dir.glob("*")):
          if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
          mask_path = None
          if label == 1:
            mask_candidate = (
                self.root / "ground_truth" / defect_dir.name / f"{p.stem}_mask.png"
            )
            if mask_candidate.exists():
              mask_path = mask_candidate
          self.samples.append((p, label, mask_path))
    else:
      raise ValueError(f"Unknown split: {split}")

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augment and split == "train":
      self.transform = T.Compose(
          [
              T.Resize((image_size, image_size)),
              T.RandomHorizontalFlip(),
              T.ToTensor(),
              normalize,
          ]
      )
    else:
      self.transform = T.Compose(
          [
              T.Resize((image_size, image_size)),
              T.ToTensor(),
              normalize,
          ]
      )
    self.mask_transform = T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> MVTecSample:
    path, label, mask_path = self.samples[idx]
    img = Image.open(path).convert("RGB")
    image = self.transform(img)
    mask = None
    if mask_path is not None:
      m = Image.open(mask_path).convert("L")
      mask = self.mask_transform(m)
      mask = (mask > 0.5).float()
    return MVTecSample(
        image=image,
        label=label,
        mask=mask,
        path=str(path),
        split=self.split,
    )


def collate_mvtec(batch: list[MVTecSample]) -> dict:
  images = torch.stack([b.image for b in batch])
  labels = torch.tensor([b.label for b in batch], dtype=torch.long)
  masks = [b.mask for b in batch]
  if all(m is None for m in masks):
    mask_tensor = None
  else:
    mask_tensor = torch.zeros(len(batch), 1, images.shape[2], images.shape[3])
    for i, m in enumerate(masks):
      if m is not None:
        mask_tensor[i] = m
  return {
      "image": images,
      "label": labels,
      "mask": mask_tensor,
      "path": [b.path for b in batch],
  }


def build_dataloader(
    root: str | Path,
    category: str,
    split: str,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    augment: bool = False,
    max_samples: int | None = None,
) -> DataLoader:
  ds = MVTecDataset(root, category, split, image_size, augment=augment)
  if max_samples is not None:
    from torch.utils.data import Subset

    ds = Subset(ds, list(range(min(max_samples, len(ds)))))
  return DataLoader(
      ds,
      batch_size=batch_size,
      shuffle=(split == "train"),
      num_workers=num_workers,
      collate_fn=collate_mvtec,
      pin_memory=False,
  )
