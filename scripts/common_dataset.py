import os
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def _list_images(dir_path: str) -> List[str]:
    out = []
    for fn in sorted(os.listdir(dir_path)):
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMG_EXTS:
            out.append(fn)
    return out


@dataclass
class PairedPaths:
    heatmap_path: str
    roof_path: str
    name: str


def discover_pairs(data_root: str, heatmap_subdir: str = "heatmap", roof_subdir: str = "roof") -> List[PairedPaths]:
    hm_dir = os.path.join(data_root, heatmap_subdir)
    rf_dir = os.path.join(data_root, roof_subdir)
    if not os.path.isdir(hm_dir):
        raise FileNotFoundError(f"Missing heatmap dir: {hm_dir}")
    if not os.path.isdir(rf_dir):
        raise FileNotFoundError(f"Missing roof dir: {rf_dir}")

    hm_files = _list_images(hm_dir)
    rf_files = set(_list_images(rf_dir))

    pairs: List[PairedPaths] = []
    missing = []
    for fn in hm_files:
        if fn not in rf_files:
            missing.append(fn)
            continue
        pairs.append(PairedPaths(
            heatmap_path=os.path.join(hm_dir, fn),
            roof_path=os.path.join(rf_dir, fn),
            name=os.path.splitext(fn)[0],
        ))

    if missing:
        raise RuntimeError(f"Found {len(missing)} heatmaps without matching roof images. Example: {missing[:5]}")
    if not pairs:
        raise RuntimeError("No image pairs found.")
    return pairs


class PairedImageDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        image_size: int,
        heatmap_subdir: str = "heatmap",
        roof_subdir: str = "roof",
        random_flip: bool = True,
        normalize_to_minus1_1: bool = True,
    ):
        self.pairs = discover_pairs(data_root, heatmap_subdir, roof_subdir)
        self.image_size = image_size

        tfms = [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
        ]
        if random_flip:
            tfms.append(T.RandomHorizontalFlip(p=0.5))
        tfms.append(T.ToTensor())

        if normalize_to_minus1_1:
            tfms.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        self.transform = T.Compose(tfms)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        hm = Image.open(p.heatmap_path).convert("RGB")
        rf = Image.open(p.roof_path).convert("RGB")
        hm_t = self.transform(hm)
        rf_t = self.transform(rf)
        return {
            "conditioning": hm_t,
            "target": rf_t,
            "name": p.name,
        }
