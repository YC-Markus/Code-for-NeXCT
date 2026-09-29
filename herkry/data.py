"""Portable, patient-split NumPy input manifest; no hidden local data paths."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch.nn.functional as F

_augment = v2.Compose([
    v2.RandomHorizontalFlip(.5), v2.RandomVerticalFlip(.5),
    v2.RandomRotation(180, interpolation=v2.InterpolationMode.BILINEAR),
])


class ManifestDataset(Dataset):
    """Each row: patient, path, optional slice. Arrays are normalized CT or log sino."""
    def __init__(self, manifest, split, kind="image"):
        manifest = Path(manifest).resolve()
        self.root = manifest.parent
        self.manifest = json.loads(manifest.read_text())
        self.rows = self.manifest[split]
        self.kind = kind
        patients = {}
        for name in ("train", "val", "test"):
            for row in self.manifest.get(name, []):
                pid = row["patient"]
                if pid in patients and patients[pid] != name:
                    raise ValueError(f"Patient leakage across splits: {pid}")
                patients[pid] = name
        if not self.rows:
            raise ValueError(f"Empty split: {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["path"]).resolve()
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if "slice" in row:
            array = array[int(row["slice"])]
        array = np.array(array, dtype=np.float32, copy=True)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"Expected finite 2D slice: {path}")
        if self.kind == "image" and (array.min() < 0 or array.max() > 1):
            raise ValueError("CT arrays must be HU-windowed and normalized to [0,1].")
        return {self.kind: torch.from_numpy(array)[None], "path": str(path),
                "patient": row["patient"], "index": index}


@torch.no_grad()
def prepare_image(batch, device, size, training=False):
    image = batch["image"].to(device, non_blocking=True)
    if training:
        image = _augment(image)
    return F.adaptive_avg_pool2d(image, (size, size))
