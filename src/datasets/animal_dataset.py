from __future__ import annotations
from typing import Callable, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

from ..preprocessing.utils import list_all_pairs_by_class, load_image, load_mask

class AnimalSegmentationDataset(Dataset):
    def __init__(self, data_root: str | Path, transform: Optional[Callable] = None):
        self.data_root = Path(data_root)
        self.items = list_all_pairs_by_class(self.data_root)
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        img_p, mask_p, label = self.items[idx]
        img = load_image(img_p)
        mask = load_mask(mask_p)
        if self.transform is not None:
            x, y = self.transform(img, mask)
        else:
            x, y = img, mask
        if isinstance(y, Image.Image):
            y = torch.from_numpy((np.array(y) > 127).astype("float32")).unsqueeze(0)
        return x, y, label

class AnimalClassificationDataset(Dataset):
    def __init__(self, data_root: str | Path, transform: Optional[Callable] = None):
        self.data_root = Path(data_root)
        self.items = list_all_pairs_by_class(self.data_root)
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        img_p, mask_p, label = self.items[idx]
        img = load_image(img_p)
        mask = load_mask(mask_p)
        x = self.transform(img, mask) if self.transform is not None else img
        if isinstance(x, Image.Image):
            x = T.ToTensor()(x)
        return x, label
