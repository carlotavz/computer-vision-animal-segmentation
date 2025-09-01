from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from .utils import mask_to_bbox, crop_by_bbox
# Normalización Imagenet por defecto (opcional)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
class SegmentationTransform:
    def __init__(self, train: bool = True, img_size: Tuple[int, int] = (256, 256)):
        aug = []
        if train:
            aug += [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.05),
            ]
        aug += [T.Resize(img_size)]
        self.img_tf = T.Compose([
            *aug,
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_tf = T.Compose([
            *aug,
            T.ToTensor(),  # quedará [0,1]
        ])
    def __call__(self, img: Image.Image, mask: Image.Image):
        return self.img_tf(img), self.mask_tf(mask)
class ClassificationTransform:
    """
    mode:
      - 'rgb': imagen completa RGB
      - 'mask4': añade la máscara como 4º canal
      - 'roi': recorta la ROI usando la máscara
    """
    def __init__(self, train: bool = True, img_size=(224, 224), mode: str = "rgb"):
        self.mode = mode
        aug = []
        if train:
            aug += [
                T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
            ]
        else:
            aug += [T.Resize(img_size), T.CenterCrop(img_size)]
        self.common = T.Compose([
            *aug,
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_resize = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),  # [0,1]
        ])
    def __call__(self, img: Image.Image, mask: Image.Image):
        if self.mode == "roi":
            m_np = np.array(mask, dtype=np.uint8)
            m_bin = (m_np > 127).astype(np.uint8)
            bbox = mask_to_bbox(m_bin)
            if bbox is not None:
                img, mask = crop_by_bbox(img, mask, bbox)
        x = self.common(img)
        if self.mode == "mask4":
            m = self.mask_resize(mask)
            x = torch.cat([x, m], dim=0)  # C pasa a 4
        return x