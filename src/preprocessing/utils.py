from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
# ----------------------------
# Reproducibilidad
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ----------------------------
# Carga de imágenes y máscaras
# ----------------------------
def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img
def load_mask(path: str | Path) -> Image.Image:
    # Devuelve máscara binaria en modo 'L' (0..255)
    m = Image.open(path).convert("L")
    return m
# ----------------------------
# Emparejar imágenes y máscaras
# ----------------------------
def _stem_key(name: str) -> str:
    """
    Intenta construir una clave común para emparejar image_0001.jpg ↔ mask_0001.png
    Elimina prefijos comunes (image_, img_, mask_) y extensiones.
    """
    base = Path(name).stem.lower()
    for pref in ["image_", "img_", "mask_"]:
        if base.startswith(pref):
            base = base[len(pref):]
    return base
def pair_images_and_masks(images_dir: str | Path, masks_dir: str | Path) -> List[Tuple[Path, Path]]:
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    assert images_dir.exists() and masks_dir.exists(), "Carpetas de images/masks no existen"
    img_map: Dict[str, Path] = {}
    for p in images_dir.glob("*.*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            img_map[_stem_key(p.name)] = p
    pairs: List[Tuple[Path, Path]] = []
    for m in masks_dir.glob("*.*"):
        if m.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            key = _stem_key(m.name)
            if key in img_map:
                pairs.append((img_map[key], m))
    # Orden estable por nombre
    pairs.sort(key=lambda t: t[0].name)
    return pairs
def list_all_pairs_by_class(data_root: str | Path) -> List[Tuple[Path, Path, int]]:
    """
    Devuelve lista [(img_path, mask_path, label_int)] recorriendo
    data/images/<cls> y data/masks/<cls>.
    El label se codifica como entero en orden alfabético de carpetas.
    """
    data_root = Path(data_root)
    images_root = data_root / "images"
    masks_root = data_root / "masks"
    classes = sorted([d.name for d in images_root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    out: List[Tuple[Path, Path, int]] = []
    for cls in classes:
        cls_img = images_root / cls
        cls_mask = masks_root / cls
        pairs = pair_images_and_masks(cls_img, cls_mask)
        out.extend([(ip, mp, class_to_idx[cls]) for ip, mp in pairs])
    return out
# ----------------------------
# Utilidades ROI
# ----------------------------
def mask_to_bbox(mask: np.ndarray, padding: int = 8) -> Optional[Tuple[int, int, int, int]]:
    """mask: ndarray binaria (H,W) con {0,1} → devuelve (x1,y1,x2,y2) o None si vacía."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    # padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(mask.shape[1] - 1, x2 + padding)
    y2 = min(mask.shape[0] - 1, y2 + padding)
    return int(x1), int(y1), int(x2), int(y2)
def crop_by_bbox(img: Image.Image, mask: Image.Image, bbox: Tuple[int, int, int, int]) -> Tuple[Image.Image, Image.Image]:
    x1, y1, x2, y2 = bbox
    # PIL crop usa (left, upper, right, lower) con right/lower exclusives → +1
    patch_box = (x1, y1, x2 + 1, y2 + 1)
    return img.crop(patch_box), mask.crop(patch_box)