from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from src.preprocessing.utils import seed_everything
from src.preprocessing.transformations import SegmentationTransform, ClassificationTransform
from src.datasets.animal_dataset import AnimalSegmentationDataset, AnimalClassificationDataset
from src.models.unet import UNet
from src.models.cnn_classifier import SimpleCNNClassifier
from src.training.trainer import train_unet, train_classifier
def make_loaders_seg(data_root: str, img_size=256, batch_size=8):
    ds = AnimalSegmentationDataset(data_root, transform=SegmentationTransform(train=True, img_size=(img_size, img_size)))
    n = len(ds)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    # En validación necesitamos transform de val (sin augmentations)
    val_tf = SegmentationTransform(train=False, img_size=(img_size, img_size))
    val_ds.dataset.transform = val_tf  # type: ignore
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    }
def make_loaders_cls(data_root: str, img_size=224, batch_size=16, mode="rgb"):
    train_tf = ClassificationTransform(train=True, img_size=(img_size, img_size), mode=mode)
    val_tf = ClassificationTransform(train=False, img_size=(img_size, img_size), mode=mode)
    ds = AnimalClassificationDataset(data_root, transform=train_tf)
    n = len(ds)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    # set val transform
    val_ds.dataset.transform = val_tf  # type: ignore
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    }
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data", help="Ruta a carpeta data con images/ y masks/")
    parser.add_argument("--task", type=str, choices=["seg", "cls"], default="seg")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--cls_mode", type=str, choices=["rgb", "mask4", "roi"], default="roi")
    args = parser.parse_args()
    seed_everything(42)
    Path("results").mkdir(exist_ok=True, parents=True)
    if args.task == "seg":
        loaders = make_loaders_seg(args.data_root, img_size=args.img_size, batch_size=args.batch_size)
        model = UNet(in_ch=3, out_ch=1)
        train_unet(model, loaders, epochs=args.epochs)
    else:
        loaders = make_loaders_cls(args.data_root, img_size=224, batch_size=args.batch_size, mode=args.cls_mode)
        in_ch = 3 if args.cls_mode in {"rgb", "roi"} else 4
        model = SimpleCNNClassifier(in_ch=in_ch, num_classes=4)
        train_classifier(model, loaders, epochs=args.epochs)
if __name__ == "__main__":
    main()