from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
# ---------------------- Utilidades de métricas ----------------------
def _cls_metrics(y_true: List[int], y_pred: List[int]):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_per_class": prec.tolist(),
        "recall_per_class": rec.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support": support.tolist(),
    }
def _plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str,
):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    # Anotaciones
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
# ------------------------- Bucle UNet (seg) -------------------------
def train_unet(
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    out_dir: str = "results",
    amp: bool = True,
    early_stopping_patience: int = 10,
):
    device = torch.device(device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_score = -1.0
    best_path = Path(out_dir) / "best_unet.pt"
    metrics_path = Path(out_dir) / "segmentation_metrics.json"
    vis_dir = Path(out_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    epochs_no_improve = 0
    def _eval_epoch():
        model.eval()
        iou_scores, dice_scores, losses = [], [], []
        with torch.no_grad():
            for x, y, _ in loaders["val"]:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y).item()
                losses.append(loss)
                # métricas
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                inter = (preds * y).sum(dim=(1, 2, 3))
                union = (preds + y - preds * y).sum(dim=(1, 2, 3))
                iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
                denom = preds.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
                dice = ((2 * inter + 1e-6) / (denom + 1e-6)).mean().item()
                iou_scores.append(iou)
                dice_scores.append(dice)
        return float(np.mean(losses)), float(np.mean(iou_scores)), float(np.mean(dice_scores))
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"[UNet][Train] Ep {ep}/{epochs}")
        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        val_loss, miou, mdice = _eval_epoch()
        print(f"[UNet][Val] loss={val_loss:.4f} | mIoU={miou:.4f} | Dice={mdice:.4f}")
        score = (miou + mdice) / 2
        sched.step(score)
        # checkpointing + early stopping
        if score > best_score:
            best_score = score
            epochs_no_improve = 0
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path.as_posix())
            print(f"[UNet] ✔ Guardado {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("[UNet] Early stopping por falta de mejora")
                break
    # guarda resumen final
    with open(metrics_path, "w") as f:
        json.dump({"best_val_score": best_score}, f, indent=2)
# -------------------- Bucle Clasificación (cls) --------------------
def train_classifier(
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    out_dir: str = "results",
    class_names: Optional[List[str]] = None,
    amp: bool = True,
    early_stopping_patience: int = 10,
):
    device = torch.device(device)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if class_names is None:
        # Por defecto las 4 clases del proyecto
        class_names = ["elephant", "rhino", "flamingo", "emu"]
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3
    )
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_f1 = -1.0
    best_path = Path(out_dir) / "best_classifier.pt"
    metrics_path = Path(out_dir) / "classification_metrics.json"
    epochs_no_improve = 0
    def _eval_epoch(split: str = "val") -> Tuple[float, Dict]:
        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        losses: List[float] = []
        with torch.no_grad():
            for x, y in loaders[split]:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y).item()
                losses.append(loss)
                pred = logits.argmax(dim=1).cpu().numpy().tolist()
                y_pred.extend(pred)
                y_true.extend(y.cpu().numpy().tolist())
        m = _cls_metrics(y_true, y_pred)
        return float(np.mean(losses)), m, y_true, y_pred
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"[CLS][Train] Ep {ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        val_loss, val_metrics, y_true, y_pred = _eval_epoch("val")
        print(
            f"[CLS][Val] loss={val_loss:.4f} | acc={val_metrics['accuracy']:.4f} | f1_macro={val_metrics['f1_macro']:.4f}"
        )
        sched.step(val_metrics["f1_macro"])  # optimizamos f1_macro
        # checkpointing + early stopping
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path.as_posix())
            print(f"[CLS] ✔ Guardado {best_path}")
            # Confusion matrix al mejor checkpoint
            _plot_confusion_matrix(
                y_true, y_pred, class_names, save_path=str(Path(out_dir) / "confusion_matrix.png")
            )
            with open(metrics_path, "w") as f:
                json.dump({"val": val_metrics, "best_f1_macro": best_f1}, f, indent=2)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("[CLS] Early stopping por falta de mejora")
                break
    # evaluación final opcional en split 'train' o 'val' ya guardada