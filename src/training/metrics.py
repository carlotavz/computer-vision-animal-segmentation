from __future__ import annotations
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
# ----- Segmentación -----
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
def iou_score(logits: torch.Tensor, y_true: torch.Tensor, thresh: float = 0.5):
    with torch.no_grad():
        probs = sigmoid(logits)
        y_pred = (probs > thresh).float()
        inter = (y_pred * y_true).sum(dim=(1,2,3))
        union = (y_pred + y_true - y_pred * y_true).sum(dim=(1,2,3))
        iou = (inter + 1e-6) / (union + 1e-6)
        return iou.mean().item()
def dice_score(logits: torch.Tensor, y_true: torch.Tensor, thresh: float = 0.5):
    with torch.no_grad():
        probs = sigmoid(logits)
        y_pred = (probs > thresh).float()
        inter = (y_pred * y_true).sum(dim=(1,2,3))
        denom = y_pred.sum(dim=(1,2,3)) + y_true.sum(dim=(1,2,3))
        dice = (2*inter + 1e-6) / (denom + 1e-6)
        return dice.mean().item()
# ----- Clasificación -----
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }