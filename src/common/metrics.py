"""Metrics shared by segmentation and classification evaluation."""
from typing import Any
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

def segmentation_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = .5) -> dict[str, float]:
    true = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_prob) >= threshold
    intersection = np.logical_and(true, pred).sum()
    union = np.logical_or(true, pred).sum()
    return {"dice": float((2 * intersection + 1e-7) / (true.sum() + pred.sum() + 1e-7)),
            "iou": float((intersection + 1e-7) / (union + 1e-7)),
            "pixel_precision": float((intersection + 1e-7) / (pred.sum() + 1e-7)),
            "pixel_recall": float((intersection + 1e-7) / (true.sum() + 1e-7))}

def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true, probability = np.asarray(y_true).astype(int), np.asarray(probability, dtype=float)
    pred = (probability >= threshold).astype(int)
    result: dict[str, Any] = {"threshold": float(threshold), "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)), "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)), "brier_score": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist()}
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    result["specificity"] = float(tn / (tn + fp)) if tn + fp else None
    result["roc_auc"] = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None
    result["pr_auc"] = float(average_precision_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None
    return result

def validation_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    thresholds = np.unique(np.clip(probability, 0, 1))
    scores = [(f1_score(y_true, probability >= t, zero_division=0), t) for t in thresholds]
    return float(max(scores, key=lambda item: (item[0], -item[1]))[1]) if scores else .5
