"""Evaluation plots and segmentation overlays."""
from pathlib import Path
import cv2, matplotlib.pyplot as plt, numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

def segmentation_overlay(image, ground_truth, predicted, output):
    overlay = image.copy(); overlay[ground_truth.astype(bool)] = (.55 * overlay[ground_truth.astype(bool)] + .45 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[predicted.astype(bool)] = (.55 * overlay[predicted.astype(bool)] + .45 * np.array([255, 0, 0])).astype(np.uint8)
    Path(output).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(output), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def classification_plots(y_true, probability, threshold, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_true, probability >= threshold); plt.savefig(out / "confusion_matrix.png"); plt.close()
    if len(np.unique(y_true)) == 2:
        RocCurveDisplay.from_predictions(y_true, probability); plt.savefig(out / "roc_curve.png"); plt.close()
        PrecisionRecallDisplay.from_predictions(y_true, probability); plt.savefig(out / "pr_curve.png"); plt.close()
    observed, predicted = calibration_curve(y_true, probability, n_bins=10, strategy="quantile")
    plt.plot(predicted, observed, marker="o"); plt.plot([0, 1], [0, 1], "--"); plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency"); plt.savefig(out / "calibration_curve.png"); plt.close()
