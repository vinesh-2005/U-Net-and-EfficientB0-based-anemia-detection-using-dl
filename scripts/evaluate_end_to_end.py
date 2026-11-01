#!/usr/bin/env python3
"""
Evaluate the trained classifier checkpoint on the held-out test set
and validation set, calculating verified metrics, confusion matrices, and ROC curves,
and updating artifacts/current/classifier/evaluation_dashboard_data.json.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
import tensorflow as tf


def run_evaluation(
    manifest_csv: str = "data/processed/roi_predicted_v2/roi_manifest.csv",
    classifier_path: str = "artifacts/current/classifier/best_classifier.keras",
    threshold_config_path: str = "artifacts/current/classifier/threshold_config.json",
    output_json_path: str = "artifacts/current/classifier/evaluation_dashboard_data.json",
):
    manifest_p = Path(manifest_csv)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Evaluation manifest not found at: {manifest_csv}")

    model_p = Path(classifier_path)
    if not model_p.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found at: {classifier_path}")

    print(f"Loading classifier from {classifier_path}...")
    raw_model = tf.keras.models.load_model(classifier_path, compile=False)
    from src.classification.model import build_inference_classifier
    model = build_inference_classifier(raw_model)

    threshold = 0.3837
    if Path(threshold_config_path).exists():
        with open(threshold_config_path, "r") as f:
            t_data = json.load(f)
            threshold = float(t_data.get("threshold", threshold))
    print(f"Using decision threshold: {threshold:.4f}")

    df = pd.read_csv(manifest_csv)
    print(f"Loaded dataset manifest with {len(df)} total samples.")

    dashboard_data = {
        "threshold": threshold,
        "model_info": {
            "architecture": "EfficientNetB0 (Fine-Tuned Transfer Learning) + U-Net",
            "segmentation_model": "U-Net Conjunctiva Segmentation (Input 256x256)",
            "classifier_model": "EfficientNetB0 (Input 224x224x3)",
            "dataset_name": "EYES-DEFY-ANEMIA Conjunctival Dataset (Italy & India Cohorts)",
            "total_samples": len(df),
            "split_counts": {
                "train": int((df["split"] == "train").sum()),
                "val": int((df["split"] == "val").sum()),
                "test": int((df["split"] == "test").sum()),
            },
            "checkpoint_path": classifier_path,
            "threshold_calibration": "Calibrated on Validation Set for optimal Sensitivity & F1",
        },
        "splits": {},
    }

    for split in ["test", "val"]:
        split_df = df[df["split"] == split].reset_index(drop=True)
        if len(split_df) == 0:
            print(f"Warning: No samples found for split '{split}'")
            continue

        imgs = []
        for p in split_df["roi_path"]:
            img = cv2.imread(p)
            if img is None:
                raise ValueError(f"Unreadable ROI image at: {p}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224)).astype(np.float32)
            imgs.append(img_resized)

        batch = np.array(imgs)
        preds = model.predict(batch, verbose=0).ravel()
        y_true = split_df["label_binary"].to_numpy().astype(int)
        y_pred = (preds >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        fpr, tpr, _ = roc_curve(y_true, preds)
        roc_auc = float(auc(fpr, tpr))
        roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]

        print(f"\n--- Results for '{split.upper()}' Set ({len(split_df)} samples) ---")
        print(f"Accuracy:    {acc * 100:.2f}%")
        print(f"Precision:   {prec * 100:.2f}%")
        print(f"Sensitivity: {rec * 100:.2f}% (Recall)")
        print(f"Specificity: {spec * 100:.2f}%")
        print(f"F1-Score:    {f1 * 100:.2f}%")
        print(f"ROC-AUC:     {roc_auc:.4f}")
        print(f"Confusion Matrix [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")

        dashboard_data["splits"][split] = {
            "total_samples": int(len(split_df)),
            "non_anemic_count": int((y_true == 0).sum()),
            "anemic_count": int((y_true == 1).sum()),
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall_sensitivity": rec,
                "specificity": spec,
                "f1_score": f1,
                "roc_auc": roc_auc,
            },
            "confusion_matrix": {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "matrix": cm.tolist(),
            },
            "roc_curve": {
                "roc_auc": roc_auc,
                "points": roc_points,
            },
        }

    output_p = Path(output_json_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    with open(output_p, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"\nSuccessfully wrote verified evaluation metrics to: {output_json_path}")
    return dashboard_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/roi_predicted_v2/roi_manifest.csv")
    parser.add_argument("--classifier", default="artifacts/current/classifier/best_classifier.keras")
    parser.add_argument("--threshold-config", default="artifacts/current/classifier/threshold_config.json")
    parser.add_argument("--output", default="artifacts/current/classifier/evaluation_dashboard_data.json")
    args = parser.parse_args()

    run_evaluation(
        manifest_csv=args.manifest,
        classifier_path=args.classifier,
        threshold_config_path=args.threshold_config,
        output_json_path=args.output,
    )
