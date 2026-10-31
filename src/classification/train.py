"""Two-stage EfficientNetB0 training with threshold calibration."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from .model import build_classifier
from .dataset import make_dataset
from ..common.metrics import classification_metrics, validation_threshold


def train(config, roi_csv, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(roi_csv)
    size = tuple(config["input_size"])
    batch_size = config.get("batch_size", 8)

    model = build_classifier(
        size,
        config.get("dropout", 0.3),
        config.get("head_units", 128),
        config.get("weights", "imagenet"),
        0,
    )

    train_frame = frame[frame.split == "train"].reset_index(drop=True)
    val_frame = frame[frame.split == "val"].reset_index(drop=True)
    test_frame = frame[frame.split == "test"].reset_index(drop=True)

    y_train = train_frame["label_binary"].to_numpy().astype(int)
    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()
    total = len(y_train)
    class_weight = {0: float(total / (2.0 * n0)), 1: float(total / (2.0 * n1))}

    print(f"Train samples: {len(train_frame)} (normal: {n0}, anemic: {n1})")
    print(f"Validation samples: {len(val_frame)}")
    print(f"Test samples: {len(test_frame)}")
    print(f"Class weights: {class_weight}")
    print(f"Batch size: {batch_size}")

    callbacks_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(
            out / "best_classifier_stage1.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.get("patience", 8),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    # ---------------- STAGE 1: Train Head Only ----------------
    print("\nStarting Stage 1 training (Frozen Backbone)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.get("stage1_learning_rate", 0.001)),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    train_ds = make_dataset(train_frame, size, batch_size, training=True)
    val_ds = make_dataset(val_frame, size, batch_size, training=False)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.get("stage1_epochs", 20),
        class_weight=class_weight,
        callbacks=callbacks_stage1,
        verbose=1,
    )

    # ---------------- STAGE 2: Fine-Tuning ----------------
    print("\nStarting Stage 2 fine-tuning...")
    backbone = next(l for l in model.layers if "efficientnet" in l.name.lower())
    fine_tune_layers = config.get("fine_tune_layers", 40)
    for layer in backbone.layers[-fine_tune_layers:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.get("stage2_learning_rate", 0.00002)),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    callbacks_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(
            out / "best_classifier.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.get("patience", 8),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        ),
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.get("stage2_epochs", 30),
        class_weight=class_weight,
        callbacks=callbacks_stage2,
        verbose=1,
    )

    # ---------------- Evaluate & Calibrate Threshold ----------------
    best_model = tf.keras.models.load_model(out / "best_classifier.keras", compile=False)

    val_preds = best_model.predict(val_ds, verbose=0).ravel()
    val_true = val_frame["label_binary"].to_numpy().astype(int)
    threshold = validation_threshold(val_true, val_preds)

    val_metrics = classification_metrics(val_true, val_preds, threshold)
    print(f"\nOptimal Validation Threshold: {threshold:.4f}")
    print(f"Validation Metrics: {val_metrics}")

    if len(test_frame) > 0:
        test_ds = make_dataset(test_frame, size, batch_size, training=False)
        test_preds = best_model.predict(test_ds, verbose=0).ravel()
        test_true = test_frame["label_binary"].to_numpy().astype(int)
        test_metrics = classification_metrics(test_true, test_preds, threshold)
        print(f"Held-Out Test Metrics at threshold {threshold:.4f}: {test_metrics}")

        thresh_info = {
            "threshold": float(threshold),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
    else:
        thresh_info = {
            "threshold": float(threshold),
            "validation_metrics": val_metrics,
        }

    with open(out / "threshold_config.json", "w") as f:
        json.dump(thresh_info, f, indent=2)

    return history1, history2

