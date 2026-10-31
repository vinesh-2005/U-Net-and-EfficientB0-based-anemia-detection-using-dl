"""ROI image dataset for classifier experiments."""

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


def load_images_and_labels(frame: pd.DataFrame, image_size=(224, 224)):
    images, labels = [], []
    for _, row in frame.iterrows():
        img = cv2.imread(str(row["roi_path"]))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        # Keep pixel values in [0.0, 255.0] range for EfficientNetB0 internal rescaling
        images.append(img.astype(np.float32))
        labels.append(float(row["label_binary"]))
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)


def make_dataset(
    frame: pd.DataFrame,
    image_size=(224, 224),
    batch_size=8,
    training=False,
    seed=2025,
    steps=None,
):
    frame = frame.reset_index(drop=True)
    images, labels = load_images_and_labels(frame, image_size)

    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)

    return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

