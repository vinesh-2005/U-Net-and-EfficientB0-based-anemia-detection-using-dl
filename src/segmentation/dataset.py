"""Manifest-backed TensorFlow segmentation dataset."""
import cv2, numpy as np, pandas as pd, tensorflow as tf
from . import model as _model

def make_dataset(frame: pd.DataFrame, image_size=(256, 256), batch_size=16, training=False, seed=2025):
    def generator():
        for row in frame.itertuples():
            image = cv2.cvtColor(cv2.imread(row.image_path), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, image_size); mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
            image, mask = image.astype("float32") / 255, (mask > 0).astype("float32")[..., None]
            if training and np.random.random() < .5: image, mask = image[:, ::-1], mask[:, ::-1]
            yield image, mask
    ds = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec((*image_size, 3), tf.float32), tf.TensorSpec((*image_size, 1), tf.float32)))
    return ds.shuffle(min(max(len(frame), 1), 16), seed=seed) .batch(batch_size).prefetch(tf.data.AUTOTUNE) if training else ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
