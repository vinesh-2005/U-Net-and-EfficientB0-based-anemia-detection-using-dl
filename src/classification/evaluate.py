"""Evaluate image- or patient-level classifier probabilities."""
import numpy as np, pandas as pd
from ..common.metrics import classification_metrics, validation_threshold

def evaluate(model, frame: pd.DataFrame, image_size=(224, 224), threshold=None, batch_size=32):
    from .dataset import make_dataset
    probability = model.predict(make_dataset(frame, image_size, batch_size), verbose=0).ravel()
    threshold = validation_threshold(frame.label_binary.to_numpy(), probability) if threshold is None else threshold
    return classification_metrics(frame.label_binary.to_numpy(), probability, threshold), probability
