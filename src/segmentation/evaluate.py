"""Evaluate pixel metrics on a patient-disjoint split."""
import numpy as np, pandas as pd
from .predict import predict_mask
from ..common.metrics import segmentation_metrics

def evaluate(model, frame: pd.DataFrame, input_size=(256, 256), threshold=.5, postprocess=None):
    values = []
    for row in frame.itertuples():
        _, prob, _, _ = predict_mask(model, row.image_path, input_size, threshold, postprocess)
        true = __import__("cv2").imread(row.mask_path, 0)
        true = __import__("cv2").resize(true, (prob.shape[1], prob.shape[0]), interpolation=__import__("cv2").INTER_NEAREST) > 0
        values.append(segmentation_metrics(true, prob, threshold))
    return {key: float(np.mean([item[key] for item in values])) for key in values[0]} if values else {}
