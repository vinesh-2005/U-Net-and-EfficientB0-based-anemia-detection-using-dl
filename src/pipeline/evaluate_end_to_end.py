"""Evaluate frozen end-to-end predictions; threshold must already be selected on validation."""
import pandas as pd
from ..common.metrics import classification_metrics

def patient_level_metrics(predictions: pd.DataFrame, threshold: float):
    grouped = predictions.groupby("patient_id", as_index=False).agg(label_binary=("label_binary", "first"), probability=("probability", "mean"))
    return classification_metrics(grouped.label_binary.to_numpy(), grouped.probability.to_numpy(), threshold), grouped
