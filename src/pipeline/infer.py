"""Single reusable full-image -> U-Net -> ROI -> EfficientNet inference contract."""
import numpy as np
from ..roi.extract import extract_roi
from ..segmentation.predict import predict_mask

def infer(image_path, segmentation_model, classifier_model, config, versions=None):
    image, probability, mask, quality = predict_mask(segmentation_model, image_path, tuple(config["segmentation"]["input_size"]), config["segmentation"]["mask_threshold"], config["segmentation"].get("postprocess"))
    flags = [] if quality["quality_status"] == "accepted" else [quality["quality_status"]]
    if flags: return {"anemia_probability": None, "predicted_label": None, "classifier_threshold": config["classifier_threshold"], "segmentation_quality": "flagged", "quality_flags": flags, "segmentation_quality_metrics": quality}
    roi, bbox = extract_roi(image, mask, tuple(config["classifier"]["input_size"]), config["classifier"]["roi_margin_fraction"], config["classifier"].get("mask_outside_roi", False))
    probability_out = float(classifier_model.predict(roi[None].astype("float32"), verbose=0)[0, 0]); threshold = config["classifier_threshold"]
    return {"anemia_probability": probability_out, "predicted_label": "anemic" if probability_out >= threshold else "normal", "classifier_threshold": threshold, "segmentation_quality": "accepted", "quality_flags": flags, "segmentation_quality_metrics": quality, "bbox": bbox, **(versions or {})}

