"""Prediction and post-processing for the segmentation model."""
import cv2, numpy as np
from .model import build_unet
from ..roi.quality import clean_mask

def predict_mask(model, image_path, input_size=(256, 256), threshold=.5, postprocess=None):
    if isinstance(image_path, np.ndarray):
        decoded = image_path
    else:
        decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if decoded is None: raise ValueError(f"Unreadable image: {image_path}")
    image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]; resized = cv2.resize(image, input_size).astype("float32") / 255.0

    # Support both TFLite Interpreter and Keras Model
    if hasattr(model, "invoke"):
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]["index"], np.expand_dims(resized, axis=0))
        model.invoke()
        probability = model.get_tensor(output_details[0]["index"])[0, ..., 0]
    else:
        probability = model.predict(resized[None], verbose=0)[0, ..., 0]

    probability = cv2.resize(probability, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    postprocess = postprocess or {}; binary, quality = clean_mask(probability, threshold, postprocess.get("min_object_pixels", 64), postprocess.get("keep_components", 3), postprocess.get("fill_holes", False))
    return image, probability, binary, quality

