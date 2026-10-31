"""Extract audit-friendly classifier ROIs from a binary conjunctiva mask."""
import cv2, numpy as np

def extract_roi(image: np.ndarray, mask: np.ndarray, output_size=(224, 224), margin_fraction=.15, mask_outside=False):
    ys, xs = np.where(mask > 0)
    if not len(xs): raise ValueError("Cannot extract ROI from an empty conjunctiva mask")
    h, w = mask.shape[:2]; mx, my = int((xs.max() - xs.min() + 1) * margin_fraction), int((ys.max() - ys.min() + 1) * margin_fraction)
    x0, x1, y0, y1 = max(0, xs.min() - mx), min(w, xs.max() + 1 + mx), max(0, ys.min() - my), min(h, ys.max() + 1 + my)
    crop = image[y0:y1, x0:x1].copy()
    if mask_outside: crop[mask[y0:y1, x0:x1] == 0] = 0
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA), {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}
