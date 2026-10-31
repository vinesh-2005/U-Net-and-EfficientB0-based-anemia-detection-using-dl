"""Deterministic mask cleanup and quality measurements."""
import cv2, numpy as np

def clean_mask(probability: np.ndarray, threshold=.5, min_object_pixels=64, keep_components=3, fill_holes=False):
    binary = (probability >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    areas = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, count)], reverse=True)[:keep_components]
    out = np.zeros_like(binary)
    for area, index in areas:
        if area >= min_object_pixels: out[labels == index] = 1
    if fill_holes:
        flood = (out * 255).astype(np.uint8); flood_mask = np.zeros((out.shape[0] + 2, out.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255); out = np.maximum(out, (flood == 0).astype(np.uint8))
    n = cv2.connectedComponents(out, 8)[0] - 1
    return out, mask_quality(probability, out, n)

def mask_quality(probability, binary, components=None):
    h, w = binary.shape[:2]; ys, xs = np.where(binary > 0)
    return {"foreground_area_ratio": float(binary.mean()), "connected_components": int(components if components is not None else cv2.connectedComponents(binary, 8)[0] - 1),
            "border_contact": bool(binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any()),
            "mean_mask_probability": float(probability[binary > 0].mean()) if binary.any() else 0.0,
            "quality_status": "accepted" if binary.any() else "flagged_empty_mask"}
