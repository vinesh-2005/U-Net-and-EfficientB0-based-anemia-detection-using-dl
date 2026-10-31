"""Verify image/mask decodability and mask semantics before modeling."""
from pathlib import Path
import cv2
import numpy as np

def inspect_annotation(image_path: str | Path, annotation_path: str | Path) -> dict[str, object]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    annotation = cv2.imread(str(annotation_path), cv2.IMREAD_UNCHANGED)
    if image is None: return {"valid": False, "reason": "corrupted_image"}
    if annotation is None: return {"valid": False, "reason": "corrupted_annotation"}
    if annotation.shape[:2] != image.shape[:2]: return {"valid": False, "reason": "invalid_dimensions"}
    unique = np.unique(annotation.reshape(-1, annotation.shape[-1] if annotation.ndim == 3 else 1), axis=0)
    if annotation.ndim == 2 and set(np.unique(annotation).tolist()).issubset({0, 1, 255}):
        semantics = "binary"
    elif annotation.ndim == 3 and len(unique) <= 32:
        semantics = "color_coded"
    else:
        semantics = "derivative_or_unknown"
    gray = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY) if annotation.ndim == 3 else annotation
    foreground = float(np.count_nonzero(gray) / gray.size)
    return {"valid": semantics in {"binary", "color_coded"} and foreground > 0, "semantics": semantics,
            "foreground_ratio": foreground, "height": int(image.shape[0]), "width": int(image.shape[1])}

def to_binary_mask(annotation_path: str | Path, semantics: str) -> np.ndarray:
    annotation = cv2.imread(str(annotation_path), cv2.IMREAD_UNCHANGED)
    if annotation is None: raise ValueError(f"Cannot decode annotation: {annotation_path}")
    gray = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY) if annotation.ndim == 3 else annotation
    if semantics not in {"binary", "color_coded"}: raise ValueError(f"Unverified annotation semantics: {semantics}")
    return (gray > 0).astype(np.uint8)
