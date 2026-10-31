import numpy as np
from src.roi.extract import extract_roi
from src.roi.quality import clean_mask

def test_roi_bbox_is_clipped_and_resized():
    image = np.zeros((20, 30, 3), dtype=np.uint8); mask = np.zeros((20, 30), dtype=np.uint8); mask[2:8, 3:10] = 1
    roi, bbox = extract_roi(image, mask, (16, 16), .5)
    assert roi.shape == (16, 16, 3); assert bbox["x0"] == 0 and bbox["y0"] == 0

def test_empty_mask_is_rejected():
    try: extract_roi(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))
    except ValueError: return
    assert False

def test_quality_flags_empty_probability():
    _, quality = clean_mask(np.zeros((8, 8)), .5)
    assert quality["quality_status"] == "flagged_empty_mask"
