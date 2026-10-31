import numpy as np
from src.common.metrics import validation_threshold

def test_threshold_is_bounded():
    assert 0 <= validation_threshold(np.array([0, 1]), np.array([.1, .9])) <= 1
