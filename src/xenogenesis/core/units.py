"""Physical unit helpers."""
import numpy as np


def clamp(value: float, min_v: float, max_v: float) -> float:
    return float(np.clip(value, min_v, max_v))
