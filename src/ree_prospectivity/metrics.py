"""Evaluation metrics shared by training and tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def peak_signal_to_noise_ratio(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    data_range: float,
) -> float:
    """Compute PSNR using an explicit data range."""

    if data_range <= 0:
        raise ValueError("data_range must be positive")
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must have the same shape")
    if not np.isfinite(prediction_array).all() or not np.isfinite(target_array).all():
        raise ValueError("prediction and target must be finite")
    mean_squared_error = float(np.mean(np.square(prediction_array - target_array)))
    if mean_squared_error == 0:
        return float("inf")
    return float(10 * np.log10(data_range**2 / mean_squared_error))
