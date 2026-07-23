import math

import numpy as np
import pytest

from ree_prospectivity.metrics import peak_signal_to_noise_ratio


def test_psnr_is_infinite_for_identical_arrays() -> None:
    values = np.ones((2, 2), dtype=np.float32)

    assert math.isinf(peak_signal_to_noise_ratio(values, values, data_range=1.0))


def test_psnr_matches_known_value() -> None:
    target = np.zeros((2, 2), dtype=np.float32)
    prediction = np.full((2, 2), 0.1, dtype=np.float32)

    result = peak_signal_to_noise_ratio(prediction, target, data_range=1.0)

    assert result == pytest.approx(20.0)
