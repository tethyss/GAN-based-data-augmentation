import numpy as np
import pytest

from ree_prospectivity.config import DownscaleConfig
from ree_prospectivity.data import generate_synthetic_batch, random_downscale


@pytest.mark.parametrize("strategy", ["legacy_diagonal", "uniform"])
def test_random_downscale_is_seeded_and_shape_safe(strategy: str) -> None:
    batch = generate_synthetic_batch(sample_count=4, channels=3, seed=11)
    config = DownscaleConfig(strategy=strategy)

    first = random_downscale(batch.high_resolution, config, seed=7)
    second = random_downscale(batch.high_resolution, config, seed=7)
    different = random_downscale(batch.high_resolution, config, seed=8)

    assert first.shape == (4, 8, 8, 3)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_random_downscale_supports_one_image() -> None:
    image = np.arange(32 * 32 * 2, dtype=np.float32).reshape(32, 32, 2)

    output = random_downscale(image, DownscaleConfig(strategy="uniform"), seed=5)

    assert output.shape == (8, 8, 2)
    assert output.min() >= image.min()
    assert output.max() <= image.max()


def test_random_downscale_rejects_invalid_spatial_shape() -> None:
    image = np.zeros((30, 32, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="divisible"):
        random_downscale(image, DownscaleConfig(), seed=1)


def test_synthetic_batch_is_balanced_and_deterministic() -> None:
    first = generate_synthetic_batch(sample_count=6, channels=4, seed=3)
    second = generate_synthetic_batch(sample_count=6, channels=4, seed=3)

    assert first.high_resolution.shape == (6, 32, 32, 4)
    assert first.labels.tolist() == [0, 1, 0, 1, 0, 1]
    np.testing.assert_array_equal(first.high_resolution, second.high_resolution)
    np.testing.assert_array_equal(
        first.as_channels_first(),
        np.moveaxis(first.high_resolution, -1, 1),
    )
