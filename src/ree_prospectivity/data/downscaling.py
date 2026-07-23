"""Seeded random downscaling described by the source publication."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ree_prospectivity.config import DownscaleConfig


def _validate_images(images: NDArray[np.floating], block_size: int) -> None:
    if images.ndim not in {3, 4}:
        raise ValueError("images must have shape (H, W, C) or (N, H, W, C)")
    height, width = images.shape[-3:-1]
    if height % block_size or width % block_size:
        raise ValueError("image dimensions must be divisible by block_size")
    if not np.isfinite(images).all():
        raise ValueError("images contain non-finite values")


def _legacy_diagonal_indices(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    """Reproduce the paired-quadrant sampling in the supplied loader."""

    first_row = int(rng.integers(0, 4))
    if first_row > 1:
        second_row = int(rng.integers(0, 2))
    else:
        second_row = int(rng.integers(2, 4))
    first_column = int(rng.integers(0, 2))
    second_column = int(rng.integers(2, 4))
    return ((first_row, first_column), (second_row, second_column))


def _uniform_indices(
    rng: np.random.Generator,
    block_size: int,
    samples_per_block: int,
) -> tuple[tuple[int, int], ...]:
    flat_indices = rng.choice(block_size**2, size=samples_per_block, replace=False)
    return tuple(
        (int(index // block_size), int(index % block_size))
        for index in flat_indices
    )


def random_downscale(
    images: NDArray[np.floating],
    config: DownscaleConfig,
    *,
    seed: int,
) -> NDArray[np.float32]:
    """Downscale NHWC/HWC arrays by averaging seeded random cells per block."""

    array = np.asarray(images)
    _validate_images(array, config.block_size)
    single_image = array.ndim == 3
    batched = array[None, ...] if single_image else array
    batch_size, height, width, channels = batched.shape
    output = np.empty(
        (
            batch_size,
            height // config.block_size,
            width // config.block_size,
            channels,
        ),
        dtype=np.float32,
    )
    rng = np.random.default_rng(seed)

    for batch_index in range(batch_size):
        for output_row in range(output.shape[1]):
            row_start = output_row * config.block_size
            for output_column in range(output.shape[2]):
                column_start = output_column * config.block_size
                block = batched[
                    batch_index,
                    row_start : row_start + config.block_size,
                    column_start : column_start + config.block_size,
                    :,
                ]
                if config.strategy == "legacy_diagonal":
                    indices = _legacy_diagonal_indices(rng)
                else:
                    indices = _uniform_indices(
                        rng,
                        config.block_size,
                        config.samples_per_block,
                    )
                selected = np.stack([block[row, column] for row, column in indices])
                output[batch_index, output_row, output_column] = selected.mean(
                    axis=0,
                    dtype=np.float64,
                )

    return output[0] if single_image else output
