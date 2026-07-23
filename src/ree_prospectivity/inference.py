"""Memory-bounded full-area sliding-window inference."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


def predict_prospectivity_grid(
    feature_cube: NDArray[np.float32],
    model: nn.Module,
    *,
    window_size: int = 32,
    batch_size: int = 256,
    transform: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None,
    device: str = "cpu",
) -> NDArray[np.float32]:
    """Predict every cell while keeping only one inference batch in memory."""

    if feature_cube.ndim != 3:
        raise ValueError("feature_cube must have shape (H, W, C)")
    if window_size <= 0 or batch_size <= 0:
        raise ValueError("window_size and batch_size must be positive")
    if not np.isfinite(feature_cube).all():
        raise ValueError("feature_cube contains non-finite values")

    before = window_size // 2
    after = window_size - before
    padded = np.pad(
        feature_cube,
        ((before, after - 1), (before, after - 1), (0, 0)),
        mode="reflect",
    )
    height, width, _ = feature_cube.shape
    output = np.empty((height, width), dtype=np.float32)
    coordinates: list[tuple[int, int]] = []
    windows: list[NDArray[np.float32]] = []
    model.eval()

    def flush() -> None:
        if not windows:
            return
        batch = np.stack(windows).astype(np.float32)
        if transform is not None:
            batch = transform(batch)
        channels_first = np.ascontiguousarray(np.moveaxis(batch, -1, 1))
        with torch.no_grad():
            probabilities = torch.sigmoid(
                model(torch.from_numpy(channels_first).to(device))
            ).cpu().numpy()
        for (row, column), probability in zip(
            coordinates,
            probabilities,
            strict=True,
        ):
            output[row, column] = float(probability)
        coordinates.clear()
        windows.clear()

    for row in range(height):
        for column in range(width):
            windows.append(
                padded[row : row + window_size, column : column + window_size]
            )
            coordinates.append((row, column))
            if len(windows) == batch_size:
                flush()
    flush()
    return output
