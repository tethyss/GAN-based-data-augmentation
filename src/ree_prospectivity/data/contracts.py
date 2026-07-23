"""Validated in-memory contracts for model-ready samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SampleBatch:
    """A group-aware batch of high-resolution samples in NHWC layout."""

    high_resolution: NDArray[np.float32]
    labels: NDArray[np.int64]
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        images = np.asarray(self.high_resolution)
        labels = np.asarray(self.labels)
        if images.ndim != 4:
            raise ValueError("high_resolution must have shape (N, H, W, C)")
        if images.dtype != np.float32:
            raise TypeError("high_resolution must use float32")
        if not np.isfinite(images).all():
            raise ValueError("high_resolution contains non-finite values")
        if labels.ndim != 1 or labels.dtype != np.int64:
            raise TypeError("labels must be a one-dimensional int64 array")

        sample_count = images.shape[0]
        if not (
            labels.shape[0]
            == len(self.sample_ids)
            == len(self.group_ids)
            == sample_count
        ):
            raise ValueError("all sample fields must have the same leading dimension")
        if len(set(self.sample_ids)) != sample_count:
            raise ValueError("sample_ids must be unique")
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("labels must be binary")

    @property
    def channels(self) -> int:
        return int(self.high_resolution.shape[-1])

    def as_channels_first(self) -> NDArray[np.float32]:
        """Return a contiguous NCHW view suitable for PyTorch."""

        return np.ascontiguousarray(np.moveaxis(self.high_resolution, -1, 1))
