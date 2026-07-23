"""Feature-cube validation, scaling, and labeled-window extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ree_prospectivity.data.contracts import SampleBatch


@dataclass(frozen=True)
class FeatureCube:
    """Aligned HWC feature rasters and channel metadata."""

    values: NDArray[np.float32]
    channel_names: tuple[str, ...]
    cell_size: float
    crs: str

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.dtype != np.float32:
            raise TypeError("values must be a float32 array with shape (H, W, C)")
        if self.values.shape[-1] != len(self.channel_names):
            raise ValueError("channel_names must match the cube channel dimension")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be unique")
        if not np.isfinite(self.values).all():
            raise ValueError("feature cube contains non-finite values")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if not self.crs.strip():
            raise ValueError("crs cannot be empty")


@dataclass(frozen=True)
class LabeledSite:
    site_id: str
    row: int
    column: int
    label: int
    region: str | None = None

    def __post_init__(self) -> None:
        if not self.site_id:
            raise ValueError("site_id cannot be empty")
        if self.row < 0 or self.column < 0:
            raise ValueError("site indices must be non-negative")
        if self.label not in {0, 1}:
            raise ValueError("site labels must be binary")


@dataclass(frozen=True)
class ChannelMinMaxScaler:
    """Channel-wise transformation into the generator's [-1, 1] range."""

    minimum: NDArray[np.float32]
    maximum: NDArray[np.float32]

    @classmethod
    def fit(cls, values: NDArray[np.float32]) -> ChannelMinMaxScaler:
        if values.ndim < 2:
            raise ValueError("values must include a channel dimension")
        flattened = values.reshape(-1, values.shape[-1])
        minimum = flattened.min(axis=0).astype(np.float32)
        maximum = flattened.max(axis=0).astype(np.float32)
        if np.any(maximum <= minimum):
            raise ValueError("every channel must have a non-zero finite range")
        return cls(minimum=minimum, maximum=maximum)

    def transform(self, values: NDArray[np.float32]) -> NDArray[np.float32]:
        self._validate_channels(values)
        scaled = 2 * (values - self.minimum) / (self.maximum - self.minimum) - 1
        return scaled.astype(np.float32)

    def inverse_transform(self, values: NDArray[np.float32]) -> NDArray[np.float32]:
        self._validate_channels(values)
        restored = (values + 1) * 0.5 * (self.maximum - self.minimum) + self.minimum
        return restored.astype(np.float32)

    def _validate_channels(self, values: NDArray[np.float32]) -> None:
        if values.shape[-1] != self.minimum.shape[0]:
            raise ValueError("values do not match fitted channel count")


def extract_site_windows(
    cube: FeatureCube,
    sites: tuple[LabeledSite, ...],
    *,
    window_size: int = 32,
    border_policy: str = "error",
) -> SampleBatch:
    """Extract site-centered windows without silently changing labels or positions."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if border_policy not in {"error", "reflect"}:
        raise ValueError("border_policy must be 'error' or 'reflect'")
    if len({site.site_id for site in sites}) != len(sites):
        raise ValueError("site identifiers must be unique")

    before = window_size // 2
    after = window_size - before
    values = cube.values
    if border_policy == "reflect":
        values = np.pad(
            values,
            ((before, after - 1), (before, after - 1), (0, 0)),
            mode="reflect",
        )

    windows: list[NDArray[np.float32]] = []
    for site in sites:
        if border_policy == "reflect":
            row, column = site.row + before, site.column + before
        else:
            row, column = site.row, site.column
            if (
                row - before < 0
                or column - before < 0
                or row + after > values.shape[0]
                or column + after > values.shape[1]
            ):
                raise ValueError(f"site {site.site_id} cannot support a full window")
        window = values[row - before : row + after, column - before : column + after]
        if window.shape[:2] != (window_size, window_size):
            raise RuntimeError(f"unexpected extracted shape for site {site.site_id}")
        windows.append(window)

    return SampleBatch(
        high_resolution=np.stack(windows).astype(np.float32),
        labels=np.asarray([site.label for site in sites], dtype=np.int64),
        sample_ids=tuple(site.site_id for site in sites),
        group_ids=tuple(site.site_id for site in sites),
    )
