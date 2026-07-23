"""Deterministic, non-sensitive synthetic samples for tests and demos."""

from __future__ import annotations

import numpy as np

from ree_prospectivity.data.contracts import SampleBatch


def generate_synthetic_batch(
    *,
    sample_count: int = 8,
    image_size: int = 32,
    channels: int = 28,
    seed: int = 2022,
) -> SampleBatch:
    """Generate balanced fictional spatial fields with no research-data lineage."""

    if sample_count < 2 or sample_count % 2:
        raise ValueError("sample_count must be an even integer of at least two")
    if image_size < 8:
        raise ValueError("image_size must be at least eight")
    if channels < 1:
        raise ValueError("channels must be positive")

    rng = np.random.default_rng(seed)
    axis = np.linspace(-1.0, 1.0, image_size, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    images = np.empty((sample_count, image_size, image_size, channels), dtype=np.float32)
    labels = np.tile(np.array([0, 1], dtype=np.int64), sample_count // 2)

    for sample_index, label in enumerate(labels):
        center_x, center_y = rng.uniform(-0.45, 0.45, size=2)
        width = rng.uniform(0.18, 0.4)
        radial = np.exp(
            -((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2 * width**2)
        )
        for channel in range(channels):
            frequency = 1 + channel % 5
            phase = rng.uniform(0, 2 * np.pi)
            background = np.sin(frequency * grid_x + phase) * np.cos(
                frequency * grid_y - phase
            )
            signal_weight = (0.15 + 0.02 * channel) * float(label)
            noise = rng.normal(0, 0.03, size=(image_size, image_size))
            images[sample_index, :, :, channel] = (
                0.2 * background + signal_weight * radial + noise
            ).astype(np.float32)

    sample_ids = tuple(f"synthetic-sample-{index:04d}" for index in range(sample_count))
    group_ids = tuple(f"synthetic-group-{index:04d}" for index in range(sample_count))
    return SampleBatch(
        high_resolution=images,
        labels=labels,
        sample_ids=sample_ids,
        group_ids=group_ids,
    )
