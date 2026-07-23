"""Generate labeled synthetic descendants from a trained generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ree_prospectivity.config import DownscaleConfig
from ree_prospectivity.data.contracts import SampleBatch
from ree_prospectivity.data.downscaling import random_downscale
from ree_prospectivity.metrics import peak_signal_to_noise_ratio


@dataclass(frozen=True)
class AugmentationResult:
    samples: SampleBatch
    parent_indices: NDArray[np.int64]
    psnr_by_sample: NDArray[np.float64]

    @property
    def mean_psnr(self) -> float:
        return float(self.psnr_by_sample.mean())


def generate_augmented_samples(
    originals: SampleBatch,
    generator: nn.Module,
    *,
    samples_per_site: int,
    downscale_config: DownscaleConfig,
    seed: int,
    data_range: float = 2.0,
    device: str = "cpu",
) -> AugmentationResult:
    """Generate descendants while retaining immutable parent group identifiers."""

    if samples_per_site <= 0:
        raise ValueError("samples_per_site must be positive")
    rng = np.random.default_rng(seed)
    generated_samples: list[NDArray[np.float32]] = []
    generated_labels: list[int] = []
    generated_ids: list[str] = []
    group_ids: list[str] = []
    parent_indices: list[int] = []
    quality: list[float] = []
    generator.eval()

    with torch.no_grad():
        for parent_index, parent in enumerate(originals.high_resolution):
            for descendant_index in range(samples_per_site):
                descendant_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                low_resolution = random_downscale(
                    parent,
                    downscale_config,
                    seed=descendant_seed,
                )
                tensor = torch.from_numpy(
                    np.ascontiguousarray(np.moveaxis(low_resolution, -1, 0)[None])
                ).to(device)
                generated = generator(tensor).cpu().numpy()[0]
                generated_hwc = np.moveaxis(generated, 0, -1).astype(np.float32)
                generated_samples.append(generated_hwc)
                generated_labels.append(int(originals.labels[parent_index]))
                generated_ids.append(
                    f"{originals.sample_ids[parent_index]}-aug-{descendant_index:04d}"
                )
                group_ids.append(originals.group_ids[parent_index])
                parent_indices.append(parent_index)
                quality.append(
                    peak_signal_to_noise_ratio(
                        generated_hwc,
                        parent,
                        data_range=data_range,
                    )
                )

    samples = SampleBatch(
        high_resolution=np.stack(generated_samples),
        labels=np.asarray(generated_labels, dtype=np.int64),
        sample_ids=tuple(generated_ids),
        group_ids=tuple(group_ids),
    )
    return AugmentationResult(
        samples=samples,
        parent_indices=np.asarray(parent_indices, dtype=np.int64),
        psnr_by_sample=np.asarray(quality, dtype=np.float64),
    )
