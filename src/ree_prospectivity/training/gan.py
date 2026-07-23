"""Alternating discriminator and generator training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from ree_prospectivity.config import DownscaleConfig, TrainingConfig
from ree_prospectivity.data.downscaling import random_downscale


@dataclass(frozen=True)
class GANEpochMetrics:
    epoch: int
    discriminator_loss: float
    generator_loss: float
    reconstruction_loss: float
    adversarial_loss: float


@dataclass(frozen=True)
class GANTrainingResult:
    generator: nn.Module
    discriminator: nn.Module
    history: tuple[GANEpochMetrics, ...]


def _tensor_from_nhwc(values: np.ndarray, device: str) -> torch.Tensor:
    channels_first = np.ascontiguousarray(np.moveaxis(values, -1, 1))
    return torch.from_numpy(channels_first).to(device)


def train_gan(
    high_resolution: np.ndarray,
    generator: nn.Module,
    discriminator: nn.Module,
    *,
    downscale_config: DownscaleConfig,
    training: TrainingConfig,
    seed: int,
    adversarial_weight: float = 1e-3,
    reconstruction_weight: float = 1.0,
    device: str = "cpu",
) -> GANTrainingResult:
    """Train the multichannel GAN using reconstruction and adversarial objectives."""

    if high_resolution.ndim != 4:
        raise ValueError("high_resolution must have shape (N, H, W, C)")
    if adversarial_weight < 0 or reconstruction_weight <= 0:
        raise ValueError("loss weights are invalid")
    generator.to(device)
    discriminator.to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=training.learning_rate,
        betas=(training.beta1, training.beta2),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=training.learning_rate,
        betas=(training.beta1, training.beta2),
    )
    rng = np.random.default_rng(seed)
    history: list[GANEpochMetrics] = []

    for epoch in range(training.epochs):
        epoch_losses: list[tuple[float, float, float, float]] = []
        permutation = rng.permutation(len(high_resolution))
        for start in range(0, len(high_resolution), training.batch_size):
            batch_indices = permutation[start : start + training.batch_size]
            real_nhwc = high_resolution[batch_indices].astype(np.float32, copy=False)
            downscale_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            low_nhwc = random_downscale(
                real_nhwc,
                downscale_config,
                seed=downscale_seed,
            )
            real = _tensor_from_nhwc(real_nhwc, device)
            low = _tensor_from_nhwc(low_nhwc, device)

            generator.train()
            discriminator.train()
            with torch.no_grad():
                detached_fake = generator(low)
            discriminator_optimizer.zero_grad(set_to_none=True)
            real_logits = discriminator(real)
            fake_logits = discriminator(detached_fake)
            discriminator_loss = 0.5 * (
                functional.binary_cross_entropy_with_logits(
                    real_logits,
                    torch.ones_like(real_logits),
                )
                + functional.binary_cross_entropy_with_logits(
                    fake_logits,
                    torch.zeros_like(fake_logits),
                )
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            generator_optimizer.zero_grad(set_to_none=True)
            fake = generator(low)
            generated_logits = discriminator(fake)
            adversarial_loss = functional.binary_cross_entropy_with_logits(
                generated_logits,
                torch.ones_like(generated_logits),
            )
            reconstruction_loss = functional.l1_loss(fake, real)
            generator_loss = (
                reconstruction_weight * reconstruction_loss
                + adversarial_weight * adversarial_loss
            )
            generator_loss.backward()
            generator_optimizer.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            epoch_losses.append(
                (
                    float(discriminator_loss.detach().cpu()),
                    float(generator_loss.detach().cpu()),
                    float(reconstruction_loss.detach().cpu()),
                    float(adversarial_loss.detach().cpu()),
                )
            )

        means = np.asarray(epoch_losses, dtype=np.float64).mean(axis=0)
        history.append(
            GANEpochMetrics(
                epoch=epoch + 1,
                discriminator_loss=float(means[0]),
                generator_loss=float(means[1]),
                reconstruction_loss=float(means[2]),
                adversarial_loss=float(means[3]),
            )
        )

    return GANTrainingResult(
        generator=generator,
        discriminator=discriminator,
        history=tuple(history),
    )
