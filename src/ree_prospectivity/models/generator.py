"""Super-resolution generator for multichannel geoscience windows."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from ree_prospectivity.models.common import initialize_keras_style


class ResidualBlock(nn.Module):
    """Residual block ordered to match the supplied Keras implementation."""

    def __init__(self, channels: int, *, batch_norm_momentum: float = 0.01) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels, momentum=batch_norm_momentum),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels, momentum=batch_norm_momentum),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.layers(inputs)


class UpsampleBlock(nn.Module):
    """Nearest-neighbor upsampling followed by convolution and ReLU."""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        upsampled = functional.interpolate(inputs, scale_factor=2, mode="nearest")
        return self.activation(self.convolution(upsampled))


class SRGenerator(nn.Module):
    """Generate 32x32 multichannel samples from 8x8 inputs by default."""

    def __init__(
        self,
        *,
        channels: int = 28,
        features: int = 64,
        residual_blocks: int = 16,
        scale_factor: int = 4,
        batch_norm_momentum: float = 0.01,
    ) -> None:
        super().__init__()
        if channels <= 0 or features <= 0 or residual_blocks <= 0:
            raise ValueError("channels, features, and residual_blocks must be positive")
        if scale_factor <= 0 or scale_factor & (scale_factor - 1):
            raise ValueError("scale_factor must be a positive power of two")

        self.channels = channels
        self.scale_factor = scale_factor
        self.input_layer = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.residual_trunk = nn.Sequential(
            *[
                ResidualBlock(features, batch_norm_momentum=batch_norm_momentum)
                for _ in range(residual_blocks)
            ]
        )
        self.post_residual = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features, momentum=batch_norm_momentum),
        )

        upsample_layers: list[nn.Module] = []
        current_channels = features
        for _ in range(int(math.log2(scale_factor))):
            upsample_layers.append(UpsampleBlock(current_channels))
            current_channels = 256
        self.upsampling = nn.Sequential(*upsample_layers)
        self.output_layer = nn.Sequential(
            nn.Conv2d(current_channels, channels, kernel_size=9, padding=4),
            nn.Tanh(),
        )
        self.apply(initialize_keras_style)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(f"expected NCHW input with {self.channels} channels")
        shallow_features = self.input_layer(inputs)
        residual_features = self.residual_trunk(shallow_features)
        features = shallow_features + self.post_residual(residual_features)
        return self.output_layer(self.upsampling(features))
