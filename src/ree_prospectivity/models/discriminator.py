"""Patch discriminator for real and generated geoscience windows."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ree_prospectivity.models.common import initialize_keras_style


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        batch_normalization: bool,
        batch_norm_momentum: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]
        if batch_normalization:
            layers.append(nn.BatchNorm2d(out_channels, momentum=batch_norm_momentum))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class PatchDiscriminator(nn.Module):
    """Return a 2x2 logits map for a 32x32 input."""

    def __init__(
        self,
        *,
        channels: int = 28,
        base_features: int = 64,
        batch_norm_momentum: float = 0.01,
    ) -> None:
        super().__init__()
        if channels <= 0 or base_features <= 0:
            raise ValueError("channels and base_features must be positive")
        self.channels = channels
        feature_schedule = (
            base_features,
            base_features,
            base_features * 2,
            base_features * 2,
            base_features * 4,
            base_features * 4,
            base_features * 8,
            base_features * 8,
        )
        stride_schedule = (1, 2, 1, 2, 1, 2, 1, 2)
        blocks: list[nn.Module] = []
        input_features = channels
        for index, (output_features, stride) in enumerate(
            zip(feature_schedule, stride_schedule, strict=True)
        ):
            blocks.append(
                DiscriminatorBlock(
                    input_features,
                    output_features,
                    stride=stride,
                    batch_normalization=index != 0,
                    batch_norm_momentum=batch_norm_momentum,
                )
            )
            input_features = output_features
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Conv2d(input_features, base_features * 16, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_features * 16, 1, kernel_size=1),
        )
        self.apply(initialize_keras_style)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(f"expected NCHW input with {self.channels} channels")
        return self.classifier(self.features(inputs))

    def predict_proba(self, inputs: Tensor) -> Tensor:
        return torch.sigmoid(self(inputs))
