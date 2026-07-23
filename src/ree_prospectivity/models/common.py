"""Shared neural-network initialization."""

from __future__ import annotations

from torch import nn


def initialize_keras_style(module: nn.Module) -> None:
    """Approximate historical Keras defaults with Glorot-uniform weights."""

    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
