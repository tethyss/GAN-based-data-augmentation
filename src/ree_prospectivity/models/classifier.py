"""Convolutional classifier for mineral prospectivity."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from ree_prospectivity.models.common import initialize_keras_style


class ProspectivityCNN(nn.Module):
    """Configurable reconstruction of the CNN shown in publication Figure 6."""

    def __init__(
        self,
        *,
        channels: int = 28,
        input_size: int = 32,
        filters: Sequence[int] = (64, 128, 256),
        hidden_dims: Sequence[int] = (1024, 512),
        dropout: float = 0.5,
        batch_norm_momentum: float = 0.01,
    ) -> None:
        super().__init__()
        if channels <= 0 or input_size <= 0:
            raise ValueError("channels and input_size must be positive")
        if len(filters) != 3 or any(value <= 0 for value in filters):
            raise ValueError("filters must contain three positive values")
        if not hidden_dims or any(value <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must contain positive values")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if input_size % 8:
            raise ValueError("input_size must be divisible by eight")

        self.channels = channels
        self.input_size = input_size
        first, second, third = filters
        self.features = nn.Sequential(
            nn.Conv2d(channels, first, kernel_size=3, padding=1),
            nn.BatchNorm2d(first, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(first, second, kernel_size=3, padding=1),
            nn.BatchNorm2d(second, momentum=batch_norm_momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(second, third, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        flattened_features = third * (input_size // 8) ** 2
        classifier_layers: list[nn.Module] = [nn.Flatten()]
        input_features = flattened_features
        for hidden_features in hidden_dims:
            classifier_layers.extend(
                [
                    nn.Linear(input_features, hidden_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            input_features = hidden_features
        classifier_layers.append(nn.Linear(input_features, 1))
        self.classifier = nn.Sequential(*classifier_layers)
        self.apply(initialize_keras_style)

    def forward(self, inputs: Tensor) -> Tensor:
        if (
            inputs.ndim != 4
            or inputs.shape[1] != self.channels
            or inputs.shape[-2:] != (self.input_size, self.input_size)
        ):
            raise ValueError(
                f"expected input shape (N, {self.channels}, "
                f"{self.input_size}, {self.input_size})"
            )
        return self.classifier(self.features(inputs)).squeeze(-1)
