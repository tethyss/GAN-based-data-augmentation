"""Typed experiment configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DownscaleStrategy = Literal["legacy_diagonal", "uniform"]
SplitPolicy = Literal["paper_faithful", "group_before_augmentation"]


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class DataConfig:
    channels: int = 28
    high_resolution_size: int = 32
    low_resolution_size: int = 8

    def __post_init__(self) -> None:
        _positive("channels", self.channels)
        _positive("high_resolution_size", self.high_resolution_size)
        _positive("low_resolution_size", self.low_resolution_size)
        if self.high_resolution_size % self.low_resolution_size:
            raise ValueError("high_resolution_size must be divisible by low_resolution_size")


@dataclass(frozen=True)
class DownscaleConfig:
    strategy: DownscaleStrategy = "legacy_diagonal"
    block_size: int = 4
    samples_per_block: int = 2

    def __post_init__(self) -> None:
        if self.strategy not in {"legacy_diagonal", "uniform"}:
            raise ValueError(f"unsupported downscale strategy: {self.strategy}")
        _positive("block_size", self.block_size)
        _positive("samples_per_block", self.samples_per_block)
        if self.samples_per_block > self.block_size**2:
            raise ValueError("samples_per_block exceeds the number of cells in a block")
        if self.strategy == "legacy_diagonal" and (
            self.block_size != 4 or self.samples_per_block != 2
        ):
            raise ValueError("legacy_diagonal requires block_size=4 and samples_per_block=2")


@dataclass(frozen=True)
class PreprocessingConfig:
    normalization: Literal["channel_minmax"] = "channel_minmax"
    window_size: int = 32
    border_policy: Literal["error", "reflect"] = "error"

    def __post_init__(self) -> None:
        if self.normalization != "channel_minmax":
            raise ValueError(f"unsupported normalization: {self.normalization}")
        _positive("window_size", self.window_size)
        if self.border_policy not in {"error", "reflect"}:
            raise ValueError(f"unsupported border policy: {self.border_policy}")


@dataclass(frozen=True)
class SplitConfig:
    validation_fraction: float = 0.2
    test_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.test_fraction < 0:
            raise ValueError("test_fraction must be non-negative")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation and test fractions must sum to less than one")


@dataclass(frozen=True)
class GeneratorConfig:
    features: int = 64
    residual_blocks: int = 16
    scale_factor: int = 4

    def __post_init__(self) -> None:
        _positive("features", self.features)
        _positive("residual_blocks", self.residual_blocks)
        _positive("scale_factor", self.scale_factor)
        if self.scale_factor & (self.scale_factor - 1):
            raise ValueError("scale_factor must be a power of two")


@dataclass(frozen=True)
class DiscriminatorConfig:
    base_features: int = 64

    def __post_init__(self) -> None:
        _positive("base_features", self.base_features)


@dataclass(frozen=True)
class ClassifierConfig:
    filters: tuple[int, int, int] = (64, 128, 256)
    hidden_dims: tuple[int, ...] = (1024, 512)
    dropout: float = 0.5

    def __post_init__(self) -> None:
        if len(self.filters) != 3:
            raise ValueError("classifier requires exactly three convolution filter counts")
        for index, value in enumerate((*self.filters, *self.hidden_dims)):
            _positive(f"classifier dimension {index}", value)
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    beta1: float
    beta2: float

    def __post_init__(self) -> None:
        _positive("epochs", self.epochs)
        _positive("batch_size", self.batch_size)
        _positive("learning_rate", self.learning_rate)
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")


@dataclass(frozen=True)
class GANLossConfig:
    reconstruction: Literal["l1"] = "l1"
    reconstruction_weight: float = 1.0
    adversarial_weight: float = 1e-3

    def __post_init__(self) -> None:
        if self.reconstruction != "l1":
            raise ValueError(f"unsupported reconstruction loss: {self.reconstruction}")
        _positive("reconstruction_weight", self.reconstruction_weight)
        if self.adversarial_weight < 0:
            raise ValueError("adversarial_weight must be non-negative")


@dataclass(frozen=True)
class EvaluationConfig:
    classification_threshold: float = 0.5
    psnr_data_range: float = 2.0

    def __post_init__(self) -> None:
        if not 0 < self.classification_threshold < 1:
            raise ValueError("classification_threshold must be in (0, 1)")
        _positive("psnr_data_range", self.psnr_data_range)


@dataclass(frozen=True)
class InferenceConfig:
    window_size: int = 32
    batch_size: int = 256
    border_policy: Literal["reflect"] = "reflect"

    def __post_init__(self) -> None:
        _positive("inference window_size", self.window_size)
        _positive("inference batch_size", self.batch_size)
        if self.border_policy != "reflect":
            raise ValueError(f"unsupported inference border policy: {self.border_policy}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    split_policy: SplitPolicy
    data: DataConfig
    preprocessing: PreprocessingConfig
    split: SplitConfig
    downscale: DownscaleConfig
    samples_per_site: int
    generator: GeneratorConfig
    discriminator: DiscriminatorConfig
    classifier: ClassifierConfig
    gan_training: TrainingConfig
    classifier_training: TrainingConfig
    gan_loss: GANLossConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name cannot be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.split_policy not in {"paper_faithful", "group_before_augmentation"}:
            raise ValueError(f"unsupported split policy: {self.split_policy}")
        _positive("samples_per_site", self.samples_per_site)
        expected_scale = self.data.high_resolution_size // self.data.low_resolution_size
        if self.generator.scale_factor != expected_scale:
            raise ValueError(
                "generator scale_factor must match high/low-resolution size ratio"
            )
        if self.downscale.block_size != expected_scale:
            raise ValueError("downscale block_size must match high/low-resolution size ratio")
        if self.preprocessing.window_size != self.data.high_resolution_size:
            raise ValueError("preprocessing window_size must match high_resolution_size")
        if self.inference.window_size != self.data.high_resolution_size:
            raise ValueError("inference window_size must match high_resolution_size")

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate an experiment configuration."""

        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ExperimentConfig:
        """Build a configuration from a TOML-shaped mapping."""

        experiment = raw["experiment"]
        training = raw["training"]
        classifier = raw["classifier"]
        return cls(
            name=str(experiment["name"]),
            seed=int(experiment["seed"]),
            split_policy=experiment["split_policy"],
            data=DataConfig(**raw["data"]),
            preprocessing=PreprocessingConfig(**raw["preprocessing"]),
            split=SplitConfig(**raw["split"]),
            downscale=DownscaleConfig(**raw["downscale"]),
            samples_per_site=int(raw["augmentation"]["samples_per_site"]),
            generator=GeneratorConfig(**raw["generator"]),
            discriminator=DiscriminatorConfig(**raw["discriminator"]),
            classifier=ClassifierConfig(
                filters=tuple(classifier["filters"]),
                hidden_dims=tuple(classifier["hidden_dims"]),
                dropout=float(classifier["dropout"]),
            ),
            gan_training=TrainingConfig(**training["gan"]),
            classifier_training=TrainingConfig(**training["classifier"]),
            gan_loss=GANLossConfig(**raw["loss"]["gan"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            inference=InferenceConfig(**raw["inference"]),
        )
