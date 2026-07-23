"""Supervised prospectivity classifier training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from ree_prospectivity.config import TrainingConfig
from ree_prospectivity.data.contracts import SampleBatch


@dataclass(frozen=True)
class ClassifierEpochMetrics:
    epoch: int
    training_loss: float
    training_accuracy: float
    validation_loss: float
    validation_accuracy: float


@dataclass(frozen=True)
class ClassifierTrainingResult:
    model: nn.Module
    history: tuple[ClassifierEpochMetrics, ...]
    best_epoch: int


def _as_tensor(batch: SampleBatch, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.from_numpy(batch.as_channels_first()).to(device)
    labels = torch.from_numpy(batch.labels.astype(np.float32)).to(device)
    return images, labels


def _evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(images)
        loss = functional.binary_cross_entropy_with_logits(logits, labels)
        accuracy = ((logits >= 0) == labels.bool()).float().mean()
    return float(loss.cpu()), float(accuracy.cpu())


def train_classifier(
    model: nn.Module,
    train: SampleBatch,
    validation: SampleBatch,
    *,
    training: TrainingConfig,
    seed: int,
    device: str = "cpu",
) -> ClassifierTrainingResult:
    """Train with deterministic sample permutations and retain the best state."""

    model.to(device)
    train_images, train_labels = _as_tensor(train, device)
    validation_images, validation_labels = _as_tensor(validation, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training.learning_rate,
        betas=(training.beta1, training.beta2),
    )
    rng = np.random.default_rng(seed)
    history: list[ClassifierEpochMetrics] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(training.epochs):
        model.train()
        batch_losses: list[float] = []
        correct = 0
        seen = 0
        permutation = rng.permutation(len(train.labels))
        for start in range(0, len(permutation), training.batch_size):
            indices = torch.as_tensor(
                permutation[start : start + training.batch_size],
                dtype=torch.long,
                device=device,
            )
            images = train_images.index_select(0, indices)
            labels = train_labels.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            correct += int(((logits.detach() >= 0) == labels.bool()).sum().cpu())
            seen += len(labels)

        validation_loss, validation_accuracy = _evaluate(
            model,
            validation_images,
            validation_labels,
        )
        epoch_metrics = ClassifierEpochMetrics(
            epoch=epoch + 1,
            training_loss=float(np.mean(batch_losses)),
            training_accuracy=correct / seen,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
        )
        history.append(epoch_metrics)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("classifier training completed without a model state")
    model.load_state_dict(best_state)
    return ClassifierTrainingResult(
        model=model,
        history=tuple(history),
        best_epoch=best_epoch,
    )
