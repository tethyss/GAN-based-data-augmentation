"""Aggregate binary-classification metrics without exposing sample records."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ree_prospectivity.data.contracts import SampleBatch


@dataclass(frozen=True)
class ClassificationReport:
    sample_count: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    brier_score: float
    roc_auc: float
    average_precision: float

    def as_public_metrics(self) -> dict[str, int | float]:
        return asdict(self)


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = labels == 1
    negative_count = int((~positives).sum())
    positive_count = int(positives.sum())
    if not positive_count or not negative_count:
        raise ValueError("ROC AUC requires both classes")
    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    sorted_probabilities = probabilities[order]
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and sorted_probabilities[end] == sorted_probabilities[start]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = ranks[positives].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive_count = int(labels.sum())
    if not positive_count:
        raise ValueError("average precision requires positive samples")
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positive_count)


def classification_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> ClassificationReport:
    """Calculate thresholded and ranking metrics from reviewed aggregate arrays."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("labels and probabilities must be matching one-dimensional arrays")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    predictions = probabilities >= threshold
    positives = labels == 1
    negatives = ~positives
    true_positive = int((predictions & positives).sum())
    true_negative = int((~predictions & negatives).sum())
    false_positive = int((predictions & negatives).sum())
    false_negative = int((~predictions & positives).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    return ClassificationReport(
        sample_count=len(labels),
        threshold=threshold,
        accuracy=(true_positive + true_negative) / len(labels),
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(np.finfo(float).eps, precision + recall),
        specificity=specificity,
        brier_score=float(np.mean((probabilities - labels) ** 2)),
        roc_auc=_roc_auc(labels, probabilities),
        average_precision=_average_precision(labels, probabilities),
    )


def evaluate_classifier(
    model: nn.Module,
    batch: SampleBatch,
    *,
    threshold: float = 0.5,
    device: str = "cpu",
) -> ClassificationReport:
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(batch.as_channels_first()).to(device)
        probabilities = torch.sigmoid(model(inputs)).cpu().numpy()
    return classification_report(batch.labels, probabilities, threshold=threshold)
