"""Group-aware and paper-faithful sample partitioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ree_prospectivity.data.contracts import SampleBatch


@dataclass(frozen=True)
class DatasetSplits:
    train: SampleBatch
    validation: SampleBatch
    test: SampleBatch | None

    def assert_group_isolation(self) -> None:
        groups = [
            set(self.train.group_ids),
            set(self.validation.group_ids),
            set(self.test.group_ids) if self.test is not None else set(),
        ]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("group leakage detected across dataset splits")


def subset_batch(batch: SampleBatch, indices: np.ndarray) -> SampleBatch:
    positions = [int(index) for index in indices]
    return SampleBatch(
        high_resolution=batch.high_resolution[positions],
        labels=batch.labels[positions],
        sample_ids=tuple(batch.sample_ids[index] for index in positions),
        group_ids=tuple(batch.group_ids[index] for index in positions),
    )


def paper_random_split(
    batch: SampleBatch,
    *,
    validation_fraction: float = 0.2,
    seed: int,
) -> DatasetSplits:
    """Reproduce a random sample-level split after augmentation."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(batch.labels))
    validation_size = max(1, round(len(indices) * validation_fraction))
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    return DatasetSplits(
        train=subset_batch(batch, train_indices),
        validation=subset_batch(batch, validation_indices),
        test=None,
    )


def stratified_group_split(
    batch: SampleBatch,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int,
) -> DatasetSplits:
    """Assign whole parent groups to partitions while preserving class balance."""

    if validation_fraction <= 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative and validation must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")

    group_labels: dict[str, int] = {}
    for group_id, label in zip(batch.group_ids, batch.labels, strict=True):
        integer_label = int(label)
        previous = group_labels.setdefault(group_id, integer_label)
        if previous != integer_label:
            raise ValueError(f"group {group_id} contains conflicting labels")

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for label in (0, 1):
        groups = np.array(
            [group for group, group_label in group_labels.items() if group_label == label],
            dtype=object,
        )
        rng.shuffle(groups)
        validation_count = max(1, round(len(groups) * validation_fraction))
        test_count = round(len(groups) * test_fraction)
        if validation_count + test_count >= len(groups):
            raise ValueError(f"not enough class-{label} groups for requested split fractions")
        for group in groups[:validation_count]:
            assignment[str(group)] = "validation"
        for group in groups[validation_count : validation_count + test_count]:
            assignment[str(group)] = "test"
        for group in groups[validation_count + test_count :]:
            assignment[str(group)] = "train"

    split_indices = {
        name: np.array(
            [
                index
                for index, group_id in enumerate(batch.group_ids)
                if assignment[group_id] == name
            ],
            dtype=np.int64,
        )
        for name in ("train", "validation", "test")
    }
    test = (
        subset_batch(batch, split_indices["test"])
        if len(split_indices["test"])
        else None
    )
    splits = DatasetSplits(
        train=subset_batch(batch, split_indices["train"]),
        validation=subset_batch(batch, split_indices["validation"]),
        test=test,
    )
    splits.assert_group_isolation()
    return splits
