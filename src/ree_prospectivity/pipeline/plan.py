"""Dependency-aware workflow plans for both validation policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ree_prospectivity.config import SplitPolicy


class PipelineStage(str, Enum):
    INGEST = "ingest"
    PREPARE = "prepare"
    SPLIT = "split"
    TRAIN_GAN = "train_gan"
    AUGMENT = "augment"
    VALIDATE_AUGMENTATION = "validate_augmentation"
    TRAIN_CLASSIFIER = "train_classifier"
    EVALUATE = "evaluate"
    PREDICT = "predict"
    EXPORT = "export"


@dataclass(frozen=True)
class StageDefinition:
    stage: PipelineStage
    dependencies: tuple[PipelineStage, ...]
    description: str
    confidential_outputs: bool = True


@dataclass(frozen=True)
class WorkflowPlan:
    """Ordered, validated stage definitions for one experiment policy."""

    split_policy: SplitPolicy
    stages: tuple[StageDefinition, ...]

    @classmethod
    def for_policy(cls, split_policy: SplitPolicy) -> WorkflowPlan:
        """Construct the published or leakage-safe workflow."""

        if split_policy == "paper_faithful":
            stages = _paper_faithful_stages()
        elif split_policy == "group_before_augmentation":
            stages = _leakage_safe_stages()
        else:
            raise ValueError(f"unsupported split policy: {split_policy}")
        plan = cls(split_policy=split_policy, stages=stages)
        plan.validate()
        return plan

    def validate(self) -> None:
        """Ensure every dependency precedes the stage that consumes it."""

        seen: set[PipelineStage] = set()
        for definition in self.stages:
            missing = set(definition.dependencies) - seen
            if missing:
                names = ", ".join(sorted(stage.value for stage in missing))
                raise ValueError(f"{definition.stage.value} has unresolved dependencies: {names}")
            if definition.stage in seen:
                raise ValueError(f"duplicate workflow stage: {definition.stage.value}")
            seen.add(definition.stage)

    def stages_through(self, target: PipelineStage) -> tuple[StageDefinition, ...]:
        """Return the prefix required to produce a target artifact."""

        for index, definition in enumerate(self.stages):
            if definition.stage == target:
                return self.stages[: index + 1]
        raise ValueError(f"target stage is not present in plan: {target.value}")


def _shared_tail(
    *,
    classifier_dependencies: tuple[PipelineStage, ...],
) -> tuple[StageDefinition, ...]:
    return (
        StageDefinition(
            PipelineStage.TRAIN_CLASSIFIER,
            classifier_dependencies,
            "Fit the prospectivity CNN and record epoch-level metrics.",
        ),
        StageDefinition(
            PipelineStage.EVALUATE,
            (PipelineStage.TRAIN_CLASSIFIER, PipelineStage.SPLIT),
            "Evaluate on untouched validation and spatial test groups.",
        ),
        StageDefinition(
            PipelineStage.PREDICT,
            (PipelineStage.TRAIN_CLASSIFIER, PipelineStage.PREPARE),
            "Apply the trained CNN with a full-area sliding window.",
        ),
        StageDefinition(
            PipelineStage.EXPORT,
            (PipelineStage.EVALUATE, PipelineStage.PREDICT),
            "Export confidential rasters and a redacted public summary.",
            confidential_outputs=False,
        ),
    )


def _paper_faithful_stages() -> tuple[StageDefinition, ...]:
    return (
        StageDefinition(
            PipelineStage.INGEST,
            (),
            "Load authorized feature rasters and labeled sites.",
        ),
        StageDefinition(
            PipelineStage.PREPARE,
            (PipelineStage.INGEST,),
            "Validate channels, normalize values, and extract 32x32 site windows.",
        ),
        StageDefinition(
            PipelineStage.TRAIN_GAN,
            (PipelineStage.PREPARE,),
            "Fit the multichannel super-resolution GAN on real site windows.",
        ),
        StageDefinition(
            PipelineStage.AUGMENT,
            (PipelineStage.TRAIN_GAN, PipelineStage.PREPARE),
            "Generate 200 synthetic descendants per original site.",
        ),
        StageDefinition(
            PipelineStage.SPLIT,
            (PipelineStage.AUGMENT,),
            "Reproduce the publication's post-augmentation random 80/20 split.",
        ),
        StageDefinition(
            PipelineStage.VALIDATE_AUGMENTATION,
            (PipelineStage.AUGMENT,),
            "Compare generated samples with real windows using PSNR and review panels.",
        ),
        *_shared_tail(
            classifier_dependencies=(
                PipelineStage.SPLIT,
                PipelineStage.VALIDATE_AUGMENTATION,
            )
        ),
    )


def _leakage_safe_stages() -> tuple[StageDefinition, ...]:
    return (
        StageDefinition(
            PipelineStage.INGEST,
            (),
            "Load authorized feature rasters and labeled sites.",
        ),
        StageDefinition(
            PipelineStage.PREPARE,
            (PipelineStage.INGEST,),
            "Validate channels and extract unnormalized 32x32 site windows.",
        ),
        StageDefinition(
            PipelineStage.SPLIT,
            (PipelineStage.PREPARE,),
            "Assign original site groups to train, validation, and spatial test partitions.",
        ),
        StageDefinition(
            PipelineStage.TRAIN_GAN,
            (PipelineStage.SPLIT,),
            "Fit preprocessing state and GAN using training groups only.",
        ),
        StageDefinition(
            PipelineStage.AUGMENT,
            (PipelineStage.TRAIN_GAN, PipelineStage.SPLIT),
            "Generate descendants only for groups allowed by the split manifest.",
        ),
        StageDefinition(
            PipelineStage.VALIDATE_AUGMENTATION,
            (PipelineStage.AUGMENT, PipelineStage.SPLIT),
            "Measure PSNR without crossing group or spatial split boundaries.",
        ),
        *_shared_tail(
            classifier_dependencies=(
                PipelineStage.AUGMENT,
                PipelineStage.SPLIT,
                PipelineStage.VALIDATE_AUGMENTATION,
            )
        ),
    )
