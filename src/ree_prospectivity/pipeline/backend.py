"""Stage service composition for a local confidential-data implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ree_prospectivity.pipeline.artifacts import StageOutput
from ree_prospectivity.pipeline.plan import PipelineStage
from ree_prospectivity.pipeline.runner import StageContext


class StageService(Protocol):
    def run(self, context: StageContext) -> StageOutput:
        """Run one bounded pipeline stage."""


@dataclass(frozen=True)
class ResearchServices:
    ingest: StageService
    prepare: StageService
    split: StageService
    train_gan: StageService
    augment: StageService
    validate_augmentation: StageService
    train_classifier: StageService
    evaluate: StageService
    predict: StageService
    export: StageService


class LocalResearchBackend:
    """Dispatch workflow stages to independently replaceable services."""

    def __init__(self, services: ResearchServices) -> None:
        self._services = {
            PipelineStage.INGEST: services.ingest,
            PipelineStage.PREPARE: services.prepare,
            PipelineStage.SPLIT: services.split,
            PipelineStage.TRAIN_GAN: services.train_gan,
            PipelineStage.AUGMENT: services.augment,
            PipelineStage.VALIDATE_AUGMENTATION: services.validate_augmentation,
            PipelineStage.TRAIN_CLASSIFIER: services.train_classifier,
            PipelineStage.EVALUATE: services.evaluate,
            PipelineStage.PREDICT: services.predict,
            PipelineStage.EXPORT: services.export,
        }

    def execute(self, stage: PipelineStage, context: StageContext) -> StageOutput:
        return self._services[stage].run(context)
