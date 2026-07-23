"""Execute a workflow plan through an injected data and training backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ree_prospectivity.config import ExperimentConfig
from ree_prospectivity.pipeline.artifacts import StageOutput, ensure_private_output_path
from ree_prospectivity.pipeline.manifest import RunManifest
from ree_prospectivity.pipeline.plan import PipelineStage, WorkflowPlan


@dataclass(frozen=True)
class StageContext:
    run_id: str
    run_directory: Path
    repository_root: Path
    config: ExperimentConfig
    manifest: RunManifest

    def dependency_output(self, stage: PipelineStage) -> StageOutput:
        return self.manifest.outputs_for(stage)


class WorkflowBackend(Protocol):
    """Boundary implemented by the local confidential-data pipeline."""

    def execute(self, stage: PipelineStage, context: StageContext) -> StageOutput:
        """Execute one stage and return references rather than raw data."""


class WorkflowRunner:
    """Run, resume, and audit the complete reconstruction workflow."""

    def __init__(
        self,
        *,
        config_path: Path,
        run_id: str,
        run_directory: Path,
        repository_root: Path,
        backend: WorkflowBackend,
    ) -> None:
        self.config_path = config_path
        self.config = ExperimentConfig.from_toml(config_path)
        self.plan = WorkflowPlan.for_policy(self.config.split_policy)
        self.backend = backend
        ensure_private_output_path(run_directory, repository_root)
        self.context = StageContext(
            run_id=run_id,
            run_directory=run_directory,
            repository_root=repository_root,
            config=self.config,
            manifest=RunManifest.create(
                run_id=run_id,
                experiment_name=self.config.name,
                split_policy=self.config.split_policy,
                config_bytes=config_path.read_bytes(),
                stage_names=tuple(definition.stage for definition in self.plan.stages),
            ),
        )

    def run(self, *, target: PipelineStage = PipelineStage.EXPORT) -> RunManifest:
        """Execute all required stages through the requested target."""

        private_manifest_path = self.context.run_directory / "manifest.private.json"
        for definition in self.plan.stages_through(target):
            stage = definition.stage
            self.context.manifest.start_stage(stage)
            self.context.manifest.write_private(private_manifest_path)
            try:
                output = self.backend.execute(stage, self.context)
            except Exception as error:
                self.context.manifest.fail_stage(stage, error)
                self.context.manifest.write_private(private_manifest_path)
                raise
            self.context.manifest.complete_stage(stage, output)
            self.context.manifest.write_private(private_manifest_path)
        return self.context.manifest
