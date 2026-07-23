"""Run manifest state and redacted serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ree_prospectivity.pipeline.artifacts import ArtifactKind, ArtifactRef, StageOutput
from ree_prospectivity.pipeline.plan import PipelineStage


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StageRecord:
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    artifacts: list[ArtifactRef] = field(default_factory=list)
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class RunManifest:
    run_id: str
    experiment_name: str
    split_policy: str
    config_digest: str
    created_at: str
    stages: dict[str, StageRecord]

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        experiment_name: str,
        split_policy: str,
        config_bytes: bytes,
        stage_names: tuple[PipelineStage, ...],
    ) -> RunManifest:
        return cls(
            run_id=run_id,
            experiment_name=experiment_name,
            split_policy=split_policy,
            config_digest=hashlib.sha256(config_bytes).hexdigest(),
            created_at=datetime.now(UTC).isoformat(),
            stages={stage.value: StageRecord() for stage in stage_names},
        )

    def start_stage(self, stage: PipelineStage) -> None:
        record = self.stages[stage.value]
        record.status = StageStatus.RUNNING
        record.started_at = datetime.now(UTC).isoformat()
        record.error = None

    def complete_stage(self, stage: PipelineStage, output: StageOutput) -> None:
        record = self.stages[stage.value]
        record.status = StageStatus.COMPLETED
        record.completed_at = datetime.now(UTC).isoformat()
        record.artifacts = list(output.artifacts)
        record.metrics = dict(output.metrics)
        record.warnings = list(output.warnings)

    def fail_stage(self, stage: PipelineStage, error: Exception) -> None:
        record = self.stages[stage.value]
        record.status = StageStatus.FAILED
        record.completed_at = datetime.now(UTC).isoformat()
        record.error = f"{type(error).__name__}: {error}"

    def outputs_for(self, stage: PipelineStage) -> StageOutput:
        record = self.stages[stage.value]
        if record.status != StageStatus.COMPLETED:
            raise RuntimeError(f"stage has not completed: {stage.value}")
        return StageOutput(
            artifacts=tuple(record.artifacts),
            metrics=dict(record.metrics),
            warnings=tuple(record.warnings),
        )

    def private_dict(self) -> dict[str, Any]:
        return asdict(self)

    def public_dict(self) -> dict[str, Any]:
        payload = self.private_dict()
        for stage in payload["stages"].values():
            stage["artifacts"] = [
                artifact.public_dict()
                for artifact in self._artifacts_from_payload(stage["artifacts"])
            ]
            stage["error"] = None
        return payload

    @staticmethod
    def _artifacts_from_payload(payloads: list[dict[str, Any]]) -> list[ArtifactRef]:
        return [
            ArtifactRef(
                kind=ArtifactKind(item["kind"]),
                uri=item["uri"],
                confidential=item["confidential"],
                checksum=item["checksum"],
                metadata=item["metadata"],
            )
            for item in payloads
        ]

    def write_private(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.private_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def write_public(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.public_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
