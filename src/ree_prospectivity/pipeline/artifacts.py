"""Serializable references to confidential and publishable workflow artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactKind(str, Enum):
    RAW_FEATURE_CUBE = "raw_feature_cube"
    SITE_LABELS = "site_labels"
    PREPROCESSING_STATE = "preprocessing_state"
    PREPARED_WINDOWS = "prepared_windows"
    SPLIT_MANIFEST = "split_manifest"
    GENERATOR_CHECKPOINT = "generator_checkpoint"
    DISCRIMINATOR_CHECKPOINT = "discriminator_checkpoint"
    AUGMENTED_WINDOWS = "augmented_windows"
    AUGMENTATION_REPORT = "augmentation_report"
    CLASSIFIER_CHECKPOINT = "classifier_checkpoint"
    EVALUATION_REPORT = "evaluation_report"
    PROSPECTIVITY_GRID = "prospectivity_grid"
    PUBLIC_SUMMARY = "public_summary"


@dataclass(frozen=True)
class ArtifactRef:
    """Reference an artifact without embedding its potentially sensitive content."""

    kind: ArtifactKind
    uri: str
    confidential: bool = True
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        """Return metadata safe for a public run summary."""

        payload = asdict(self)
        payload["kind"] = self.kind.value
        if self.confidential:
            payload["uri"] = "<confidential>"
            payload["checksum"] = None
            payload["metadata"] = {
                key: value
                for key, value in self.metadata.items()
                if key in {"shape", "dtype", "schema_version", "sample_count"}
            }
        return payload


@dataclass(frozen=True)
class StageOutput:
    """Artifacts, aggregate metrics, and warnings produced by one stage."""

    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def ensure_private_output_path(path: Path, repository_root: Path) -> None:
    """Reject artifact output paths that could enter the public source tree."""

    resolved = path.resolve()
    root = repository_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return
    if not relative.parts or relative.parts[0] not in {"output", "outputs", "artifacts"}:
        raise ValueError(
            "workflow artifacts inside the repository must be written below "
            "output/, outputs/, or artifacts/"
        )
