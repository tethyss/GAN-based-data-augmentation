"""Command-line entry points for inspecting and executing workflow stages."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ree_prospectivity.config import ExperimentConfig
from ree_prospectivity.pipeline.plan import PipelineStage, WorkflowPlan
from ree_prospectivity.pipeline.runner import WorkflowBackend, WorkflowRunner


def _load_backend(factory_path: str) -> WorkflowBackend:
    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator:
        raise ValueError("backend factory must use the form 'module:function'")
    module = importlib.import_module(module_name)
    factory: Callable[[], WorkflowBackend] = getattr(module, attribute_name)
    return factory()


def _plan_payload(config_path: Path) -> dict[str, Any]:
    config = ExperimentConfig.from_toml(config_path)
    plan = WorkflowPlan.for_policy(config.split_policy)
    return {
        "experiment": config.name,
        "split_policy": config.split_policy,
        "stages": [
            {
                "name": definition.stage.value,
                "dependencies": [stage.value for stage in definition.dependencies],
                "description": definition.description,
                "confidential_outputs": definition.confidential_outputs,
            }
            for definition in plan.stages
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ree-prospectivity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="print the resolved workflow plan")
    plan_parser.add_argument("--config", type=Path, required=True)

    run_parser = subparsers.add_parser("run", help="execute the workflow through one stage")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--backend-factory", required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--run-directory", type=Path, required=True)
    run_parser.add_argument(
        "--target",
        type=PipelineStage,
        choices=list(PipelineStage),
        default=PipelineStage.EXPORT,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(_plan_payload(args.config), indent=2, ensure_ascii=False))
        return 0

    repository_root = Path.cwd()
    runner = WorkflowRunner(
        config_path=args.config,
        run_id=args.run_id,
        run_directory=args.run_directory,
        repository_root=repository_root,
        backend=_load_backend(args.backend_factory),
    )
    manifest = runner.run(target=args.target)
    print(json.dumps(manifest.public_dict(), indent=2, ensure_ascii=False, default=str))
    return 0
