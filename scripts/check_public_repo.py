"""Reject files that should not enter the public repository."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".csv",
    ".dbf",
    ".grd",
    ".h5",
    ".hdf",
    ".hdf5",
    ".joblib",
    ".keras",
    ".key",
    ".nc",
    ".npy",
    ".npz",
    ".pdf",
    ".pem",
    ".pickle",
    ".pkl",
    ".prj",
    ".pt",
    ".pth",
    ".sbn",
    ".sbx",
    ".shp",
    ".shx",
    ".tif",
    ".tiff",
    ".tsv",
    ".xls",
    ".xlsm",
    ".xlsx",
}

FORBIDDEN_DIRECTORY_NAMES = {
    "artifacts",
    "checkpoints",
    "images",
    "mlruns",
    "output",
    "outputs",
    "pre",
    "private",
    "raster",
    "rawdata",
    "saved_model",
    "test_images",
    "wandb",
}

ALLOWED_DATA_PATHS = {
    PurePosixPath("data/README.md"),
    PurePosixPath("data/synthetic/README.md"),
}

LOCAL_PATH_PATTERN = re.compile(
    r"(?i)(?:[a-z]:[\\/](?:users|data)[\\/]|/"
    r"home/[^/\s]+/)"
)
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def normalize_path(path: Path) -> PurePosixPath:
    """Return a repository-style path independent of the host OS."""

    return PurePosixPath(path.as_posix())


def find_violations(path: Path, repository_root: Path) -> list[str]:
    """Return public-repository policy violations for one file."""

    violations: list[str] = []
    absolute_path = path if path.is_absolute() else repository_root / path

    try:
        relative_path = absolute_path.resolve().relative_to(repository_root.resolve())
    except ValueError:
        return [f"{path}: path is outside the repository"]

    normalized = normalize_path(relative_path)
    lowered_parts = {part.lower() for part in normalized.parts}

    if (
        normalized.parts
        and normalized.parts[0].lower() == "data"
        and normalized not in ALLOWED_DATA_PATHS
    ):
        violations.append(f"{normalized}: only reviewed data documentation is publishable")

    forbidden_directories = lowered_parts & FORBIDDEN_DIRECTORY_NAMES
    if forbidden_directories:
        names = ", ".join(sorted(forbidden_directories))
        violations.append(f"{normalized}: forbidden private/output directory ({names})")

    if absolute_path.suffix.lower() in FORBIDDEN_SUFFIXES:
        violations.append(f"{normalized}: forbidden data, model, credential, or paper format")

    if not absolute_path.exists() or not absolute_path.is_file():
        return violations

    size = absolute_path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        violations.append(
            f"{normalized}: {size} bytes exceeds the {MAX_FILE_SIZE_BYTES}-byte public limit"
        )

    is_legacy = bool(normalized.parts and normalized.parts[0].lower() == "legacy")
    if not is_legacy and absolute_path.suffix.lower() in TEXT_SUFFIXES:
        try:
            content = absolute_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"{normalized}: text file is not valid UTF-8")
        else:
            if LOCAL_PATH_PATTERN.search(content):
                violations.append(f"{normalized}: contains a workstation-specific absolute path")

    return violations


def git_visible_files(repository_root: Path) -> list[Path]:
    """List tracked and untracked files, respecting .gitignore."""

    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]


def check_files(paths: Iterable[Path], repository_root: Path) -> list[str]:
    """Collect violations for all candidate files."""

    return [violation for path in paths for violation in find_violations(path, repository_root)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="check all tracked and non-ignored untracked files",
    )
    parser.add_argument("files", nargs="*", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    paths = git_visible_files(repository_root) if args.all else args.files
    violations = check_files(paths, repository_root)

    if violations:
        print("Public repository guard failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1

    print(f"Public repository guard passed for {len(paths)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
