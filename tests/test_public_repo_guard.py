from pathlib import Path

from scripts.check_public_repo import find_violations


def test_guard_rejects_confidential_spreadsheet(tmp_path: Path) -> None:
    spreadsheet = tmp_path / "data.xlsx"
    spreadsheet.touch()

    violations = find_violations(spreadsheet, tmp_path)

    assert any("forbidden data" in violation for violation in violations)


def test_guard_rejects_unreviewed_data_file(tmp_path: Path) -> None:
    data_file = tmp_path / "data" / "sample.json"
    data_file.parent.mkdir()
    data_file.write_text("{}", encoding="utf-8")

    violations = find_violations(data_file, tmp_path)

    assert any("only reviewed data documentation" in violation for violation in violations)


def test_guard_allows_synthetic_data_readme(tmp_path: Path) -> None:
    readme = tmp_path / "data" / "synthetic" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("# Synthetic", encoding="utf-8")

    assert find_violations(readme, tmp_path) == []


def test_guard_rejects_workstation_path_in_maintained_code(tmp_path: Path) -> None:
    source = tmp_path / "src" / "loader.py"
    source.parent.mkdir()
    local_path = "E:" + "/data/private/input.csv"
    source.write_text(f'DATA = "{local_path}"', encoding="utf-8")

    violations = find_violations(source, tmp_path)

    assert any("workstation-specific absolute path" in violation for violation in violations)


def test_guard_permits_workstation_path_in_legacy_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "legacy" / "loader.py"
    source.parent.mkdir()
    local_path = "E:" + "/data/private/input.csv"
    source.write_text(f'DATA = "{local_path}"', encoding="utf-8")

    assert find_violations(source, tmp_path) == []
