"""Tests for the project-local .venv check in the doctor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from deepbook.cli.doctor import _is_project_venv


def _project(tmp_path: Path, interpreter: str) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    executable = root / ".venv" / interpreter
    executable.parent.mkdir(parents=True)
    executable.touch()
    return root, executable


def test_windows_project_local_venv_accepted(tmp_path: Path) -> None:
    root, executable = _project(tmp_path, "Scripts/python.exe")

    assert _is_project_venv(root, str(executable), "Windows") is True


def test_posix_project_local_venv_accepted(tmp_path: Path) -> None:
    root, executable = _project(tmp_path, "bin/python")

    assert _is_project_venv(root, str(executable), "Linux") is True


def test_unrelated_venv_rejected(tmp_path: Path) -> None:
    root, _ = _project(tmp_path, "Scripts/python.exe")
    unrelated = tmp_path / "other" / ".venv" / "Scripts" / "python.exe"
    unrelated.parent.mkdir(parents=True)
    unrelated.touch()

    assert _is_project_venv(root, str(unrelated), "Windows") is False


def test_global_interpreter_rejected(tmp_path: Path) -> None:
    root, _ = _project(tmp_path, "bin/python")

    assert _is_project_venv(root, "/usr/bin/python3", "Linux") is False


def test_repo_root_detection_failure_is_unknown() -> None:
    with patch("deepbook.cli.doctor._repo_root", side_effect=OSError("unavailable")):
        assert _is_project_venv() is None
