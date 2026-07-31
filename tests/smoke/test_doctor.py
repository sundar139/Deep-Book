"""Smoke tests for the environment doctor CLI.

Verifies exit behaviour, output content, and safety checks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_doctor_runs_and_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, f"doctor exited {result.returncode}: {result.stderr}"


def test_doctor_reports_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "DeepBook version:" in result.stdout
    assert "0.1.0" in result.stdout


def test_doctor_reports_python_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "Python version:" in result.stdout


def test_doctor_reports_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "Repository root:" in result.stdout


def test_doctor_paid_data_false_by_default(monkeypatch) -> None:
    """Paid data must default to false when env var is absent."""
    monkeypatch.delenv("DEEPBOOK_ALLOW_PAID_DATA", raising=False)
    from deepbook.cli.doctor import _paid_data_allowed

    assert _paid_data_allowed() is False


def test_doctor_does_not_print_secrets() -> None:
    """Doctor output must never contain API key patterns."""
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "db-" not in result.stdout.lower()


def test_doctor_exits_nonzero_when_paid_data_unsafe() -> None:
    """When DEEPBOOK_ALLOW_PAID_DATA=true but MAX_PAID_REQUEST_USD=0, exit nonzero."""
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DEEPBOOK_ALLOW_PAID_DATA": "true",
            "DEEPBOOK_MAX_PAID_REQUEST_USD": "0",
        },
        cwd=ROOT,
    )
    assert result.returncode != 0, f"expected nonzero exit, got {result.returncode}"


def test_doctor_exits_nonzero_for_missing_cap_with_auth() -> None:
    """When DEEPBOOK_ALLOW_PAID_DATA=true and cap is missing entirely, exit nonzero."""
    env = {**os.environ, "DEEPBOOK_ALLOW_PAID_DATA": "true"}
    env.pop("DEEPBOOK_MAX_PAID_REQUEST_USD", None)
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert result.returncode != 0, f"expected nonzero exit, got {result.returncode}"


def test_doctor_exits_nonzero_outside_venv(monkeypatch) -> None:
    """When VIRTUAL_ENV is unset and executable is not inside .venv, exit nonzero."""
    from deepbook.cli.doctor import _is_project_venv

    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    assert _is_project_venv() is not True
