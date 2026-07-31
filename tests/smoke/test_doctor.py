"""Smoke test for the environment doctor CLI.

Verifies exit behaviour and output content.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_doctor_runs_and_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    # Should exit 0 in normal venv with no paid data
    assert result.returncode == 0, f"doctor exited {result.returncode}: {result.stderr}"


def test_doctor_reports_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert "DeepBook version:" in result.stdout
    assert "0.1.0" in result.stdout


def test_doctor_reports_python_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert "Python version:" in result.stdout


def test_doctor_reports_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert "Repository root:" in result.stdout


def test_doctor_reports_paid_data_false_by_default() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "DEEPBOOK_ALLOW_PAID_DATA": "false"},
        cwd=Path(__file__).resolve().parents[2],
    )
    assert "Paid data authorized: False" in result.stdout


def test_doctor_does_not_print_secrets() -> None:
    """Doctor output must never contain API key patterns."""
    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    # No db- pattern (Databento key prefix)
    assert "db-" not in result.stdout.lower()


def test_doctor_exits_nonzero_when_paid_data_unsafe() -> None:
    """When ALLOW_PAID_DATA=true but MAX_PAID_REQUEST_USD=0, exit nonzero."""
    import os

    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DEEPBOOK_ALLOW_PAID_DATA": "true",
            "DEEPBOOK_MAX_PAID_REQUEST_USD": "0",
        },
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode != 0, f"expected nonzero exit, got {result.returncode}"
