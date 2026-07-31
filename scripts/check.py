"""Aggregate quality gate script for DeepBook.

Usage:
    python scripts/check.py

Runs: ruff check, ruff format check, mypy, unit/property/smoke tests,
repository-policy check, pre-commit all-files, package build, environment doctor.
Exits zero only when all gates pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], desc: str) -> int:
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n  FAILED: {desc}")
    else:
        print(f"  PASSED: {desc}")
    return result.returncode


def main() -> int:
    failures = 0

    failures += run(
        [sys.executable, "-m", "ruff", "check", "."],
        "Ruff lint",
    )
    failures += run(
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        "Ruff format check",
    )
    failures += run(
        [sys.executable, "-m", "mypy", "src"],
        "Mypy type check",
    )
    failures += run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit",
            "tests/property",
            "tests/smoke",
            "-q",
            "--tb=short",
        ],
        "Pytest (unit + property + smoke)",
    )
    failures += run(
        [sys.executable, str(ROOT / "scripts" / "check_repository_policy.py")],
        "Repository policy check",
    )
    failures += run(
        [sys.executable, "-m", "pre_commit", "run", "--all-files"],
        "Pre-commit (all files)",
    )
    failures += run(
        [sys.executable, "-m", "build"],
        "Package build",
    )
    failures += run(
        [sys.executable, "-m", "deepbook.cli.doctor"],
        "Environment doctor",
    )

    print(f"\n{'=' * 60}")
    if failures == 0:
        print("  ALL GATES PASSED")
    else:
        print(f"  {failures} GATE(S) FAILED")
    print(f"{'=' * 60}")
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
