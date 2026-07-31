"""Environment diagnostic command for DeepBook.

Usage:
    python -m deepbook.cli.doctor
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_repo_root(),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            capture_output=True,
            cwd=_repo_root(),
            timeout=5,
        )
        return result.returncode != 0
    except Exception:
        return None


def _venv_path() -> str | None:
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return venv
    exe = Path(sys.executable)
    if exe.parent.name == "Scripts" and (exe.parents[1] / "pyvenv.cfg").exists():
        return str(exe.parents[1])
    return None


def _paid_data_allowed() -> bool:
    val = os.environ.get("DEEPBOOK_ALLOW_PAID_DATA", "false").strip().lower()
    return val in ("1", "true", "yes")


def run() -> int:
    """Run the environment doctor. Returns exit code."""
    root = _repo_root()
    venv = _venv_path()
    exe = sys.executable
    version = sys.version.split()[0]
    commit = _git_commit()
    dirty = _git_dirty()
    data_root = os.environ.get("DEEPBOOK_DATA_ROOT", str(root / "data"))
    artifact_root = os.environ.get("DEEPBOOK_ARTIFACT_ROOT", str(root / "artifacts"))
    paid = _paid_data_allowed()
    max_paid = os.environ.get("DEEPBOOK_MAX_PAID_REQUEST_USD", "0")

    try:
        import deepbook  # noqa: F811

        pkg_version = deepbook.__version__
    except ImportError:
        pkg_version = "unknown"

    lines = [
        f"DeepBook version: {pkg_version}",
        f"Python version: {version}",
        f"Python executable: {exe}",
        f"Repository root: {root}",
        f"Virtual environment: {venv or 'none'}",
        f"Git commit: {commit or 'unknown'}",
        f"Working tree clean: {dirty if dirty is not None else 'unknown'}",
        f"Data root: {data_root}",
        f"Artifact root: {artifact_root}",
        f"Paid data authorized: {paid}",
        f"Max paid request (USD): {max_paid}",
    ]
    for line in lines:
        print(line)

    exit_code = 0

    if paid:
        max_usd = float(max_paid) if max_paid else 0.0
        if max_usd <= 0:
            print("ERROR: DEEPBOOK_ALLOW_PAID_DATA=true but DEEPBOOK_MAX_PAID_REQUEST_USD is 0")
            exit_code = 1
        else:
            print(f"WARNING: paid data authorized up to ${max_usd:.2f}")

    if venv is None:
        print("ERROR: Python interpreter is not inside project .venv")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(run())
