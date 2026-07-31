"""Environment diagnostic command for DeepBook.

Usage:
    python -m deepbook.cli.doctor
"""

from __future__ import annotations

import os
import platform
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
    """Return True when the working tree has uncommitted changes.

    Covers tracked modified and untracked non-ignored files.
    ``None`` means the Git query failed.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=_repo_root(),
            timeout=5,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() != ""
    except Exception:
        return None


def _is_project_venv(
    root: Path | None = None,
    executable: str | None = None,
    system: str | None = None,
) -> bool | None:
    """Check whether the current interpreter is the repository-local .venv."""
    try:
        repository = (root or _repo_root()).resolve()
    except (OSError, RuntimeError):
        return None
    if not repository.exists():
        return None

    expected_venv = repository / ".venv"
    if not expected_venv.exists():
        return False

    exe = Path(executable or sys.executable).resolve()
    current_system = system or platform.system()
    interpreter = "Scripts/python.exe" if current_system == "Windows" else "bin/python"
    return exe == (expected_venv / interpreter).resolve()


def _paid_data_allowed() -> bool:
    val = os.environ.get("DEEPBOOK_ALLOW_PAID_DATA", "false").strip().lower()
    return val in ("1", "true", "yes")


def run() -> int:
    """Run the environment doctor. Returns exit code."""
    root = _repo_root()
    is_project_env = _is_project_venv()
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

    env_status = (
        "active" if is_project_env is True else "inactive" if is_project_env is False else "unknown"
    )

    lines = [
        f"DeepBook version: {pkg_version}",
        f"Python version: {version}",
        f"Python executable: {exe}",
        f"Repository root: {root}",
        f"Project environment: {env_status}",
        f"Git commit: {commit or 'unknown'}",
        f"Working tree: {'dirty' if dirty is True else 'clean' if dirty is False else 'unknown'}",
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

    if is_project_env is not True:
        print("ERROR: Python interpreter is not inside the project-local .venv")
        print(f"       Expected: {root / '.venv'}")
        print(f"       Actual: {exe}")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(run())
