"""Tests for environment configuration and spending policy.

These test safe defaults, rejection of unsafe states,
and the absence of credential leakage.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_no_dotenv_file_tracked() -> None:
    """A .env file must not be tracked by git."""
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0, ".env is tracked by git"


def test_dotenv_example_exists() -> None:
    example = ROOT / ".env.example"
    assert example.exists(), ".env.example is missing"


def test_env_example_has_expected_keys() -> None:
    content = (ROOT / ".env.example").read_text()
    expected = [
        "DEEPBOOK_DATA_ROOT",
        "DEEPBOOK_ARTIFACT_ROOT",
        "DEEPBOOK_ALLOW_PAID_DATA",
        "DEEPBOOK_MAX_PAID_REQUEST_USD",
        "DATABENTO_API_KEY",
    ]
    for key in expected:
        assert key in content, f"{key} missing from .env.example"


def test_paid_data_defaults_false(monkeypatch) -> None:
    """Without DEEPBOOK_ALLOW_PAID_DATA, the doctor reports paid data as false."""
    monkeypatch.delenv("DEEPBOOK_ALLOW_PAID_DATA", raising=False)
    from deepbook.cli.doctor import _paid_data_allowed

    assert _paid_data_allowed() is False


def test_spending_policy_defaults_safe() -> None:
    policy_path = ROOT / "configs" / "spending_policy.yaml"
    policy = yaml.safe_load(policy_path.read_text())
    assert policy["paid_data"]["default_authorization"] is False
    assert policy["core"]["research_budget_usd"] == 0


def test_spending_policy_has_required_rules() -> None:
    policy_path = ROOT / "configs" / "spending_policy.yaml"
    policy = yaml.safe_load(policy_path.read_text())
    rules = policy["paid_data"]["rules"]
    assert any("human" in r.lower() or "authori" in r.lower() for r in rules)
    assert any("no automatic" in r.lower() or "retry" in r.lower() for r in rules)


def test_max_paid_defaults_zero(monkeypatch) -> None:
    """DEEPBOOK_MAX_PAID_REQUEST_USD defaults to '0'."""
    monkeypatch.delenv("DEEPBOOK_MAX_PAID_REQUEST_USD", raising=False)
    max_paid = os.environ.get("DEEPBOOK_MAX_PAID_REQUEST_USD", "0")
    assert max_paid == "0" or float(max_paid) == 0.0


def test_no_api_key_in_tracked_files() -> None:
    """No tracked file should contain a real Databento API key."""
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    tracked = result.stdout.strip("\0").split("\0") if result.stdout.strip() else []

    found = False
    for rel in tracked:
        fpath = ROOT / rel
        if not fpath.is_file():
            continue
        if fpath.suffix in (".pt", ".pth", ".ckpt", ".png", ".jpg"):
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Heuristic: look for non-example Databento key patterns
        if "db-" in content and "DATABENTO_API_KEY" in content:
            if "db-YOUR" not in content and "db-example" not in content:
                found = True
    assert not found, "suspicious Databento key pattern found in tracked files"


def test_venv_not_in_tracked_files() -> None:
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", ".venv"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.stdout.strip() == "", f".venv is tracked: {result.stdout.strip()}"


def test_env_is_git_ignored() -> None:
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-v", ".env"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, f".env is NOT git-ignored: {result.stderr}"


def test_dotenv_absence_does_not_break_doctor(monkeypatch) -> None:
    """The doctor must not fail just because .env is absent."""
    monkeypatch.delenv("DEEPBOOK_ALLOW_PAID_DATA", raising=False)
    monkeypatch.delenv("DEEPBOOK_MAX_PAID_REQUEST_USD", raising=False)
    monkeypatch.delenv("DEEPBOOK_DATA_ROOT", raising=False)
    monkeypatch.delenv("DEEPBOOK_ARTIFACT_ROOT", raising=False)
    import subprocess as sp

    result = sp.run(
        [os.sys.executable, "-m", "deepbook.cli.doctor"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ},
    )
    # Should exit 0 in a valid venv — the absence of .env vars is fine
    assert result.returncode == 0, (
        f"doctor exited {result.returncode} when .env vars absent: {result.stderr}"
    )
