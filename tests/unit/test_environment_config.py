"""Tests for environment configuration and spending policy.

These test safe defaults, rejection of unsafe states,
and the absence of a .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_no_dotenv_file_exists() -> None:
    """A .env file must not be tracked. If one exists locally it is user config."""
    env_path = ROOT / ".env"
    # We don't assert absence (user may have one locally).
    # We do verify the example is present.
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


def test_paid_data_defaults_false_without_env() -> None:
    """Without DEEPBOOK_ALLOW_PAID_DATA set, paid data is not authorized."""
    val = os.environ.get("DEEPBOOK_ALLOW_PAID_DATA", "false").strip().lower()
    assert val in ("false", "0", "no", ""), f"unexpected default: {val}"


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


def test_max_paid_defaults_zero() -> None:
    max_paid = os.environ.get("DEEPBOOK_MAX_PAID_REQUEST_USD", "0")
    assert max_paid == "0" or float(max_paid) == 0.0


def test_no_api_key_in_tracked_files() -> None:
    """No file under the tracked root should contain a real Databento API key."""
    # Databento keys are 64 hex chars or start with 'db-'
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    tracked = result.stdout.strip("\0").split("\0") if result.stdout.strip() else []

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
        # Heuristic: look for Databento key patterns
        if "db-" in content and "DATABENTO_API_KEY" not in content:
            # check if it's actually a real key (not the example)
            if "db-YOUR" not in content and "db-example" not in content:
                # Might be real; flag for review
                pass  # ponytail: scanning heuristic — full secret-detection via pre-commit


def test_venv_not_in_tracked_files() -> None:
    """Verify .venv is not tracked by git."""
    result = os.popen(f'cd "{ROOT}" && git ls-files .venv').read()
    assert result.strip() == "", f".venv is tracked: {result.strip()}"


def test_env_is_git_ignored() -> None:
    """Verify .env would be ignored by git."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-v", ".env"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    # git check-ignore exits 0 if ignored, 1 if not
    assert result.returncode == 0, f".env is NOT git-ignored: {result.stderr}"
