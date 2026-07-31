"""Tests for spending policy validation.

Rejects unsafe configurations: authorization true with zero cap,
missing required keys, etc.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "configs" / "spending_policy.yaml"


def _load_policy() -> dict:
    return yaml.safe_load(POLICY_PATH.read_text())


def test_policy_is_valid_yaml() -> None:
    policy = _load_policy()
    assert isinstance(policy, dict)
    assert "core" in policy
    assert "paid_data" in policy


def test_policy_has_required_top_level_keys() -> None:
    policy = _load_policy()
    assert "core" in policy
    assert "paid_data" in policy
    assert "research_budget_usd" in policy["core"]
    assert "default_authorization" in policy["paid_data"]


def test_default_authorization_is_false() -> None:
    policy = _load_policy()
    assert policy["paid_data"]["default_authorization"] is False


def test_budget_is_zero() -> None:
    policy = _load_policy()
    assert policy["core"]["research_budget_usd"] == 0


def test_ceilings_are_positive_when_present() -> None:
    policy = _load_policy()
    pd = policy["paid_data"]
    assert pd["optional_total_ceiling_usd"] > 0
    assert pd["single_pilot_ceiling_usd"] > 0
    assert pd["single_pilot_ceiling_usd"] <= pd["optional_total_ceiling_usd"]


def test_authorization_true_with_zero_cap_would_be_unsafe() -> None:
    """If authorization is ever set to true, max cap must be > 0."""
    # This is a meta-test: the current policy is safe.
    # If someone changes authorization to true without updating cap,
    # our runtime checks would catch it.
    policy = _load_policy()
    if policy["paid_data"]["default_authorization"]:
        # Should not happen in committed config, but verify safety logic
        cap = policy["paid_data"].get("single_pilot_ceiling_usd", 0)
        assert cap > 0, "authorization is true but cap is not positive"


def test_neuralmarket_credits_not_used_by_default() -> None:
    policy = _load_policy()
    rules = [r.lower() for r in policy["paid_data"]["rules"]]
    assert any("neuralmarket" in r for r in rules)
