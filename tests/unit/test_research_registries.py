"""Tests for hypothesis and ablation registries.

Validates ID uniqueness, allowed statuses, and basic structure using the
real registry validator from src/deepbook/registry.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deepbook.registry import RegistryError, validate_ablations, validate_hypotheses

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_hypotheses_have_unique_ids() -> None:
    data = _load_yaml("hypotheses.yaml")
    validate_hypotheses(data)  # raises RegistryError on duplicate IDs


def test_hypotheses_all_start_planned() -> None:
    data = _load_yaml("hypotheses.yaml")
    for h in data["hypotheses"]:
        assert h["status"] == "planned", f"{h['id']}: expected 'planned', got '{h['status']}'"


def test_every_hypothesis_has_falsification() -> None:
    data = _load_yaml("hypotheses.yaml")
    validate_hypotheses(data)  # validates falsification field


def test_every_hypothesis_has_controls() -> None:
    data = _load_yaml("hypotheses.yaml")
    validate_hypotheses(data)  # validates required_controls


def test_ablations_have_unique_ids() -> None:
    data = _load_yaml("ablations.yaml")
    validate_ablations(data)  # raises RegistryError on duplicate IDs


def test_ablations_all_planned() -> None:
    data = _load_yaml("ablations.yaml")
    for a in data["ablations"]:
        assert a["status"] == "planned", f"{a['id']}: expected 'planned', got '{a['status']}'"


def test_synthetic_duplicate_id_rejected() -> None:
    """Two entries with the same ID must be rejected by the validator."""
    data = _load_yaml("hypotheses.yaml")
    # Force a duplicate
    data["hypotheses"][0]["id"] = data["hypotheses"][1]["id"]
    with pytest.raises(RegistryError, match="duplicate"):
        validate_hypotheses(data)


def test_synthetic_malformed_status_rejected() -> None:
    """An invalid status string must be rejected by the validator."""
    data = _load_yaml("hypotheses.yaml")
    data["hypotheses"][0]["status"] = "not_a_real_status"
    with pytest.raises(RegistryError, match="status.*not in"):
        validate_hypotheses(data)


def test_synthetic_missing_required_field_rejected() -> None:
    """A missing required field must be rejected by the validator."""
    data = _load_yaml("hypotheses.yaml")
    del data["hypotheses"][0]["falsification"]
    with pytest.raises(RegistryError, match="missing required field"):
        validate_hypotheses(data)


def test_no_phase_terminology_in_ids() -> None:
    """IDs must not contain phase/step/milestone implementation-stage terminology."""
    data_h = _load_yaml("hypotheses.yaml")
    data_a = _load_yaml("ablations.yaml")
    all_ids = [h["id"] for h in data_h["hypotheses"]] + [a["id"] for a in data_a["ablations"]]
    banned = {"phase", "step", "milestone"}
    for id_val in all_ids:
        lower = id_val.lower()
        for b in banned:
            assert b not in lower, f"ID '{id_val}' contains banned term '{b}'"
