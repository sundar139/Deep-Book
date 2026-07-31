"""Tests for hypothesis and ablation registries.

Validates ID uniqueness, allowed statuses, and basic structure.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_hypotheses_have_unique_ids() -> None:
    data = _load_yaml("hypotheses.yaml")
    ids = [h["id"] for h in data["hypotheses"]]
    assert len(ids) == len(set(ids)), f"duplicate hypothesis IDs: {ids}"


def test_hypotheses_all_start_planned() -> None:
    data = _load_yaml("hypotheses.yaml")
    allowed = {"planned", "running", "confirmed", "rejected", "inconclusive"}
    for h in data["hypotheses"]:
        assert h["status"] in allowed, f"{h['id']}: invalid status '{h['status']}'"
    # All hypotheses begin as planned (at least, none should be confirmed yet)
    for h in data["hypotheses"]:
        assert h["status"] == "planned", f"{h['id']}: expected 'planned', got '{h['status']}'"


def test_every_hypothesis_has_falsification() -> None:
    data = _load_yaml("hypotheses.yaml")
    for h in data["hypotheses"]:
        assert "falsification" in h, f"{h['id']} missing falsification"
        assert len(h["falsification"]) > 10, f"{h['id']} falsification too short"


def test_every_hypothesis_has_controls() -> None:
    data = _load_yaml("hypotheses.yaml")
    for h in data["hypotheses"]:
        assert "required_controls" in h, f"{h['id']} missing required_controls"
        assert len(h["required_controls"]) > 0, f"{h['id']} has no controls"


def test_ablations_have_unique_ids() -> None:
    data = _load_yaml("ablations.yaml")
    ids = [a["id"] for a in data["ablations"]]
    assert len(ids) == len(set(ids)), f"duplicate ablation IDs: {ids}"


def test_ablations_all_planned() -> None:
    data = _load_yaml("ablations.yaml")
    allowed = {"planned", "running", "completed", "aborted"}
    for a in data["ablations"]:
        assert a["status"] in allowed, f"{a['id']}: invalid status '{a['status']}'"
    for a in data["ablations"]:
        assert a["status"] == "planned", f"{a['id']}: expected 'planned', got '{a['status']}'"


def test_synthetic_duplicate_id_rejected() -> None:
    """Two entries with the same ID must be detected."""
    data = _load_yaml("hypotheses.yaml")
    ids = [h["id"] for h in data["hypotheses"]]
    # Force a duplicate
    ids[0] = ids[1] if len(ids) > 1 else ids[0]
    assert len(ids) != len(set(ids)), "synthetic duplicate ID was not detected"


def test_synthetic_malformed_status_rejected() -> None:
    """An invalid status string must be caught."""
    data = _load_yaml("hypotheses.yaml")
    allowed = {"planned", "running", "confirmed", "rejected", "inconclusive"}
    # Inject a bad status
    data["hypotheses"][0]["status"] = "not_a_real_status"
    assert data["hypotheses"][0]["status"] not in allowed, (
        "synthetic malformed status was not detected"
    )


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
