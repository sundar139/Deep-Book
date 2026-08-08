"""Tests for the FI-2010 classical/MLP-LOB snapshot and the suite snapshot.

Tests write to temporary directories via tmp_path. No test writes to
tracked reproduction files. Provenance failure tests use temporary
provenance YAML files — never touch the tracked provenance file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRACKED_JSON = ROOT / "reports" / "reproductions" / "fi2010_classical_mlplob.json"
TRACKED_MD = ROOT / "reports" / "reproductions" / "fi2010_classical_mlplob.md"
TRACKED_SUITE_JSON = ROOT / "reports" / "reproductions" / "fi2010_baseline_suite.json"
TRACKED_SUITE_MD = ROOT / "reports" / "reproductions" / "fi2010_baseline_suite.md"
SCHEMA_PATH = ROOT / "data_contracts" / "fi2010_classical_mlplob_reproduction.schema.json"
SUITE_SCHEMA_PATH = ROOT / "data_contracts" / "fi2010_baseline_suite_reproduction.schema.json"
TRACKED_PROVENANCE = ROOT / "configs" / "references" / "fi2010_classical_snapshot_provenance.yaml"

_SELECTED_MODELS = (
    "majority",
    "causal_persistence",
    "logistic_current_event",
    "random_forest",
    "mlplob",
)
_ALL_SEEDS = (1337, 2027, 31415, 424242, 8675309)
_HORIZONS = (10, 20, 30, 50, 100)

_ACCEPTED_HISTORICAL_JSON_SHA256 = (
    "bc2619908651e78b81a1d7878d56d0fccca89fcc0acddc4e4cf8fdd006b364ac"
)
_ACCEPTED_HISTORICAL_MD_SHA256 = "d83c247ca05927c0a42be075f37a2694ae97500ed43e6977c4d643db775ece3f"
_ACCEPTED_HISTORICAL_SCHEMA_SHA256 = (
    "22a358971a4acc3282cc9bd70e29942a79827d76c5a931ce96882be2496395b5"
)
_FROZEN_RUN_INDEX_SHA256 = "e2a77af4488eaab152d41d56ac6d7f3659948dcad20c30f2038d87db4b04bcb8"
_FROZEN_REPORT_JSON_SHA256 = "7caf67c12f0c4a23ed1895b92c0e69943fdf6d7e4aa9883e369b41871f0f410e"
_FROZEN_REPORT_MD_SHA256 = "bd5410ba7e5cae0938cd0eb682b3d79acd6e94ff8499aa9c6d80b9a76aff00f1"


# ============================================================
# Helpers
# ============================================================


def _valid_provenance() -> dict:
    return {
        "creation_time_raw_report_hashes": {
            "run_index_sha256": _FROZEN_RUN_INDEX_SHA256,
            "report_json_sha256": _FROZEN_REPORT_JSON_SHA256,
            "report_md_sha256": _FROZEN_REPORT_MD_SHA256,
        }
    }


def _write_provenance(tmp_path: Path, prov: dict) -> Path:
    p = tmp_path / "provenance.yaml"
    p.write_text(yaml.dump(prov), encoding="utf-8")
    return p


def _generate_snapshot(tmp_path: Path, prov_path: Path | None = None) -> tuple[dict, Path, Path]:
    from deepbook.training.fi2010_snapshot import write_snapshot

    json_path, md_path = write_snapshot(
        ROOT, output_dir=tmp_path, provenance_path_override=prov_path
    )
    return json.loads(json_path.read_text(encoding="utf-8")), json_path, md_path


def _generate_suite(tmp_path: Path) -> tuple[dict, Path, Path]:
    from deepbook.training.fi2010_suite_snapshot import write_suite_snapshot

    json_path, md_path = write_suite_snapshot(ROOT, output_dir=tmp_path)
    return json.loads(json_path.read_text(encoding="utf-8")), json_path, md_path


# ============================================================
# Tracked file fixtures (read-only)
# ============================================================


@pytest.fixture(scope="module")
def tracked_snapshot():
    if not TRACKED_JSON.is_file():
        pytest.skip("Historical snapshot not generated")
    return json.loads(TRACKED_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tracked_suite():
    if not TRACKED_SUITE_JSON.is_file():
        pytest.skip("Suite snapshot not generated")
    return json.loads(TRACKED_SUITE_JSON.read_text(encoding="utf-8"))


# ============================================================
# Historical byte identity
# ============================================================


def test_historical_bytes_match_accepted():
    import hashlib

    assert hashlib.sha256(TRACKED_JSON.read_bytes()).hexdigest() == _ACCEPTED_HISTORICAL_JSON_SHA256
    assert hashlib.sha256(TRACKED_MD.read_bytes()).hexdigest() == _ACCEPTED_HISTORICAL_MD_SHA256
    assert (
        hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest() == _ACCEPTED_HISTORICAL_SCHEMA_SHA256
    )


def test_generated_matches_tracked(tmp_path):
    prov = _write_provenance(tmp_path, _valid_provenance())
    _, json_path, md_path = _generate_snapshot(tmp_path, prov_path=prov)
    assert json_path.read_bytes() == TRACKED_JSON.read_bytes()
    assert md_path.read_bytes() == TRACKED_MD.read_bytes()


# ============================================================
# Schema
# ============================================================


def test_historical_schema_valid(tracked_snapshot):
    import jsonschema

    jsonschema.validate(
        instance=tracked_snapshot,
        schema=json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
    )


def test_historical_schema_requires_report_hashes():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    required = schema["properties"]["hashes"]["required"]
    for field in ("run_index_sha256", "report_json_sha256", "report_md_sha256"):
        assert field in required


def test_suite_schema_valid(tracked_suite):
    import jsonschema

    jsonschema.validate(
        instance=tracked_suite,
        schema=json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8")),
    )


def test_suite_schema_disclosures_enforced():
    """Suite schema requires all push-status fields with additionalProperties: false."""
    schema = json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8"))
    disc = schema["properties"]["disclosures"]
    assert disc["additionalProperties"] is False
    required = disc["required"]
    for field in (
        "remote_main_commit",
        "deeplob_result_commit",
        "deeplob_result_commit_pushed",
        "historical_snapshot_repair_commit",
        "historical_snapshot_repair_commit_pushed",
        "provenance_hardening_commit",
        "provenance_hardening_commit_pushed",
        "current_finalization_commit_pushed",
        "prior_no_push_violation",
        "public_history_rewritten",
    ):
        assert field in required, f"missing required disclosure: {field}"


def test_suite_schema_rejects_malformed_commit():
    """Schema rejects non-hex commit SHA."""
    import jsonschema

    suite = json.loads(TRACKED_SUITE_JSON.read_text(encoding="utf-8"))
    schema = json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = dict(suite)
    bad["disclosures"]["remote_main_commit"] = "not-a-commit-sha"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_suite_schema_rejects_missing_field():
    """Schema rejects missing required disclosure field."""
    import jsonschema

    suite = json.loads(TRACKED_SUITE_JSON.read_text(encoding="utf-8"))
    schema = json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = dict(suite)
    del bad["disclosures"]["prior_no_push_violation"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_suite_schema_rejects_extra_property():
    """Schema rejects unknown extra disclosure property."""
    import jsonschema

    suite = json.loads(TRACKED_SUITE_JSON.read_text(encoding="utf-8"))
    schema = json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = dict(suite)
    bad["disclosures"]["made_up_field"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


# ============================================================
# Historical coverage
# ============================================================


def test_exact_650_selected_runs(tracked_snapshot):
    assert tracked_snapshot["coverage"]["selected_confirmatory_total"] == 650


def test_exact_setup_totals(tracked_snapshot):
    assert tracked_snapshot["coverage"]["by_setup"]["anchored_forward"] == 585
    assert tracked_snapshot["coverage"]["by_setup"]["first_seven_final_three"] == 65


# ============================================================
# Historical frozen hashes
# ============================================================


def test_frozen_run_index_hash(tracked_snapshot):
    assert tracked_snapshot["hashes"]["run_index_sha256"] == _FROZEN_RUN_INDEX_SHA256


def test_frozen_report_json_hash(tracked_snapshot):
    assert tracked_snapshot["hashes"]["report_json_sha256"] == _FROZEN_REPORT_JSON_SHA256


def test_frozen_report_md_hash(tracked_snapshot):
    assert tracked_snapshot["hashes"]["report_md_sha256"] == _FROZEN_REPORT_MD_SHA256


# ============================================================
# Provenance fail-loud (all using tmp_path)
# ============================================================


def test_provenance_missing_file_fails(tmp_path):
    from deepbook.training.fi2010_snapshot import build_snapshot

    nonexistent = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError, match="provenance file missing"):
        build_snapshot(ROOT, provenance_path_override=nonexistent)


def test_provenance_missing_section_fails(tmp_path):
    from deepbook.training.fi2010_snapshot import build_snapshot

    prov_path = _write_provenance(tmp_path, {"wrong_section": True})
    with pytest.raises(ValueError, match="creation_time_raw_report_hashes"):
        build_snapshot(ROOT, provenance_path_override=prov_path)


def test_provenance_missing_hash_field_fails(tmp_path):
    from deepbook.training.fi2010_snapshot import build_snapshot

    bad = _valid_provenance()
    del bad["creation_time_raw_report_hashes"]["run_index_sha256"]
    prov_path = _write_provenance(tmp_path, bad)
    with pytest.raises(ValueError, match="run_index_sha256"):
        build_snapshot(ROOT, provenance_path_override=prov_path)


def test_provenance_malformed_hash_fails(tmp_path):
    from deepbook.training.fi2010_snapshot import build_snapshot

    bad = _valid_provenance()
    bad["creation_time_raw_report_hashes"]["run_index_sha256"] = "too-short"
    prov_path = _write_provenance(tmp_path, bad)
    with pytest.raises(ValueError, match="64-character hex"):
        build_snapshot(ROOT, provenance_path_override=prov_path)


def test_tracked_provenance_unchanged():
    """Tracked provenance file must not be modified by tests."""
    import hashlib

    p = TRACKED_PROVENANCE.read_bytes()
    assert (
        hashlib.sha256(p).hexdigest()
        == "82241928826187a1ca3ef7a16d2996b856892222b9d7aa3090b443269eebe095"
    )


# ============================================================
# Failure isolation
# ============================================================


def assert_snapshot_bytes_match(actual: Path, expected: bytes) -> None:
    """Assert generated output against an explicit expected byte sequence."""
    actual_bytes = actual.read_bytes()
    if actual_bytes != expected:
        raise AssertionError(f"snapshot bytes differ: {actual}")


def test_mismatched_bytes_detected_without_dirtying_tracked(tmp_path):
    """The real mismatch path fails without dirtying tracked files or config."""
    import hashlib
    import subprocess

    tracked_before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (TRACKED_JSON, TRACKED_MD, SCHEMA_PATH, TRACKED_PROVENANCE)
    }
    status_before = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout

    prov = _write_provenance(tmp_path, _valid_provenance())
    _, jp, mp = _generate_snapshot(tmp_path, prov_path=prov)
    assert_snapshot_bytes_match(jp, TRACKED_JSON.read_bytes())
    assert_snapshot_bytes_match(mp, TRACKED_MD.read_bytes())
    with pytest.raises(AssertionError, match="snapshot bytes differ"):
        assert_snapshot_bytes_match(jp, b"definitely wrong bytes")

    assert (
        status_before
        == subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    )
    assert tracked_before == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (TRACKED_JSON, TRACKED_MD, SCHEMA_PATH, TRACKED_PROVENANCE)
    }


def test_snapshot_does_not_dirty_repo():
    """Running snapshot CLI does not change git status."""
    import hashlib
    import subprocess
    import sys

    j_before = hashlib.sha256(TRACKED_JSON.read_bytes()).hexdigest()
    m_before = hashlib.sha256(TRACKED_MD.read_bytes()).hexdigest()

    for cmd in ("snapshot", "snapshot-suite"):
        subprocess.run(
            [sys.executable, "-m", "deepbook.cli.fi2010_baselines", cmd],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    assert hashlib.sha256(TRACKED_JSON.read_bytes()).hexdigest() == j_before
    assert hashlib.sha256(TRACKED_MD.read_bytes()).hexdigest() == m_before


# ============================================================
# Determinism
# ============================================================


def test_snapshot_generator_deterministic(tmp_path):
    prov = _write_provenance(tmp_path, _valid_provenance())
    _, j1, m1 = _generate_snapshot(tmp_path / "run1", prov_path=prov)
    _, j2, m2 = _generate_snapshot(tmp_path / "run2", prov_path=prov)
    assert j1.read_bytes() == j2.read_bytes()
    assert m1.read_bytes() == m2.read_bytes()


def test_suite_generator_deterministic(tmp_path):
    _, j1, m1 = _generate_suite(tmp_path / "run1")
    _, j2, m2 = _generate_suite(tmp_path / "run2")
    assert j1.read_bytes() == j2.read_bytes()
    assert m1.read_bytes() == m2.read_bytes()


# ============================================================
# Suite coverage
# ============================================================


def test_suite_exact_900_coverage(tracked_suite):
    cov = tracked_suite["coverage"]
    assert cov["planned_total"] == 900
    assert cov["completed_confirmatory"] == 900
    assert cov["missing"] == 0


def test_suite_model_totals(tracked_suite):
    bm = tracked_suite["coverage"]["by_model"]
    assert bm["deeplob"] == 250
    assert bm["mlplob"] == 250
    assert bm["random_forest"] == 250


def test_suite_setup_totals(tracked_suite):
    bs = tracked_suite["coverage"]["by_setup"]
    assert bs["anchored_forward"] == 810
    assert bs["first_seven_final_three"] == 90


def test_suite_deeplob_cuda(tracked_suite):
    dl = tracked_suite["deeplob"]
    assert dl["cuda_runs"] == 250
    assert dl["nonzero_gpu_memory_runs"] == 250
    assert dl["collapse_count"] == 0
    assert dl["parameter_count"] == 143907


# ============================================================
# Suite push disclosure
# ============================================================


def test_suite_push_disclosure_truthful(tracked_suite):
    d = tracked_suite["disclosures"]
    assert d["deeplob_result_commit_pushed"] is True
    assert d["historical_snapshot_repair_commit_pushed"] is True
    assert d["provenance_hardening_commit_pushed"] is True
    assert d["current_finalization_commit_pushed"] is False
    assert d["prior_no_push_violation"] is True
    assert d["public_history_rewritten"] is False
    assert "40d77e1" in d["push_status"]
    assert "52fd936" in d["push_status"]
    assert "00da49d" in d["push_status"]


def _synthetic_git_state(remote="00da49d"):
    return {
        "remote_main_commit": remote,
        "ls_remote_main_commit": remote,
        "contains": {
            "deeplob_result_commit": True,
            "historical_snapshot_repair_commit": True,
            "provenance_hardening_commit": True,
            "current_finalization_commit": False,
        },
    }


def _synthetic_disclosure():
    return {
        "remote_main_commit": "00da49d",
        "deeplob_result_commit_pushed": True,
        "historical_snapshot_repair_commit_pushed": True,
        "provenance_hardening_commit_pushed": True,
        "current_finalization_commit_pushed": False,
    }


def test_declared_git_state_accepts_frozen_disclosure():
    from deepbook.training.fi2010_suite_snapshot import validate_declared_git_state

    validate_declared_git_state(_synthetic_disclosure(), _synthetic_git_state())


@pytest.mark.parametrize(
    ("field", "observed"),
    [
        ("remote_main_commit", _synthetic_git_state(remote="other")),
        (
            "historical_snapshot_repair_commit_pushed",
            {
                **_synthetic_git_state(),
                "contains": {
                    **_synthetic_git_state()["contains"],
                    "historical_snapshot_repair_commit": False,
                },
            },
        ),
        (
            "current_finalization_commit_pushed",
            {
                **_synthetic_git_state(),
                "contains": {
                    **_synthetic_git_state()["contains"],
                    "current_finalization_commit": True,
                },
            },
        ),
    ],
)
def test_declared_git_state_rejects_mismatches(field, observed):
    from deepbook.training.fi2010_suite_snapshot import validate_declared_git_state

    with pytest.raises(ValueError, match=field):
        validate_declared_git_state(_synthetic_disclosure(), observed)


def test_declared_git_state_rejects_stale_tracking_ref():
    from deepbook.training.fi2010_suite_snapshot import validate_declared_git_state

    observed = _synthetic_git_state()
    observed["ls_remote_main_commit"] = "other"
    with pytest.raises(ValueError, match="ls-remote"):
        validate_declared_git_state(_synthetic_disclosure(), observed)


def test_suite_no_absolute_paths(tracked_suite):
    json_text = TRACKED_SUITE_JSON.read_text(encoding="utf-8")
    assert "C:\\\\Users" not in json_text
    assert "rohit" not in json_text


def test_suite_sklearn_limitation(tracked_suite):
    assert "unknown" in tracked_suite["disclosures"]["scikit_learn_limitation"].lower()


# ============================================================
# Integrity
# ============================================================


def test_historical_no_local_paths(tracked_snapshot):
    json_text = TRACKED_JSON.read_text(encoding="utf-8")
    assert "C:\\\\Users" not in json_text


def test_majority_counts(tracked_snapshot):
    mc = tracked_snapshot["majority_collapse"]
    assert mc["stationary_only"] == 33
    assert mc["up_only"] == 17


def test_causal_all_verified(tracked_snapshot):
    cp = tracked_snapshot["causal_persistence_sample_counts"]
    assert cp["all_verified"] is True
    assert cp["total_causal_persistence_runs"] == 50


def test_reconciliation(tracked_snapshot):
    assert tracked_snapshot["reconciliation"]["count"] == 6


def test_deterministic_json_reserialized(tracked_snapshot):
    reserialized = json.dumps(tracked_snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert reserialized == TRACKED_JSON.read_text(encoding="utf-8")


def test_suite_deterministic_json(tracked_suite):
    reserialized = json.dumps(tracked_suite, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert reserialized == TRACKED_SUITE_JSON.read_text(encoding="utf-8")
