"""Focused result-packaging tests for the TransLOB reproduction snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRACKED_JSON = ROOT / "reports" / "reproductions" / "fi2010_translob.json"
TRACKED_MD = ROOT / "reports" / "reproductions" / "fi2010_translob.md"
SCHEMA_PATH = ROOT / "data_contracts" / "fi2010_translob_reproduction.schema.json"
RUNS_DIR = ROOT / "artifacts" / "fi2010" / "baselines" / "runs"

_ACCEPTED_HISTORICAL_JSON_SHA256 = (
    "bc2619908651e78b81a1d7878d56d0fccca89fcc0acddc4e4cf8fdd006b364ac"
)
_ACCEPTED_HISTORICAL_MD_SHA256 = "d83c247ca05927c0a42be075f37a2694ae97500ed43e6977c4d643db775ece3f"
_ACCEPTED_SUITE_JSON_SHA256 = "11ce9b3e0f9f9120be8b78a4e6c1a24e0fcffe0d839a9a692aef0cc1e63f605a"
_ACCEPTED_SUITE_MD_SHA256 = "f2c5bc229bb41f596d30e54ac9ca35a4fc8370a0bcddcd820d00f853b22501af"
_EXECUTION_COMMIT = "0e2209cc2190fac8f3370ac8a88019131fe3dfba"
_HORIZONS = (10, 20, 30, 50, 100)
_SEEDS = (1337, 2027, 31415, 424242, 8675309)


@pytest.fixture(scope="module")
def tracked_snapshot():
    if not TRACKED_JSON.is_file():
        pytest.skip("TransLOB snapshot not generated")
    return json.loads(TRACKED_JSON.read_text(encoding="utf-8"))


def _load_manifests():
    manifests = {}
    for fpath in sorted(RUNS_DIR.glob("translob-*.json")):
        m = json.loads(fpath.read_text(encoding="utf-8"))
        manifests[m["run_id"]] = m
    return manifests


# ============================================================
# Coverage
# ============================================================


def test_translob_exactly_250_complete(tracked_snapshot):
    assert tracked_snapshot["coverage"]["completed_confirmatory"] == 250
    assert tracked_snapshot["coverage"]["planned_total"] == 250


def test_setup_totals(tracked_snapshot):
    bs = tracked_snapshot["coverage"]["by_setup"]
    assert bs["anchored_forward"] == 225
    assert bs["first_seven_final_three"] == 25


def test_five_seeds_and_five_horizons(tracked_snapshot):
    assert tracked_snapshot["coverage"]["seeds"] == list(_SEEDS)
    assert tracked_snapshot["coverage"]["horizons"] == list(_HORIZONS)
    for seed in map(str, _SEEDS):
        assert tracked_snapshot["coverage"]["seed_completeness"][seed]["complete"] is True
    for h in map(str, _HORIZONS):
        assert tracked_snapshot["coverage"]["horizon_completeness"][h]["complete"] is True


def test_nine_setup1_folds_and_setup2_group(tracked_snapshot):
    assert tracked_snapshot["coverage"]["folds"] == list(range(1, 10))


def test_code_commit_uniform_0e2209c():
    manifests = _load_manifests()
    assert len(manifests) == 250
    for rid, m in manifests.items():
        assert m["status"] == "completed", rid
        assert m["code_commit"] == _EXECUTION_COMMIT, rid
        assert m["model"] == "translob", rid


def test_cuda_250_of_250_and_gpu_memory():
    manifests = _load_manifests()
    for rid, m in manifests.items():
        assert m["device"] == "cuda", rid
        assert int(m.get("peak_gpu_memory_bytes", 0)) > 0, rid


def test_parameter_count_101895(tracked_snapshot):
    assert tracked_snapshot["protocol_provenance"]["parameter_count"] == 101895


def test_no_test_selection_leakage():
    manifests = _load_manifests()
    for rid, m in manifests.items():
        assert m.get("test_set_used_for_selection") is False, rid
        assert m.get("labels_regenerated") is False, rid


def test_no_best_seed_filtering(tracked_snapshot):
    for seed in map(str, _SEEDS):
        assert tracked_snapshot["coverage"]["seed_completeness"][seed]["completed"] == 50


# ============================================================
# Disclosures
# ============================================================


def test_wo_ambiguity_disclosure(tracked_snapshot):
    d = tracked_snapshot["disclosures"]["wo_source_conflict"]
    assert "AMBIGUOUS_SOURCE_CONFLICT" in d
    assert "W^O" in d or "W_O" in d


def test_l2_ambiguity_disclosure(tracked_snapshot):
    d = tracked_snapshot["disclosures"]["l2_ambiguity"]
    assert "L2" in d
    assert "coefficient" in d.lower()


def test_horizon_30_literature_handling(tracked_snapshot):
    d = tracked_snapshot["disclosures"]["literature_horizon_30"]
    assert "unmatched" in d.lower()


def test_tlob_pending_disclosure(tracked_snapshot):
    assert "unexecuted" in tracked_snapshot["disclosures"]["tlob_pending"]


def test_label_policy(tracked_snapshot):
    assert "official" in tracked_snapshot["disclosures"]["label_policy"].lower()


# ============================================================
# Collapse audit
# ============================================================


def test_collapse_audit_present(tracked_snapshot):
    col = tracked_snapshot["collapse_audit"]
    assert col["total_runs"] == 250
    assert col["single_class_runs"] >= 0


# ============================================================
# Schema and determinism
# ============================================================


def test_schema_valid(tracked_snapshot):
    import jsonschema

    jsonschema.validate(
        instance=tracked_snapshot,
        schema=json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
    )


def test_deterministic_json(tracked_snapshot, tmp_path):
    from deepbook.training.fi2010_translob_snapshot import write_translob_snapshot

    j1, _ = write_translob_snapshot(ROOT, output_dir=tmp_path / "run1")
    j2, _ = write_translob_snapshot(ROOT, output_dir=tmp_path / "run2")
    assert j1.read_bytes() == j2.read_bytes()


def test_deterministic_markdown(tracked_snapshot, tmp_path):
    from deepbook.training.fi2010_translob_snapshot import write_translob_snapshot

    _, m1 = write_translob_snapshot(ROOT, output_dir=tmp_path / "run1")
    _, m2 = write_translob_snapshot(ROOT, output_dir=tmp_path / "run2")
    assert m1.read_bytes() == m2.read_bytes()


def test_deterministic_json_reserialized(tracked_snapshot):
    reserialized = json.dumps(tracked_snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert reserialized == TRACKED_JSON.read_text(encoding="utf-8")


def test_no_absolute_paths(tracked_snapshot):
    json_text = TRACKED_JSON.read_text(encoding="utf-8")
    assert "C:\\Users" not in json_text
    assert "rohit" not in json_text


def test_no_raw_arrays_or_full_manifests(tracked_snapshot):
    json_text = TRACKED_JSON.read_text(encoding="utf-8")
    for forbidden in ("y_true", "y_pred", "probabilities", "checkpoint_state", "optimizer_state"):
        assert forbidden not in json_text


# ============================================================
# Accepted historical hashes unchanged
# ============================================================


def test_accepted_historical_hashes_unchanged():
    assert (
        hashlib.sha256(
            (ROOT / "reports/reproductions/fi2010_classical_mlplob.json").read_bytes()
        ).hexdigest()
        == _ACCEPTED_HISTORICAL_JSON_SHA256
    )
    assert (
        hashlib.sha256(
            (ROOT / "reports/reproductions/fi2010_classical_mlplob.md").read_bytes()
        ).hexdigest()
        == _ACCEPTED_HISTORICAL_MD_SHA256
    )
    assert (
        hashlib.sha256(
            (ROOT / "reports/reproductions/fi2010_baseline_suite.json").read_bytes()
        ).hexdigest()
        == _ACCEPTED_SUITE_JSON_SHA256
    )
    assert (
        hashlib.sha256(
            (ROOT / "reports/reproductions/fi2010_baseline_suite.md").read_bytes()
        ).hexdigest()
        == _ACCEPTED_SUITE_MD_SHA256
    )


# ============================================================
# Run-index expectations
# ============================================================


def test_run_index_1150_completed_250_missing():
    index = json.loads(
        (ROOT / "artifacts/fi2010/baselines/run_index.json").read_text(encoding="utf-8")
    )
    totals = index.get("planned_totals", {})
    assert len(index.get("completed_confirmatory", [])) == 1150
    assert totals.get("planned_by_model", {}).get("translob") == 250
    assert totals.get("planned_by_model", {}).get("tlob") == 250
    assert totals.get("manifests_by_model", {}).get("translob") == 250
    assert "tlob" not in totals.get("manifests_by_model", {})
    assert totals.get("manifests_by_setup", {}).get("anchored_forward") == 1035
    assert totals.get("manifests_by_setup", {}).get("first_seven_final_three") == 115
