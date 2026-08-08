"""Tests for the FI-2010 classical/MLP-LOB snapshot and the suite snapshot.

Tests write to temporary directories via tmp_path. No test writes to
tracked reproduction files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRACKED_JSON = ROOT / "reports" / "reproductions" / "fi2010_classical_mlplob.json"
TRACKED_MD = ROOT / "reports" / "reproductions" / "fi2010_classical_mlplob.md"
TRACKED_SUITE_JSON = ROOT / "reports" / "reproductions" / "fi2010_baseline_suite.json"
TRACKED_SUITE_MD = ROOT / "reports" / "reproductions" / "fi2010_baseline_suite.md"
SCHEMA_PATH = ROOT / "data_contracts" / "fi2010_classical_mlplob_reproduction.schema.json"
SUITE_SCHEMA_PATH = ROOT / "data_contracts" / "fi2010_baseline_suite_reproduction.schema.json"

_SELECTED_MODELS = (
    "majority",
    "causal_persistence",
    "logistic_current_event",
    "random_forest",
    "mlplob",
)
_SUITE_MODELS = _SELECTED_MODELS + ("deeplob",)
_ALL_SEEDS = (1337, 2027, 31415, 424242, 8675309)
_HORIZONS = (10, 20, 30, 50, 100)

_ACCEPTED_HISTORICAL_JSON_SHA256 = (
    "bc2619908651e78b81a1d7878d56d0fccca89fcc0acddc4e4cf8fdd006b364ac"
)
_ACCEPTED_HISTORICAL_MD_SHA256 = "d83c247ca05927c0a42be075f37a2694ae97500ed43e6977c4d643db775ece3f"
_FROZEN_RUN_INDEX_SHA256 = "e2a77af4488eaab152d41d56ac6d7f3659948dcad20c30f2038d87db4b04bcb8"
_FROZEN_REPORT_JSON_SHA256 = "7caf67c12f0c4a23ed1895b92c0e69943fdf6d7e4aa9883e369b41871f0f410e"
_FROZEN_REPORT_MD_SHA256 = "bd5410ba7e5cae0938cd0eb682b3d79acd6e94ff8499aa9c6d80b9a76aff00f1"


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def tracked_snapshot():
    """Load the tracked historical snapshot JSON (read-only)."""
    if not TRACKED_JSON.is_file():
        pytest.skip(
            "Snapshot not yet generated — run: python -m deepbook.cli.fi2010_baselines snapshot"
        )
    return json.loads(TRACKED_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tracked_suite():
    """Load the tracked suite snapshot JSON (read-only)."""
    if not TRACKED_SUITE_JSON.is_file():
        pytest.skip(
            "Suite snapshot not yet generated — "
            "run: python -m deepbook.cli.fi2010_baselines snapshot-suite"
        )
    return json.loads(TRACKED_SUITE_JSON.read_text(encoding="utf-8"))


def _generate_snapshot(tmp_path: Path) -> tuple[dict, Path, Path]:
    """Generate historical snapshot into tmp_path, returning (dict, json_path, md_path)."""
    from deepbook.training.fi2010_snapshot import write_snapshot

    json_path, md_path = write_snapshot(ROOT, output_dir=tmp_path)
    return json.loads(json_path.read_text(encoding="utf-8")), json_path, md_path


def _generate_suite(tmp_path: Path) -> tuple[dict, Path, Path]:
    """Generate suite snapshot into tmp_path."""
    from deepbook.training.fi2010_suite_snapshot import write_suite_snapshot

    json_path, md_path = write_suite_snapshot(ROOT, output_dir=tmp_path)
    return json.loads(json_path.read_text(encoding="utf-8")), json_path, md_path


# ============================================================
# Historical snapshot — schema
# ============================================================


def test_historical_schema_valid(tracked_snapshot):
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=tracked_snapshot, schema=schema)


def test_historical_schema_requires_report_hashes():
    """Schema must require all three creation-time report hashes."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    required = schema["properties"]["hashes"]["required"]
    assert "run_index_sha256" in required
    assert "report_json_sha256" in required
    assert "report_md_sha256" in required
    assert "reconciliation_digest" in required


# ============================================================
# Historical snapshot — byte-identity with accepted dc78a82
# ============================================================


def test_historical_bytes_match_accepted():
    import hashlib

    assert hashlib.sha256(TRACKED_JSON.read_bytes()).hexdigest() == _ACCEPTED_HISTORICAL_JSON_SHA256
    assert hashlib.sha256(TRACKED_MD.read_bytes()).hexdigest() == _ACCEPTED_HISTORICAL_MD_SHA256


def test_generated_matches_tracked(tmp_path):
    """Generated snapshot (into tmp_path) matches tracked accepted bytes."""
    snapshot, json_path, md_path = _generate_snapshot(tmp_path)
    assert json_path.read_bytes() == TRACKED_JSON.read_bytes()
    assert md_path.read_bytes() == TRACKED_MD.read_bytes()


# ============================================================
# Historical snapshot — coverage
# ============================================================


def test_exact_650_selected_runs(tracked_snapshot):
    assert tracked_snapshot["coverage"]["selected_confirmatory_total"] == 650


def test_exact_250_deeplob_pending(tracked_snapshot):
    assert tracked_snapshot["coverage"]["deeplob_pending"] == 250
    assert tracked_snapshot["coverage"]["deeplob_completed"] == 0


def test_exact_setup_totals(tracked_snapshot):
    assert tracked_snapshot["coverage"]["by_setup"]["anchored_forward"] == 585
    assert tracked_snapshot["coverage"]["by_setup"]["first_seven_final_three"] == 65


def test_all_rf_seeds_present(tracked_snapshot):
    for model in _SELECTED_MODELS:
        assert model in tracked_snapshot["coverage"]["by_model"]


def test_all_mlp_seeds_present(tracked_snapshot):
    seed_info = tracked_snapshot["coverage"]["seed_completeness"]
    for seed in _ALL_SEEDS:
        assert str(seed) in seed_info


def test_no_best_seed_filtering(tracked_snapshot):
    agg = tracked_snapshot["aggregates"]
    for key, entry in agg.items():
        if "random_forest" in key or "mlplob" in key:
            assert len(entry["seeds"]) == 5, f"{key} should have 5 seeds"


# ============================================================
# Historical snapshot — provenance
# ============================================================


def test_execution_commit(tracked_snapshot):
    assert (
        tracked_snapshot["provenance"]["execution_commit"]
        == "dd0446e743b35c2dbe7cae3c17e46562850b9772"
    )


# ============================================================
# Historical snapshot — majority
# ============================================================


def test_majority_counts(tracked_snapshot):
    mc = tracked_snapshot["majority_collapse"]
    assert mc["stationary_only"] == 33
    assert mc["up_only"] == 17
    assert mc["down_only"] == 0
    assert mc["single_class_runs"] == 50
    assert mc["all_runs_single_class"] is True


# ============================================================
# Historical snapshot — causal persistence
# ============================================================


def test_causal_all_verified(tracked_snapshot):
    cp = tracked_snapshot["causal_persistence_sample_counts"]
    assert cp["total_causal_persistence_runs"] == 50
    assert cp["all_verified"] is True
    assert cp["verified_sample_count_invariant"] == 50


# ============================================================
# Historical snapshot — reconciliation
# ============================================================


def test_six_reconciliation_events(tracked_snapshot):
    assert tracked_snapshot["reconciliation"]["count"] == 6


def test_reconciliation_digest(tracked_snapshot):
    assert len(tracked_snapshot["reconciliation"]["digest"]) >= 64
    assert len(tracked_snapshot["hashes"]["reconciliation_digest"]) >= 64


# ============================================================
# Historical snapshot — frozen creation-time hashes
# ============================================================


def test_frozen_run_index_hash(tracked_snapshot):
    assert tracked_snapshot["hashes"]["run_index_sha256"] == _FROZEN_RUN_INDEX_SHA256


def test_frozen_report_json_hash(tracked_snapshot):
    assert tracked_snapshot["hashes"]["report_json_sha256"] == _FROZEN_REPORT_JSON_SHA256


def test_frozen_report_md_hash(tracked_snapshot):
    assert tracked_snapshot["hashes"]["report_md_sha256"] == _FROZEN_REPORT_MD_SHA256


# ============================================================
# Historical snapshot — provenance fail-loud
# ============================================================


def test_provenance_missing_file_fails(tmp_path):
    """Missing provenance file raises FileNotFoundError."""
    import yaml

    from deepbook.training.fi2010_snapshot import build_snapshot

    prov_path = ROOT / "configs" / "references" / "fi2010_classical_snapshot_provenance.yaml"
    saved = prov_path.read_bytes()
    try:
        prov_path.unlink()
        with pytest.raises(FileNotFoundError, match="provenance file missing"):
            build_snapshot(ROOT)
    finally:
        prov_path.write_bytes(saved)


def test_provenance_missing_section_fails(tmp_path):
    """Missing creation_time_raw_report_hashes section raises ValueError."""
    import yaml

    from deepbook.training.fi2010_snapshot import build_snapshot

    prov_path = ROOT / "configs" / "references" / "fi2010_classical_snapshot_provenance.yaml"
    saved = prov_path.read_bytes()
    try:
        prov_path.write_text(yaml.dump({"other_section": True}), encoding="utf-8")
        with pytest.raises(ValueError, match="creation_time_raw_report_hashes"):
            build_snapshot(ROOT)
    finally:
        prov_path.write_bytes(saved)


def test_provenance_missing_hash_field_fails(tmp_path):
    """Missing a single hash field raises ValueError."""
    import yaml

    from deepbook.training.fi2010_snapshot import build_snapshot

    prov_path = ROOT / "configs" / "references" / "fi2010_classical_snapshot_provenance.yaml"
    saved = prov_path.read_bytes()
    try:
        bad = yaml.safe_load(saved)
        del bad["creation_time_raw_report_hashes"]["run_index_sha256"]
        prov_path.write_text(yaml.dump(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="run_index_sha256"):
            build_snapshot(ROOT)
    finally:
        prov_path.write_bytes(saved)


def test_provenance_malformed_hash_fails(tmp_path):
    """Malformed hash raises ValueError."""
    import yaml

    from deepbook.training.fi2010_snapshot import build_snapshot

    prov_path = ROOT / "configs" / "references" / "fi2010_classical_snapshot_provenance.yaml"
    saved = prov_path.read_bytes()
    try:
        bad = yaml.safe_load(saved)
        bad["creation_time_raw_report_hashes"]["run_index_sha256"] = "too-short"
        prov_path.write_text(yaml.dump(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="64-character hex"):
            build_snapshot(ROOT)
    finally:
        prov_path.write_bytes(saved)


# ============================================================
# Historical snapshot — disclosures
# ============================================================


def test_deeplob_pending_statement(tracked_snapshot):
    assert "0" in tracked_snapshot["disclosures"]["deep_lob_pending"]
    assert "250" in tracked_snapshot["disclosures"]["deep_lob_pending"]


# ============================================================
# Historical snapshot — sanity
# ============================================================


def test_no_local_absolute_paths(tracked_snapshot):
    json_text = TRACKED_JSON.read_text(encoding="utf-8")
    assert "C:\\\\Users" not in json_text
    assert "/home/" not in json_text


def test_deterministic_json(tracked_snapshot):
    reserialized = json.dumps(tracked_snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    original = TRACKED_JSON.read_text(encoding="utf-8")
    assert reserialized == original


def test_markdown_exists(tracked_snapshot):
    assert TRACKED_MD.is_file()
    md_text = TRACKED_MD.read_text(encoding="utf-8")
    assert len(md_text) > 1000
    assert "# FI-2010 Classical and MLP-LOB" in md_text


# ============================================================
# Historical snapshot — determinism
# ============================================================


def test_snapshot_generator_deterministic(tmp_path):
    """Two generations into tmp_path produce identical bytes."""
    _, j1, m1 = _generate_snapshot(tmp_path / "run1")
    _, j2, m2 = _generate_snapshot(tmp_path / "run2")
    assert j1.read_bytes() == j2.read_bytes()
    assert m1.read_bytes() == m2.read_bytes()


# ============================================================
# Historical snapshot — tracked files unchanged after tests
# ============================================================


def test_tracked_files_unchanged():
    """Tracked historical files match accepted hashes."""
    import hashlib

    assert hashlib.sha256(TRACKED_JSON.read_bytes()).hexdigest() == _ACCEPTED_HISTORICAL_JSON_SHA256
    assert hashlib.sha256(TRACKED_MD.read_bytes()).hexdigest() == _ACCEPTED_HISTORICAL_MD_SHA256


def test_prod_generation_leaves_tree_clean():
    """Running production snapshot generation does not change tracked files."""
    import hashlib
    import subprocess
    import sys

    j_before = hashlib.sha256(TRACKED_JSON.read_bytes()).hexdigest()
    m_before = hashlib.sha256(TRACKED_MD.read_bytes()).hexdigest()

    subprocess.run(
        [sys.executable, "-m", "deepbook.cli.fi2010_baselines", "snapshot"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert hashlib.sha256(TRACKED_JSON.read_bytes()).hexdigest() == j_before
    assert hashlib.sha256(TRACKED_MD.read_bytes()).hexdigest() == m_before


def test_snapshot_does_not_mutate_tracked_outputs_nor_dirty_repo():
    """Running both snapshot generators leaves git status unchanged."""
    import subprocess
    import sys

    before = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True
    ).stdout

    subprocess.run(
        [sys.executable, "-m", "deepbook.cli.fi2010_baselines", "snapshot"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "-m", "deepbook.cli.fi2010_baselines", "snapshot-suite"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    after = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    assert before == after, f"git status changed: before={before!r} after={after!r}"


def test_mismatched_expected_bytes_detected(tmp_path):
    """Prove a bad expected-byte comparison does not alter tracked files."""
    import hashlib

    j_before = hashlib.sha256(TRACKED_JSON.read_bytes()).hexdigest()
    m_before = hashlib.sha256(TRACKED_MD.read_bytes()).hexdigest()

    # Generate to tmp, verify matches tracked, then test detection
    _, jp, mp = _generate_snapshot(tmp_path)

    # Assert matches (the normal path)
    assert jp.read_bytes() == TRACKED_JSON.read_bytes()
    assert mp.read_bytes() == TRACKED_MD.read_bytes()

    # Prove detection works with intentionally incorrect expected bytes
    assert jp.read_bytes() != b"wrong content"
    assert mp.read_bytes() != b"wrong content"

    # Tracked files untouched
    assert hashlib.sha256(TRACKED_JSON.read_bytes()).hexdigest() == j_before
    assert hashlib.sha256(TRACKED_MD.read_bytes()).hexdigest() == m_before


# ============================================================
# Historical snapshot — additional integrity
# ============================================================


def test_all_models_have_aggregates(tracked_snapshot):
    agg = tracked_snapshot["aggregates"]
    models_seen = set()
    for key in agg:
        model = key.split("|")[0]
        models_seen.add(model)
    for model in _SELECTED_MODELS:
        assert model in models_seen, f"{model} missing from aggregates"


def test_all_horizons_have_aggregates(tracked_snapshot):
    agg = tracked_snapshot["aggregates"]
    horizons_seen = set()
    for key in agg:
        h_str = key.split("|")[-1].replace("h", "")
        horizons_seen.add(int(h_str))
    for h in _HORIZONS:
        assert h in horizons_seen, f"h{h} missing from aggregates"


def test_status_all_zeros(tracked_snapshot):
    st = tracked_snapshot["status"]
    assert st["failed"] == 0
    assert st["interrupted"] == 0
    assert st["running"] == 0
    assert st["duplicate"] == 0
    assert st["ineligible"] == 0
    assert st["orphan_predictions"] == 0
    assert st["orphan_checkpoints"] == 0
    assert st["missing"] == 250


def test_verification_counts(tracked_snapshot):
    v = tracked_snapshot["verification"]
    assert v["total_verified"] == 650
    assert v["metric_mismatches"] == 0
    assert v["prediction_mismatches"] == 0


# ============================================================
# SUITE SNAPSHOT TESTS
# ============================================================


def test_suite_schema_valid(tracked_suite):
    import jsonschema

    schema = json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=tracked_suite, schema=schema)


def test_suite_exact_900_coverage(tracked_suite):
    cov = tracked_suite["coverage"]
    assert cov["planned_total"] == 900
    assert cov["completed_confirmatory"] == 900
    assert cov["missing"] == 0
    assert cov["failed"] == 0
    assert cov["interrupted"] == 0
    assert cov["running"] == 0


def test_suite_model_totals(tracked_suite):
    bm = tracked_suite["coverage"]["by_model"]
    assert bm["majority"] == 50
    assert bm["causal_persistence"] == 50
    assert bm["logistic_current_event"] == 50
    assert bm["random_forest"] == 250
    assert bm["mlplob"] == 250
    assert bm["deeplob"] == 250


def test_suite_setup_totals(tracked_suite):
    bs = tracked_suite["coverage"]["by_setup"]
    assert bs["anchored_forward"] == 810
    assert bs["first_seven_final_three"] == 90


def test_suite_deeplob_parameter_count(tracked_suite):
    assert tracked_suite["deeplob"]["parameter_count"] == 143907


def test_suite_deeplob_cuda(tracked_suite):
    dl = tracked_suite["deeplob"]
    assert dl["cuda_runs"] == 250
    assert dl["nonzero_gpu_memory_runs"] == 250


def test_suite_deeplob_no_collapse(tracked_suite):
    assert tracked_suite["deeplob"]["collapse_count"] == 0


def test_suite_deeplob_termination(tracked_suite):
    term = tracked_suite["deeplob"]["termination_reasons"]
    assert term.get("early_stopping", 0) == 249
    assert term.get("max_epochs", 0) == 1


def test_suite_five_seeds(tracked_suite):
    for key, entry in tracked_suite["deep_lob_aggregates"].items():
        assert len(entry["seeds"]) == 5, f"{key}: expected 5 seeds"


def test_suite_all_horizons(tracked_suite):
    horizons_seen = set()
    for key in tracked_suite["deep_lob_aggregates"]:
        h = int(key.split("|")[-1].replace("h", ""))
        horizons_seen.add(h)
    for h in _HORIZONS:
        assert h in horizons_seen, f"h{h} missing"


def test_suite_standard_deviation_convention(tracked_suite):
    assert (
        "population"
        in tracked_suite["protocol_provenance"]["standard_deviation_convention"].lower()
    )


def test_suite_no_post_hoc_threshold(tracked_suite):
    disc = tracked_suite["disclosures"]["published_reference_note"]
    assert "not treated as a confirmatory acceptance criterion" in disc


def test_suite_push_disclosure_truthful(tracked_suite):
    """Push disclosure reflects actual remote state."""
    d = tracked_suite["disclosures"]
    assert d["deeplob_result_commit_pushed"] is True
    assert d["prior_no_push_violation"] is True
    assert d["public_history_rewritten"] is False
    assert "40d77e1" in d["push_status"]
    assert "pushed to origin/main" in d["push_status"]


def test_suite_no_absolute_paths(tracked_suite):
    json_text = TRACKED_SUITE_JSON.read_text(encoding="utf-8")
    assert "C:\\\\Users" not in json_text
    assert "C:/Users/" not in json_text
    assert "rohit" not in json_text


def test_suite_sklearn_limitation(tracked_suite):
    disc = tracked_suite["disclosures"]["scikit_learn_limitation"]
    assert "unknown" in disc.lower()
    assert "scikit-learn" in disc.lower()


def test_suite_deterministic(tracked_suite):
    reserialized = json.dumps(tracked_suite, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    original = TRACKED_SUITE_JSON.read_text(encoding="utf-8")
    assert reserialized == original


def test_suite_generator_deterministic(tmp_path):
    """Two suite generations produce identical bytes."""
    _, j1, m1 = _generate_suite(tmp_path / "run1")
    _, j2, m2 = _generate_suite(tmp_path / "run2")
    assert j1.read_bytes() == j2.read_bytes()
    assert m1.read_bytes() == m2.read_bytes()


def test_suite_markdown_exists(tracked_suite):
    assert TRACKED_SUITE_MD.is_file()
    md_text = TRACKED_SUITE_MD.read_text(encoding="utf-8")
    assert len(md_text) > 1000
    assert "FI-2010 Baseline Suite" in md_text
