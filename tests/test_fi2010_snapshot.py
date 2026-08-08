"""Tests for the FI-2010 classical/MLP-LOB snapshot and the suite snapshot."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_JSON = ROOT / "reports" / "reproductions" / "fi2010_classical_mlplob.json"
SNAPSHOT_MD = ROOT / "reports" / "reproductions" / "fi2010_classical_mlplob.md"
SUITE_JSON = ROOT / "reports" / "reproductions" / "fi2010_baseline_suite.json"
SUITE_MD = ROOT / "reports" / "reproductions" / "fi2010_baseline_suite.md"
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


# ============================================================
# Historical snapshot fixtures
# ============================================================


@pytest.fixture(scope="module")
def snapshot():
    """Load the generated historical snapshot JSON."""
    if not SNAPSHOT_JSON.is_file():
        pytest.skip(
            "Snapshot not yet generated — run: python -m deepbook.cli.fi2010_baselines snapshot"
        )
    return json.loads(SNAPSHOT_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def suite_snapshot():
    """Load the generated suite snapshot JSON."""
    if not SUITE_JSON.is_file():
        pytest.skip(
            "Suite snapshot not yet generated — "
            "run: python -m deepbook.cli.fi2010_baselines snapshot-suite"
        )
    return json.loads(SUITE_JSON.read_text(encoding="utf-8"))


# ============================================================
# Historical snapshot — schema
# ============================================================


def test_historical_schema_valid(snapshot):
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=snapshot, schema=schema)


# ============================================================
# Historical snapshot — coverage
# ============================================================


def test_exact_650_selected_runs(snapshot):
    assert snapshot["coverage"]["selected_confirmatory_total"] == 650


def test_exact_250_deeplob_pending(snapshot):
    assert snapshot["coverage"]["deeplob_pending"] == 250
    assert snapshot["coverage"]["deeplob_completed"] == 0


def test_exact_setup_totals(snapshot):
    assert snapshot["coverage"]["by_setup"]["anchored_forward"] == 585
    assert snapshot["coverage"]["by_setup"]["first_seven_final_three"] == 65


def test_all_rf_seeds_present(snapshot):
    for model in _SELECTED_MODELS:
        assert model in snapshot["coverage"]["by_model"]


def test_all_mlp_seeds_present(snapshot):
    seed_info = snapshot["coverage"]["seed_completeness"]
    for seed in _ALL_SEEDS:
        assert str(seed) in seed_info


def test_no_best_seed_filtering(snapshot):
    agg = snapshot["aggregates"]
    for key, entry in agg.items():
        if "random_forest" in key or "mlplob" in key:
            assert len(entry["seeds"]) == 5, f"{key} should have 5 seeds"


# ============================================================
# Historical snapshot — provenance
# ============================================================


def test_execution_commit(snapshot):
    assert snapshot["provenance"]["execution_commit"] == "dd0446e743b35c2dbe7cae3c17e46562850b9772"


# ============================================================
# Historical snapshot — majority
# ============================================================


def test_majority_explanation_present(snapshot):
    mc = snapshot["majority_collapse"]
    assert len(mc["explanation"]) > 50


def test_majority_counts(snapshot):
    mc = snapshot["majority_collapse"]
    assert mc["stationary_only"] == 33
    assert mc["up_only"] == 17
    assert mc["down_only"] == 0
    assert mc["single_class_runs"] == 50
    assert mc["all_runs_single_class"] is True


# ============================================================
# Historical snapshot — causal persistence
# ============================================================


def test_causal_count_explanation(snapshot):
    cp = snapshot["causal_persistence_sample_counts"]
    assert "segments" in cp["explanation"].lower()
    assert "horizon" in cp["explanation"].lower()


def test_causal_all_verified(snapshot):
    cp = snapshot["causal_persistence_sample_counts"]
    assert cp["total_causal_persistence_runs"] == 50
    assert cp["all_verified"] is True
    assert cp["verified_sample_count_invariant"] == 50


# ============================================================
# Historical snapshot — reconciliation
# ============================================================


def test_six_reconciliation_events(snapshot):
    assert snapshot["reconciliation"]["count"] == 6


def test_reconciliation_digest(snapshot):
    assert len(snapshot["reconciliation"]["digest"]) >= 64
    assert len(snapshot["hashes"]["reconciliation_digest"]) >= 64


# ============================================================
# Historical snapshot — hashes
# ============================================================


def test_report_hashes_present(snapshot):
    h = snapshot["hashes"]
    assert len(h["reconciliation_digest"]) >= 64


def test_reconciliation_digest_present(snapshot):
    assert len(snapshot["reconciliation"]["digest"]) > 0


# ============================================================
# Historical snapshot — disclosures
# ============================================================


def test_pushed_commit_disclosure(snapshot):
    d = snapshot["disclosures"]
    assert "de61432" in d["de61432_and_dd0446e_were_pushed"]
    assert "dd0446e" in d["de61432_and_dd0446e_were_pushed"]


def test_result_commit_unpushed(snapshot):
    assert snapshot["disclosures"]["result_commit_local_only"] is True


def test_deeplob_pending_statement(snapshot):
    assert "0" in snapshot["disclosures"]["deep_lob_pending"]
    assert "250" in snapshot["disclosures"]["deep_lob_pending"]


# ============================================================
# Historical snapshot — sanity
# ============================================================


def test_no_raw_prediction_arrays(snapshot):
    json_text = SNAPSHOT_JSON.read_text(encoding="utf-8")
    assert '"probabilities"' not in json_text


def test_no_full_manifests(snapshot):
    json_text = SNAPSHOT_JSON.read_text(encoding="utf-8")
    assert '"best_checkpoint_sha256"' not in json_text
    assert '"prediction_sha256"' not in json_text


def test_no_checkpoint_content(snapshot):
    json_text = SNAPSHOT_JSON.read_text(encoding="utf-8")
    assert '"checkpoint_path"' not in json_text


def test_no_local_absolute_paths(snapshot):
    """No local absolute paths in historical snapshot."""
    json_text = SNAPSHOT_JSON.read_text(encoding="utf-8")
    assert "C:\\\\Users" not in json_text
    assert "/home/" not in json_text


def test_no_credential_content(snapshot):
    json_text = SNAPSHOT_JSON.read_text(encoding="utf-8")
    for pattern in ("password", "token", "secret", "api_key", "API_KEY"):
        assert pattern not in json_text.lower(), f"Found credential-like pattern: {pattern}"


def test_deterministic_json(snapshot):
    reserialized = json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    original = SNAPSHOT_JSON.read_text(encoding="utf-8")
    assert reserialized == original


def test_markdown_exists(snapshot):
    assert SNAPSHOT_MD.is_file()
    md_text = SNAPSHOT_MD.read_text(encoding="utf-8")
    assert len(md_text) > 1000
    assert "# FI-2010 Classical and MLP-LOB" in md_text


def test_snapshot_byte_identical(snapshot):
    """Snapshot generation is deterministic — does not mutate tracked files."""
    import subprocess
    import sys

    # Write to temp dir, verify runs twice produce same output
    with tempfile.TemporaryDirectory():
        # Simulate by reading the existing snapshot, writing a fresh copy via the generator
        # and verifying it matches the tracked file. The generator writes to fixed paths,
        # so we can't redirect output. Instead verify the tracked file is idempotent.
        before_json = SNAPSHOT_JSON.read_bytes()
        before_md = SNAPSHOT_MD.read_bytes()

        result = subprocess.run(
            [sys.executable, "-m", "deepbook.cli.fi2010_baselines", "snapshot"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Second snapshot failed: {result.stderr}"

        after_json = SNAPSHOT_JSON.read_bytes()
        after_md = SNAPSHOT_MD.read_bytes()

        assert before_json == after_json, "JSON not byte-identical across two runs"
        assert before_md == after_md, "Markdown not byte-identical across two runs"


# ============================================================
# Historical snapshot — additional integrity
# ============================================================


def test_all_models_have_aggregates(snapshot):
    agg = snapshot["aggregates"]
    models_seen = set()
    for key in agg:
        model = key.split("|")[0]
        models_seen.add(model)
    for model in _SELECTED_MODELS:
        assert model in models_seen, f"{model} missing from aggregates"


def test_all_horizons_have_aggregates(snapshot):
    agg = snapshot["aggregates"]
    horizons_seen = set()
    for key in agg:
        h_str = key.split("|")[-1].replace("h", "")
        horizons_seen.add(int(h_str))
    for h in _HORIZONS:
        assert h in horizons_seen, f"h{h} missing from aggregates"


def test_status_all_zeros(snapshot):
    st = snapshot["status"]
    assert st["failed"] == 0
    assert st["interrupted"] == 0
    assert st["running"] == 0
    assert st["duplicate"] == 0
    assert st["ineligible"] == 0
    assert st["orphan_predictions"] == 0
    assert st["orphan_checkpoints"] == 0
    assert st["missing"] == 250


def test_verification_counts(snapshot):
    v = snapshot["verification"]
    assert v["total_verified"] == 650
    assert v["metric_mismatches"] == 0
    assert v["prediction_mismatches"] == 0


# ============================================================
# SUITE SNAPSHOT TESTS
# ============================================================


def test_suite_schema_valid(suite_snapshot):
    import jsonschema

    schema = json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=suite_snapshot, schema=schema)


def test_suite_exact_900_coverage(suite_snapshot):
    cov = suite_snapshot["coverage"]
    assert cov["planned_total"] == 900
    assert cov["completed_confirmatory"] == 900
    assert cov["missing"] == 0
    assert cov["failed"] == 0
    assert cov["interrupted"] == 0
    assert cov["running"] == 0


def test_suite_model_totals(suite_snapshot):
    bm = suite_snapshot["coverage"]["by_model"]
    assert bm["majority"] == 50
    assert bm["causal_persistence"] == 50
    assert bm["logistic_current_event"] == 50
    assert bm["random_forest"] == 250
    assert bm["mlplob"] == 250
    assert bm["deeplob"] == 250


def test_suite_setup_totals(suite_snapshot):
    bs = suite_snapshot["coverage"]["by_setup"]
    assert bs["anchored_forward"] == 810
    assert bs["first_seven_final_three"] == 90


def test_suite_deeplob_execution_commit(suite_snapshot):
    assert (
        suite_snapshot["protocol_provenance"]["execution_commit"]
        == "dc78a82d206ab50399bea0a0c147884a94c66e8f"
    )


def test_suite_deeplob_parameter_count(suite_snapshot):
    assert suite_snapshot["deeplob"]["parameter_count"] == 143907


def test_suite_deeplob_cuda(suite_snapshot):
    dl = suite_snapshot["deeplob"]
    assert dl["cuda_runs"] == 250
    assert dl["nonzero_gpu_memory_runs"] == 250


def test_suite_deeplob_no_collapse(suite_snapshot):
    assert suite_snapshot["deeplob"]["collapse_count"] == 0


def test_suite_deeplob_termination(suite_snapshot):
    term = suite_snapshot["deeplob"]["termination_reasons"]
    assert term.get("early_stopping", 0) == 249
    assert term.get("max_epochs", 0) == 1


def test_suite_five_seeds(suite_snapshot):
    for key, entry in suite_snapshot["deep_lob_aggregates"].items():
        assert len(entry["seeds"]) == 5, f"{key}: expected 5 seeds, got {len(entry['seeds'])}"


def test_suite_all_horizons(suite_snapshot):
    horizons_seen = set()
    for key in suite_snapshot["deep_lob_aggregates"]:
        h = int(key.split("|")[-1].replace("h", ""))
        horizons_seen.add(h)
    for h in _HORIZONS:
        assert h in horizons_seen, f"h{h} missing from DeepLOB aggregates"


def test_suite_standard_deviation_convention(suite_snapshot):
    assert (
        "population"
        in suite_snapshot["protocol_provenance"]["standard_deviation_convention"].lower()
    )


def test_suite_no_post_hoc_threshold(suite_snapshot):
    disc = suite_snapshot["disclosures"]["published_reference_note"]
    assert "not treated as a confirmatory acceptance criterion" in disc


def test_suite_parameter_discrepancy(suite_snapshot):
    note = suite_snapshot["deeplob"]["parameter_count_note"]
    assert "143,907" in note
    assert "60,000" in note or "60k" in note.lower()


def test_suite_logits_note(suite_snapshot):
    note = suite_snapshot["deeplob"]["logits_note"]
    assert "logits" in note.lower()
    assert "CrossEntropyLoss" in note


def test_suite_reconciliation_separate(suite_snapshot):
    rec = suite_snapshot["reconciliation"]
    assert "historical_classical_reconciliation" in rec
    assert "deeplob_execution_reconciliation" in rec
    assert rec["historical_classical_reconciliation"]["count"] == 6
    assert rec["deeplob_execution_reconciliation"]["count"] == 9


def test_suite_no_absolute_paths(suite_snapshot):
    json_text = SUITE_JSON.read_text(encoding="utf-8")
    assert "C:\\\\Users" not in json_text
    assert "C:/Users/" not in json_text
    assert "rohit" not in json_text


def test_suite_sklearn_limitation(suite_snapshot):
    disc = suite_snapshot["disclosures"]["scikit_learn_limitation"]
    assert "unknown" in disc.lower()
    assert "scikit-learn" in disc.lower()


def test_suite_push_disclosure(suite_snapshot):
    disc = suite_snapshot["disclosures"]["push_status"]
    assert "c3c9b98" in disc
    assert "dc78a82" in disc
    assert "local" in disc


def test_suite_deterministic(suite_snapshot):
    reserialized = json.dumps(suite_snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    original = SUITE_JSON.read_text(encoding="utf-8")
    assert reserialized == original


def test_suite_markdown_exists(suite_snapshot):
    assert SUITE_MD.is_file()
    md_text = SUITE_MD.read_text(encoding="utf-8")
    assert len(md_text) > 1000
    assert "FI-2010 Baseline Suite" in md_text


def test_snapshot_generator_does_not_mutate_tracked_outputs():
    """Running snapshot does not change git-tracked files."""
    import subprocess
    import sys

    before = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True
    ).stdout

    result = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.fi2010_baselines", "snapshot"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    result2 = subprocess.run(
        [sys.executable, "-m", "deepbook.cli.fi2010_baselines", "snapshot-suite"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result2.returncode == 0

    after = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    assert before == after, f"git status changed: before={before!r} after={after!r}"
