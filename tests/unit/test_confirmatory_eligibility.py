"""Confirmatory eligibility, logical duplicate detection, and run-index planning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deepbook.evaluation.prediction import save_prediction_artifact
from deepbook.training.fi2010 import (
    SETUP_ANCHORED_FORWARD,
    SETUP_FIRST_SEVEN_FINAL_THREE,
    expected_archive_sha256,
    expected_data_fingerprint,
)
from deepbook.training.runner import (
    EligibilityContext,
    _checkpoint_is_valid,
    _prediction_is_valid,
    classify_run,
    duplicate_logical_identities,
    generate_run_index,
    logical_identity,
    planned_run_specs,
    write_run_index,
)

ROOT = Path(__file__).resolve().parents[2]


def _context(**overrides: object) -> EligibilityContext:
    base: dict[str, object] = {
        "model": "deeplob",
        "status": "completed",
        "termination_reason": "early_stopping",
        "smoke": False,
        "git_dirty": False,
        "protocol_ancestry_ok": True,
        "protocol_hash_matches": True,
        "configuration_hash_matches": True,
        "data_fingerprint_matches": True,
        "archive_sha256_matches": True,
        "prediction_valid": True,
        "checkpoint_valid": True,
        "configured_max_epochs": 50,
        "actual_epochs_completed": 17,
        "best_epoch": 12,
    }
    base.update(overrides)
    return EligibilityContext(**base)  # type: ignore[arg-type]


def _classical_context(**overrides: object) -> EligibilityContext:
    base: dict[str, object] = {
        "model": "majority",
        "termination_reason": "not_applicable",
        "checkpoint_valid": False,
        "configured_max_epochs": None,
        "actual_epochs_completed": None,
        "best_epoch": None,
    }
    base.update(overrides)
    return _context(**base)


# --- Accepted ---------------------------------------------------------------


def test_early_stopping_is_confirmatory() -> None:
    kind, eligible, reasons = classify_run(_context(termination_reason="early_stopping"))
    assert (kind, eligible, reasons) == ("confirmatory", True, [])


def test_max_epochs_is_confirmatory() -> None:
    kind, eligible, reasons = classify_run(
        _context(termination_reason="max_epochs", actual_epochs_completed=50, best_epoch=50)
    )
    assert (kind, eligible, reasons) == ("confirmatory", True, [])


def test_deterministic_classical_completion_is_confirmatory() -> None:
    kind, eligible, reasons = classify_run(_classical_context())
    assert (kind, eligible, reasons) == ("confirmatory", True, [])


# --- Rejected ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"smoke": True}, "explicitly marked as smoke"),
        ({"git_dirty": True}, "git tree is dirty"),
        ({"status": "failed"}, "not completed"),
        ({"status": "interrupted"}, "not completed"),
        ({"protocol_ancestry_ok": False}, "does not descend from protocol commit"),
        ({"protocol_hash_matches": False}, "protocol hash"),
        ({"configuration_hash_matches": False}, "configuration hash"),
        ({"data_fingerprint_matches": False}, "frozen FI-2010 data identity"),
        ({"archive_sha256_matches": False}, "archive digest"),
        ({"prediction_valid": False}, "prediction artifact is missing or invalid"),
        ({"checkpoint_valid": False}, "best-model checkpoint is missing or invalid"),
        ({"actual_epochs_completed": 0}, "at least 1"),
        ({"actual_epochs_completed": None}, "at least 1"),
        ({"actual_epochs_completed": 99}, "exceeds configured_max_epochs"),
        ({"best_epoch": 40, "actual_epochs_completed": 17}, "exceeds actual_epochs_completed"),
        ({"best_epoch": 0}, "best_epoch must be at least 1"),
        ({"configured_max_epochs": 0}, "positive integer"),
        ({"termination_reason": "not_applicable"}, "invalid neural termination reason"),
        ({"termination_reason": "failed"}, "invalid neural termination reason"),
    ],
)
def test_neural_eligibility_rejections(overrides: dict[str, object], fragment: str) -> None:
    kind, eligible, reasons = classify_run(_context(**overrides))
    assert not eligible
    assert kind == "smoke"
    assert any(fragment in reason for reason in reasons), reasons


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"configured_max_epochs": 50}, "null configured_max_epochs"),
        ({"actual_epochs_completed": 3}, "null actual_epochs_completed"),
        ({"best_epoch": 2}, "null best_epoch"),
        ({"termination_reason": "early_stopping"}, "must be not_applicable"),
        ({"prediction_valid": False}, "prediction artifact is missing or invalid"),
    ],
)
def test_classical_eligibility_rejections(overrides: dict[str, object], fragment: str) -> None:
    _kind, eligible, reasons = classify_run(_classical_context(**overrides))
    assert not eligible
    assert any(fragment in reason for reason in reasons), reasons


def test_every_failure_is_reported_not_just_the_first() -> None:
    _kind, eligible, reasons = classify_run(
        _context(git_dirty=True, data_fingerprint_matches=False, actual_epochs_completed=0)
    )
    assert not eligible
    assert len(reasons) >= 3


# --- Frozen data identity ---------------------------------------------------


def test_frozen_fingerprint_differs_per_setup_cell() -> None:
    fold_one = expected_data_fingerprint(ROOT, setup=SETUP_ANCHORED_FORWARD, fold=1)
    fold_seven = expected_data_fingerprint(ROOT, setup=SETUP_ANCHORED_FORWARD, fold=7)
    day_group = expected_data_fingerprint(
        ROOT, setup=SETUP_FIRST_SEVEN_FINAL_THREE, day_group="days_8_9_10"
    )
    assert len({fold_one, fold_seven, day_group}) == 3
    assert len(expected_archive_sha256(ROOT)) == 64


def test_changed_data_identity_is_rejected() -> None:
    frozen = expected_data_fingerprint(ROOT, setup=SETUP_ANCHORED_FORWARD, fold=1)
    recorded = "tampered-fingerprint"
    _kind, eligible, reasons = classify_run(
        _classical_context(data_fingerprint_matches=recorded == frozen)
    )
    assert not eligible
    assert any("frozen FI-2010 data identity" in reason for reason in reasons)
    # The same run with the true frozen fingerprint is accepted.
    _kind, eligible, _reasons = classify_run(
        _classical_context(data_fingerprint_matches=frozen == frozen)
    )
    assert eligible


# --- Artifact validity helpers ---------------------------------------------


def test_prediction_validity_requires_a_loadable_artifact_of_the_recorded_size(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.npz"
    count = 30
    rng = np.random.default_rng(1)
    save_prediction_artifact(
        path,
        y_true=rng.integers(0, 3, count).astype(np.int64),
        y_pred=rng.integers(0, 3, count).astype(np.int64),
        probabilities=np.full((count, 3), 1.0 / 3.0),
        sample_index=np.arange(count, dtype=np.int64),
        source_file_id=np.ones(count, dtype=np.int64),
        day_boundary_id=np.full(count, 8, dtype=np.int64),
    )
    assert _prediction_is_valid(path, count)
    assert not _prediction_is_valid(path, count + 1)
    assert not _prediction_is_valid(tmp_path / "missing.npz", count)
    assert not _prediction_is_valid(None, count)

    (tmp_path / "broken.npz").write_bytes(b"not an npz")
    assert not _prediction_is_valid(tmp_path / "broken.npz", None)


def test_checkpoint_validity_requires_a_matching_digest(tmp_path: Path) -> None:
    path = tmp_path / "run.best.pt"
    path.write_bytes(b"weights")
    from deepbook.evaluation.prediction import sha256_file

    digest = sha256_file(path)
    assert _checkpoint_is_valid(path, digest)
    assert not _checkpoint_is_valid(path, "0" * 64)
    assert not _checkpoint_is_valid(path, None)
    assert not _checkpoint_is_valid(tmp_path / "missing.pt", digest)


# --- Logical duplicates -----------------------------------------------------


def _manifest(run_id: str, **overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "run_id": run_id,
        "model": "majority",
        "setup": SETUP_ANCHORED_FORWARD,
        "fold": 1,
        "day_group": None,
        "horizon": 10,
        "seed": 1337,
        "configuration_hash": "a" * 64,
        "data_fingerprint": "fingerprint",
        "run_kind": "confirmatory",
        "status": "completed",
        "eligible_for_confirmatory_report": True,
        "metrics": {"test": {"macro_f1": 0.5}},
    }
    manifest.update(overrides)
    return manifest


def test_duplicate_logical_identity_is_detected_across_distinct_run_ids() -> None:
    first = _manifest("majority-anchored-forward-f1-h10-s1337")
    second = _manifest("majority-anchored-forward-f1-h10-s1337-rerun")
    assert logical_identity(first) == logical_identity(second)

    conflicts = duplicate_logical_identities([first, second])
    assert len(conflicts) == 1
    assert conflicts[0]["run_ids"] == sorted([str(first["run_id"]), str(second["run_id"])])
    assert conflicts[0]["model"] == "majority"


def test_distinct_cells_are_not_duplicates() -> None:
    manifests = [
        _manifest("a", fold=1),
        _manifest("b", fold=2),
        _manifest("c", seed=2027),
        _manifest("d", setup=SETUP_FIRST_SEVEN_FINAL_THREE, fold=None, day_group="days_8_9_10"),
    ]
    assert duplicate_logical_identities(manifests) == []


def test_duplicate_logical_identities_are_excluded_from_confirmatory_totals(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    for run_id in ("first-run", "second-run"):
        (runs / f"{run_id}.json").write_text(
            json.dumps(_manifest(run_id), sort_keys=True), encoding="utf-8"
        )
    index = generate_run_index(ROOT, tmp_path)
    assert index["duplicate_run_id_count"] == 2
    assert len(index["duplicate_logical_identities"]) == 1
    assert index["completed_confirmatory"] == []
    assert sorted(index["completed_smoke"]) == ["first-run", "second-run"]
    assert all(entry["duplicate_logical_identity"] for entry in index["runs"])


# --- Run index --------------------------------------------------------------


def test_run_index_reports_planned_totals_for_both_setups(tmp_path: Path) -> None:
    index = generate_run_index(ROOT, tmp_path)
    totals = index["planned_totals"]
    assert index["planned_cell_count"] == len(planned_run_specs(ROOT))
    assert totals["planned_by_setup"] == {
        SETUP_ANCHORED_FORWARD: 810,
        SETUP_FIRST_SEVEN_FINAL_THREE: 90,
    }
    assert totals["planned_by_model"]["majority"] == 50
    assert totals["planned_by_model"]["deeplob"] == 250
    assert totals["planned_by_fold_or_day_group"]["day_group_days_8_9_10"] == 90
    assert sorted(totals["planned_by_horizon"]) == ["h10", "h100", "h20", "h30", "h50"]
    assert len(totals["planned_by_seed"]) == 5


def test_run_index_is_written_atomically_and_deterministically(tmp_path: Path) -> None:
    first_path = write_run_index(ROOT, tmp_path)
    assert first_path.is_file()
    first = first_path.read_bytes()
    second = write_run_index(ROOT, tmp_path).read_bytes()
    assert first == second
    payload = json.loads(first.decode("utf-8"))
    assert payload["total_manifest_count"] == 0
    assert payload["generated_from_manifests_utc"] is None
    assert payload["missing_manifests"] == payload["planned_cells"]


def test_run_index_detects_failures_interruptions_and_orphans(tmp_path: Path) -> None:
    paths = tmp_path
    (paths / "runs").mkdir(parents=True)
    (paths / "checkpoints").mkdir(parents=True)
    (paths / "predictions").mkdir(parents=True)
    (paths / "runs" / "failed-run.json").write_text(
        json.dumps(_manifest("failed-run", status="failed"), sort_keys=True), encoding="utf-8"
    )
    (paths / "runs" / "interrupted-run.json").write_text(
        json.dumps(_manifest("interrupted-run", status="interrupted", fold=2), sort_keys=True),
        encoding="utf-8",
    )
    (paths / "checkpoints" / "unclaimed.best.pt").write_bytes(b"x")
    (paths / "predictions" / "unclaimed.npz").write_bytes(b"x")

    index = generate_run_index(ROOT, paths)
    assert index["failed"] == ["failed-run"]
    assert index["interrupted"] == ["interrupted-run"]
    assert index["orphan_checkpoints"] == ["unclaimed.best.pt"]
    assert index["orphan_predictions"] == ["unclaimed.npz"]
    assert len(index["missing_manifests"]) == index["planned_cell_count"]
