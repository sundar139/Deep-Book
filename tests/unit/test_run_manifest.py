"""Run manifest construction, validation, and fingerprint tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from deepbook.training.fi2010 import (
    build_run_manifest,
    chronological_training_validation_split,
    configuration_hash,
    protocol_sha256,
    report_fingerprint,
    resolve_protocol_commit,
    validate_run_manifest,
)

_PLACEHOLDER_COMMIT = "0" * 40


def _protocol_commit(root: Path) -> str:
    """Resolve the protocol commit, falling back outside a Git work tree.

    The schema assertions below are about manifest shape, not Git provenance, so
    they must still run from a plain source export.
    """
    return resolve_protocol_commit(root) or _PLACEHOLDER_COMMIT


def _minimal_valid_manifest(root: Path) -> dict:
    """Return a minimal valid manifest using real file-system values where needed."""
    return build_run_manifest(
        run_id="unit-test",
        code_commit="0" * 40,
        dirty=False,
        model="majority",
        setup="anchored_forward",
        fold=1,
        horizon=10,
        seed=1337,
        data_fingerprint="data-fingerprint",
        configuration_hash="a" * 64,
        configuration_path="configs/experiments/fi2010/classical.yaml",
        status="completed",
        metrics={"accuracy": 0.5},
        run_kind="smoke",
        eligible_for_confirmatory_report=False,
        exclusion_reasons=["test"],
        protocol_commit=_protocol_commit(root),
        protocol_sha256=protocol_sha256(root),
        configured_max_epochs=None,
        actual_epochs_completed=None,
        best_epoch=None,
        termination_reason="not_applicable",
        resumed=False,
        resumed_from_run_id=None,
        day_group=None,
        started_utc="2026-01-01T00:00:00Z",
        completed_utc="2026-01-01T00:01:00Z",
        archive_sha256="c" * 64,
        training_file_sha256="d" * 64,
        testing_file_sha256="e" * 64,
        parameter_count=0,
        device="cpu",
        environment={"python": "3.11"},
        labels_regenerated=False,
        test_set_used_for_selection=False,
        # A completed run must persist a verifiable prediction artifact.
        prediction_path="artifacts/fi2010/baselines/predictions/unit-test.npz",
        prediction_sha256="f" * 64,
        sample_count=128,
        class_order=["up", "stationary", "down"],
        testing_file_sha256_by_day={"2": "e" * 64},
        day_index_map={"2": {"source_fold": 1, "file_sha256": "e" * 64, "observations": 38397}},
    )


def test_chronological_split_has_purge_and_embargo_gap() -> None:
    train, validation = chronological_training_validation_split(
        observation_count=100,
        validation_fraction=0.2,
        purge_events=3,
        embargo_events=4,
    )
    assert train.tolist() == list(range(77))
    assert validation.tolist() == list(range(84, 100))
    assert set(train).isdisjoint(validation)
    assert validation[0] - train[-1] - 1 == 7


def test_configuration_and_report_fingerprints_are_stable() -> None:
    configuration = {"b": [2, 1], "a": {"z": True}}
    assert configuration_hash(configuration) == configuration_hash({"a": {"z": True}, "b": [2, 1]})
    report = {"runs": [{"seed": 1337, "accuracy": 0.5}], "schema_version": 1}
    assert report_fingerprint(report) == report_fingerprint(json.loads(json.dumps(report)))
    timestamped = {**report, "generated_utc": "2026-01-01T00:00:00Z"}
    assert report_fingerprint(report) == report_fingerprint(timestamped)
    # generated_from_manifests_utc also excluded
    timestamped2 = {**report, "generated_from_manifests_utc": "2026-01-01T00:00:00Z"}
    assert report_fingerprint(report) == report_fingerprint(timestamped2)


def test_run_manifest_validates_against_repository_schema() -> None:
    root = Path(__file__).resolve().parents[2]
    schema_path = root / "data_contracts" / "fi2010_run_manifest.schema.json"
    manifest = _minimal_valid_manifest(root)
    validate_run_manifest(manifest, schema_path)
    assert manifest["schema_version"] == 1


def test_manifest_with_empty_metrics_fails_validation() -> None:
    """Empty metrics block must fail schema validation (minProperties: 1)."""
    root = Path(__file__).resolve().parents[2]
    schema_path = root / "data_contracts" / "fi2010_run_manifest.schema.json"
    manifest = _minimal_valid_manifest(root)
    manifest["metrics"] = {}
    try:
        validate_run_manifest(manifest, schema_path)
        pytest.fail("empty metrics should have raised validation error")
    except Exception:
        pass


def test_protocol_sha256_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[2]
    h1 = protocol_sha256(root)
    h2 = protocol_sha256(root)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_resolve_protocol_commit_returns_valid_sha() -> None:
    root = Path(__file__).resolve().parents[2]
    commit = resolve_protocol_commit(root)
    if not commit:
        pytest.skip("protocol commit resolution requires a Git work tree")
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_resolve_protocol_commit_is_empty_outside_a_git_work_tree(tmp_path: Path) -> None:
    """A non-repository must yield no provenance rather than crash."""
    assert resolve_protocol_commit(tmp_path) == ""


def test_completed_run_without_a_prediction_artifact_fails_validation() -> None:
    root = Path(__file__).resolve().parents[2]
    schema_path = root / "data_contracts" / "fi2010_run_manifest.schema.json"
    for missing in ("prediction_path", "prediction_sha256", "sample_count", "class_order"):
        manifest = _minimal_valid_manifest(root)
        del manifest[missing]
        with pytest.raises(Exception, match="."):
            validate_run_manifest(manifest, schema_path)


def test_eligible_run_must_have_no_exclusion_reasons_and_a_clean_tree() -> None:
    root = Path(__file__).resolve().parents[2]
    schema_path = root / "data_contracts" / "fi2010_run_manifest.schema.json"
    eligible = _minimal_valid_manifest(root)
    eligible.update(
        eligible_for_confirmatory_report=True, run_kind="confirmatory", exclusion_reasons=[]
    )
    validate_run_manifest(eligible, schema_path)

    with pytest.raises(Exception, match="."):
        validate_run_manifest({**eligible, "exclusion_reasons": ["late reason"]}, schema_path)
    with pytest.raises(Exception, match="."):
        validate_run_manifest({**eligible, "git_tree_dirty": True}, schema_path)
    with pytest.raises(Exception, match="."):
        validate_run_manifest({**eligible, "run_kind": "smoke"}, schema_path)


def test_completed_classical_run_may_not_claim_epochs_or_checkpoints() -> None:
    root = Path(__file__).resolve().parents[2]
    schema_path = root / "data_contracts" / "fi2010_run_manifest.schema.json"
    manifest = _minimal_valid_manifest(root)
    for field, value in (
        ("configured_max_epochs", 50),
        ("actual_epochs_completed", 3),
        ("best_epoch", 2),
        ("termination_reason", "early_stopping"),
        ("best_checkpoint_path", "artifacts/run.best.pt"),
    ):
        with pytest.raises(Exception, match="."):
            validate_run_manifest({**manifest, field: value}, schema_path)


def test_completed_neural_run_requires_epoch_and_best_checkpoint_provenance() -> None:
    root = Path(__file__).resolve().parents[2]
    schema_path = root / "data_contracts" / "fi2010_run_manifest.schema.json"
    neural = _minimal_valid_manifest(root)
    neural.update(
        model="deeplob",
        configured_max_epochs=50,
        actual_epochs_completed=17,
        best_epoch=12,
        termination_reason="early_stopping",
        best_checkpoint_path="artifacts/run.best.pt",
        best_checkpoint_sha256="a" * 64,
        last_checkpoint_path="artifacts/run.last.pt",
    )
    validate_run_manifest(neural, schema_path)

    for field in (
        "configured_max_epochs",
        "actual_epochs_completed",
        "best_epoch",
        "best_checkpoint_path",
        "best_checkpoint_sha256",
        "last_checkpoint_path",
    ):
        broken = dict(neural)
        del broken[field]
        with pytest.raises(Exception, match="."):
            validate_run_manifest(broken, schema_path)

    with pytest.raises(Exception, match="."):
        validate_run_manifest({**neural, "actual_epochs_completed": 0}, schema_path)
    with pytest.raises(Exception, match="."):
        validate_run_manifest({**neural, "termination_reason": "not_applicable"}, schema_path)
