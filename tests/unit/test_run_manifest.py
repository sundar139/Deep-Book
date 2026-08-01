from __future__ import annotations

import json
from pathlib import Path

from deepbook.training.fi2010 import (
    build_run_manifest,
    chronological_training_validation_split,
    configuration_hash,
    report_fingerprint,
    validate_run_manifest,
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


def test_run_manifest_validates_against_repository_schema(tmp_path: Path) -> None:
    schema_path = Path("data_contracts/fi2010_run_manifest.schema.json")
    manifest = build_run_manifest(
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
        status="completed",
        metrics={"accuracy": 0.5},
    )
    validate_run_manifest(manifest, schema_path)
    assert manifest["schema_version"] == 1
