"""End-to-end pipeline test using synthetic data only."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from deepbook.evaluation.classification import classification_metrics
from deepbook.evaluation.prediction import (
    load_prediction_artifact,
    recompute_metrics_from_artifact,
    save_prediction_artifact,
    sha256_file,
)
from deepbook.training.fi2010 import (
    build_run_manifest,
    configuration_hash,
    validate_run_manifest,
)


def _make_synthetic_predictions(n: int = 200) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 3, size=n).astype(np.int64)
    y_pred = rng.integers(0, 3, size=n).astype(np.int64)
    proba = np.eye(3)[y_pred].astype(np.float64)
    jitter = rng.uniform(0, 0.05, size=proba.shape)
    proba = proba + jitter
    proba /= proba.sum(axis=1, keepdims=True)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": proba,
        "sample_index": np.arange(n, dtype=np.int64),
        "source_file_id": np.full(n, 1, dtype=np.int64),
        "day_boundary_id": np.zeros(n, dtype=np.int64),
    }


class TestBaselineRunPipeline:
    """End-to-end synthetic pipeline: predict, save, manifest, validate, recompute."""

    def test_prediction_artifact_roundtrip(self) -> None:
        preds = _make_synthetic_predictions(100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.npz"
            save_prediction_artifact(
                path,
                y_true=preds["y_true"],
                y_pred=preds["y_pred"],
                probabilities=preds["probabilities"],
                sample_index=preds["sample_index"],
                source_file_id=preds["source_file_id"],
                day_boundary_id=preds["day_boundary_id"],
            )
            assert path.is_file()
            loaded = load_prediction_artifact(path)
            assert np.array_equal(loaded["y_true"], preds["y_true"])
            assert np.array_equal(loaded["y_pred"], preds["y_pred"])
            assert np.allclose(loaded["probabilities"], preds["probabilities"])
            assert tuple(loaded["class_order"]) == ("up", "stationary", "down")

    def test_metric_recomputation_from_artifact(self) -> None:
        preds = _make_synthetic_predictions(200)
        metrics_direct = classification_metrics(
            preds["y_true"], preds["y_pred"], preds["probabilities"]
        )
        metrics_from_artifact = recompute_metrics_from_artifact(preds)
        for key in ("macro_f1", "mcc", "accuracy", "balanced_accuracy", "nll", "brier", "ece"):
            assert np.allclose(
                float(metrics_direct[key]),
                float(metrics_from_artifact[key]),
                rtol=1e-10,
                atol=1e-8,
            ), f"Mismatch for {key}"
        assert metrics_direct["confusion_matrix"] == metrics_from_artifact["confusion_matrix"]

    def test_manifest_validation_with_prediction_hash(self) -> None:
        preds = _make_synthetic_predictions(50)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pred_path = tmp / "predictions" / "test-run.npz"
            save_prediction_artifact(
                pred_path,
                y_true=preds["y_true"],
                y_pred=preds["y_pred"],
                probabilities=preds["probabilities"],
                sample_index=preds["sample_index"],
                source_file_id=preds["source_file_id"],
                day_boundary_id=preds["day_boundary_id"],
            )
            pred_hash = sha256_file(pred_path)

            metrics = classification_metrics(
                preds["y_true"], preds["y_pred"], preds["probabilities"]
            )
            config = {"training": {"max_epochs": 50}}
            manifest = build_run_manifest(
                run_id="test-run",
                code_commit="0" * 40,
                dirty=False,
                model="test_model",
                setup="anchored_forward",
                fold=1,
                horizon=10,
                seed=1337,
                data_fingerprint="test_fingerprint",
                configuration_hash=configuration_hash(config),
                status="completed",
                metrics={"test": metrics},
                run_kind="smoke",
                eligible_for_confirmatory_report=False,
                exclusion_reasons=["test run"],
                configured_max_epochs=50,
                actual_epochs_completed=1,
                prediction_sha256=pred_hash,
                prediction_path=str(pred_path),
                checkpoint_sha256=None,
                checkpoint_path=None,
                started_utc="2026-01-01T00:00:00Z",
                completed_utc="2026-01-01T00:01:00Z",
            )
            # Schema validation
            schema_path = (
                Path(__file__).resolve().parents[2]
                / "data_contracts"
                / "fi2010_run_manifest.schema.json"
            )
            validate_run_manifest(manifest, schema_path)

            # Write manifest, read back, verify
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            reloaded = json.loads(manifest_path.read_text())
            assert reloaded["run_id"] == "test-run"
            assert reloaded["prediction_sha256"] == pred_hash
            assert not reloaded["eligible_for_confirmatory_report"]
            assert reloaded["exclusion_reasons"] == ["test run"]

    def test_smoke_is_excluded_from_confirmatory(self) -> None:
        """Smoke run must have eligible_for_confirmatory_report=False."""
        manifest = build_run_manifest(
            run_id="smoke-test",
            code_commit="0" * 40,
            dirty=True,
            model="test",
            setup="anchored_forward",
            fold=1,
            horizon=10,
            seed=1337,
            data_fingerprint="fp",
            configuration_hash="0" * 64,
            status="completed",
            metrics={},
            run_kind="smoke",
            eligible_for_confirmatory_report=False,
            exclusion_reasons=["git tree is dirty"],
        )
        assert manifest["run_kind"] == "smoke"
        assert not manifest["eligible_for_confirmatory_report"]
        assert len(manifest["exclusion_reasons"]) == 1
