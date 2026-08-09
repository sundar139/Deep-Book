"""End-to-end pipeline tests driving the real production orchestration.

Every test here runs the same ``execute_run`` the CLI runs, against a synthetic
repository root and a synthetic data provider. Nothing about the run, manifest,
eligibility, index, report, or verification path is reimplemented in the test.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from deepbook.cli.fi2010_baselines import _cmd_prepare, _cmd_report, _cmd_verify_run
from deepbook.evaluation.classification import classification_metrics
from deepbook.evaluation.prediction import (
    load_prediction_artifact,
    recompute_metrics_from_artifact,
    save_prediction_artifact,
    sha256_file,
)
from deepbook.training.fi2010 import (
    DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
    SETUP_ANCHORED_FORWARD,
    SETUP_FIRST_SEVEN_FINAL_THREE,
    code_commit_provenance_reasons,
    configuration_hash,
    expected_archive_sha256,
    expected_data_fingerprint,
    frozen_data_identity,
    protocol_sha256,
    validate_run_manifest,
)
from deepbook.training.runner import (
    RunData,
    RunSpec,
    SourceSegment,
    _record_run,
    completed_run_skip_reasons,
    execute_run,
    generate_run_index,
    reconcile_interrupted_artifacts,
    run_paths,
    write_report,
    write_run_index,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRACKED_INPUTS = (
    # The real ignore rules come along so generated artifacts leave the tree clean,
    # exactly as they do in the repository.
    ".gitignore",
    "data_contracts/fi2010_run_manifest.schema.json",
    "configs/data/fi2010.yaml",
    "configs/references/fi2010_frozen_data_identity.yaml",
    "configs/references/deeplob_fi2010.yaml",
    "configs/references/translob_fi2010.yaml",
    "configs/references/tlob_fi2010.yaml",
    "configs/experiments/fi2010/classical.yaml",
    "configs/experiments/fi2010/mlplob.yaml",
    "configs/experiments/fi2010/deeplob.yaml",
    "configs/experiments/fi2010/translob.yaml",
    "configs/experiments/fi2010/tlob.yaml",
    "reports/protocol/fi2010_baseline_reproduction.md",
    "reports/protocol/fi2010_transformer_architecture_freeze.md",
)


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """A committed repository root carrying the real frozen contract files."""
    root = tmp_path / "repository"
    for relative in TRACKED_INPUTS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / relative, destination)
    interim = root / "data" / "interim" / "fi2010"
    interim.mkdir(parents=True, exist_ok=True)
    (interim / "fi2010_split_manifest.json").write_text(
        json.dumps({"schema_version": 1, "splits": []}), encoding="utf-8"
    )
    _git(root, "init", "--quiet", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "synthetic frozen contract")
    return root


def _segment(day_index: int, source_fold: int, observations: int, digest: str) -> SourceSegment:
    rng = np.random.default_rng(day_index * 17)
    classes = rng.integers(0, 3, observations)
    lob = (rng.standard_normal((40, observations)) * 0.4 + classes * 1.5).astype(np.float32)
    engineered = rng.standard_normal((104, observations)).astype(np.float32)
    labels = np.tile(classes + 1, (5, 1)).astype(np.int8)
    return SourceSegment(
        day_index=day_index,
        source_fold=source_fold,
        file_sha256=digest,
        lob=lob,
        features=np.concatenate([lob, engineered]),
        labels=labels,
    )


def _training(observations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    classes = rng.integers(0, 3, observations)
    lob = (rng.standard_normal((40, observations)) * 0.4 + classes * 1.5).astype(np.float32)
    engineered = rng.standard_normal((104, observations)).astype(np.float32)
    labels = np.tile(classes + 1, (5, 1)).astype(np.int8)
    return lob, np.concatenate([lob, engineered]), labels


def _provider(root: Path, observations: int = 2400, test_observations: int = 260):
    """Return a run-data provider matching the frozen contract for either setup."""
    frozen = frozen_data_identity(root)

    def provide(provider_root: Path, spec: RunSpec, _config_path: str) -> RunData:
        training_lob, training_features, training_labels = _training(observations)
        if spec.setup == SETUP_ANCHORED_FORWARD:
            entry = frozen["folds"][int(spec.fold or 1)]
            segments = (
                _segment(
                    int(entry["testing_day_index"]),
                    int(spec.fold or 1),
                    test_observations,
                    str(entry["testing_file_sha256"]),
                ),
            )
            training_digest = str(entry["training_file_sha256"])
            by_day = {str(entry["testing_day_index"]): str(entry["testing_file_sha256"])}
        else:
            group = frozen["day_groups"][str(spec.day_group)]
            segments = tuple(
                _segment(
                    int(day["day_index"]),
                    int(day["source_fold"]),
                    test_observations,
                    str(day["file_sha256"]),
                )
                for day in group["test_days"]
            )
            training_digest = str(group["training_file_sha256"])
            by_day = {str(day["day_index"]): str(day["file_sha256"]) for day in group["test_days"]}
        return RunData(
            setup=spec.setup,
            fold=spec.fold,
            day_group=spec.day_group,
            training_lob=training_lob,
            training_features=training_features,
            training_labels=training_labels,
            test_segments=segments,
            archive_sha256=expected_archive_sha256(provider_root),
            training_file_sha256=training_digest,
            testing_file_sha256=str(frozen["archive_sha256"]),
            testing_file_sha256_by_day=by_day,
            data_fingerprint=expected_data_fingerprint(
                provider_root,
                setup=spec.setup,
                fold=spec.fold,
                day_group=spec.day_group,
            ),
        )

    return provide


def _classical_spec(setup: str = SETUP_ANCHORED_FORWARD) -> RunSpec:
    if setup == SETUP_ANCHORED_FORWARD:
        return RunSpec(model="majority", setup=setup, horizon=10, seed=1337, fold=1)
    return RunSpec(
        model="majority",
        setup=setup,
        horizon=10,
        seed=1337,
        day_group=DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
    )


class TestProductionOrchestration:
    """The CLI's own run, index, report, and verification path end to end."""

    def test_confirmatory_classical_run_completes_the_whole_pipeline(
        self, synthetic_root: Path
    ) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        spec = _classical_spec()
        manifest = execute_run(
            synthetic_root,
            spec,
            artifact_root=artifacts,
            run_data_provider=_provider(synthetic_root),
        )

        manifest_path = run_paths(artifacts).runs / f"{spec.run_id}.json"
        assert manifest_path.is_file()
        validate_run_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            synthetic_root / "data_contracts" / "fi2010_run_manifest.schema.json",
        )
        assert manifest["exclusion_reasons"] == []
        assert manifest["eligible_for_confirmatory_report"] is True
        assert manifest["run_kind"] == "confirmatory"
        assert manifest["termination_reason"] == "not_applicable"
        assert manifest["configured_max_epochs"] is None
        assert manifest["best_checkpoint_path"] is None

        prediction_path = Path(manifest["prediction_path"])
        assert prediction_path.is_file()
        assert sha256_file(prediction_path) == manifest["prediction_sha256"]
        payload = load_prediction_artifact(prediction_path)
        valid = payload["y_pred"] >= 0
        recomputed = classification_metrics(
            payload["y_true"][valid], payload["y_pred"][valid], payload["probabilities"][valid]
        )
        assert recomputed["macro_f1"] == pytest.approx(manifest["metrics"]["test"]["macro_f1"])

        index = json.loads(write_run_index(synthetic_root, artifacts).read_text(encoding="utf-8"))
        assert index["completed_confirmatory"] == [spec.run_id]
        assert index["duplicate_logical_identities"] == []

        json_path, markdown_path = write_report(synthetic_root, artifacts)
        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert [m["run_id"] for m in report["confirmatory_runs"]] == [spec.run_id]
        assert "INCOMPLETE" in markdown_path.read_text(encoding="utf-8")

        assert _cmd_verify_run(synthetic_root, spec.run_id) == 0

    def test_smoke_run_is_excluded_by_the_orchestrator(self, synthetic_root: Path) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        spec = _classical_spec()
        manifest = execute_run(
            synthetic_root,
            spec,
            artifact_root=artifacts,
            smoke=True,
            run_data_provider=_provider(synthetic_root),
        )
        assert manifest["run_kind"] == "smoke"
        assert manifest["eligible_for_confirmatory_report"] is False
        assert "run explicitly marked as smoke" in manifest["exclusion_reasons"]

        index = generate_run_index(synthetic_root, artifacts)
        assert index["completed_confirmatory"] == []
        assert index["completed_smoke"] == [spec.run_id]

        json_path, _ = write_report(synthetic_root, artifacts)
        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert report["confirmatory_runs"] == []
        assert [m["run_id"] for m in report["smoke_runs"]] == [spec.run_id]

    @pytest.mark.parametrize("model", ["mlplob", "translob", "tlob"])
    def test_tiny_neural_run_writes_best_and_last_checkpoints(
        self, synthetic_root: Path, model: str
    ) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        spec = RunSpec(model=model, setup=SETUP_ANCHORED_FORWARD, horizon=10, seed=1337, fold=1)
        manifest = execute_run(
            synthetic_root,
            spec,
            artifact_root=artifacts,
            max_epochs=2,
            smoke=True,
            run_data_provider=_provider(synthetic_root, observations=2400, test_observations=400),
        )
        assert manifest["model"] == model
        assert manifest["termination_reason"] in {"early_stopping", "max_epochs"}
        assert 1 <= manifest["actual_epochs_completed"] <= manifest["configured_max_epochs"]
        assert 1 <= manifest["best_epoch"] <= manifest["actual_epochs_completed"]

        best = Path(manifest["best_checkpoint_path"])
        last = Path(manifest["last_checkpoint_path"])
        assert best.is_file() and last.is_file()
        assert best != last
        assert sha256_file(best) == manifest["best_checkpoint_sha256"]
        assert sha256_file(last) == manifest["last_checkpoint_sha256"]
        assert manifest["checkpoint_round_trip"]["passed"] is True
        assert _cmd_verify_run(synthetic_root, spec.run_id) == 0

    def test_setup_two_run_keeps_three_independent_days(self, synthetic_root: Path) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        spec = _classical_spec(SETUP_FIRST_SEVEN_FINAL_THREE)
        per_day = 260
        manifest = execute_run(
            synthetic_root,
            spec,
            artifact_root=artifacts,
            run_data_provider=_provider(synthetic_root, test_observations=per_day),
        )
        assert manifest["setup"] == SETUP_FIRST_SEVEN_FINAL_THREE
        assert manifest["fold"] is None
        assert manifest["day_group"] == DAY_GROUP_FIRST_SEVEN_FINAL_THREE
        assert sorted(manifest["day_index_map"]) == ["10", "8", "9"]
        assert manifest["sample_count"] == 3 * per_day

        payload = load_prediction_artifact(Path(manifest["prediction_path"]))
        for day, source_fold in ((8, 7), (9, 8), (10, 9)):
            mask = payload["day_boundary_id"] == day
            assert int(mask.sum()) == per_day
            assert set(payload["source_file_id"][mask].tolist()) == {source_fold}
        assert sorted(manifest["metrics"]["test_by_day"]) == ["10", "8", "9"]
        assert _cmd_verify_run(synthetic_root, spec.run_id) == 0

    def test_prepare_and_report_both_persist_the_run_index(self, synthetic_root: Path) -> None:
        index_path = synthetic_root / "artifacts" / "fi2010" / "baselines" / "run_index.json"
        assert not index_path.exists()
        assert _cmd_prepare(synthetic_root) == 0
        assert index_path.is_file()

        index_path.unlink()
        assert _cmd_report(synthetic_root) == 0
        assert index_path.is_file()
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        assert payload["planned_cell_count"] == 1400

    def test_report_artifacts_are_byte_identical_across_unchanged_runs(
        self, synthetic_root: Path
    ) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        execute_run(
            synthetic_root,
            _classical_spec(),
            artifact_root=artifacts,
            run_data_provider=_provider(synthetic_root),
        )
        json_path, markdown_path = write_report(synthetic_root, artifacts)
        index_path = write_run_index(synthetic_root, artifacts)
        first = (json_path.read_bytes(), markdown_path.read_bytes(), index_path.read_bytes())

        write_report(synthetic_root, artifacts)
        write_run_index(synthetic_root, artifacts)
        second = (json_path.read_bytes(), markdown_path.read_bytes(), index_path.read_bytes())
        assert first == second

    def test_duplicate_logical_identity_blocks_confirmatory_aggregation(
        self, synthetic_root: Path
    ) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        spec = _classical_spec()
        manifest = execute_run(
            synthetic_root,
            spec,
            artifact_root=artifacts,
            run_data_provider=_provider(synthetic_root),
        )
        assert manifest["eligible_for_confirmatory_report"] is True

        clone = dict(manifest)
        clone["run_id"] = f"{spec.run_id}-rerun"
        (run_paths(artifacts).runs / f"{clone['run_id']}.json").write_text(
            json.dumps(clone, indent=2, sort_keys=True), encoding="utf-8"
        )
        index = generate_run_index(synthetic_root, artifacts)
        assert index["completed_confirmatory"] == []
        assert len(index["duplicate_logical_identities"]) == 1
        json_path, _ = write_report(synthetic_root, artifacts)
        assert json.loads(json_path.read_text(encoding="utf-8"))["confirmatory_runs"] == []

    def test_verify_run_rejects_tampering_without_crashing(self, synthetic_root: Path) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        spec = _classical_spec()
        manifest = execute_run(
            synthetic_root,
            spec,
            artifact_root=artifacts,
            run_data_provider=_provider(synthetic_root),
        )
        manifest_path = run_paths(artifacts).runs / f"{spec.run_id}.json"
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert _cmd_verify_run(synthetic_root, spec.run_id) == 0

        for field, value in (
            ("configuration_hash", "a" * 64),
            ("protocol_sha256", "b" * 64),
            ("data_fingerprint", "c" * 64),
            ("archive_sha256", "d" * 64),
            ("sample_count", 999_999),
            ("protocol_commit", "f" * 40),
            ("prediction_sha256", "e" * 64),
        ):
            manifest_path.write_text(
                json.dumps({**original, field: value}, indent=2, sort_keys=True), encoding="utf-8"
            )
            assert _cmd_verify_run(synthetic_root, spec.run_id) == 1, field
        manifest_path.write_text(json.dumps(original, indent=2, sort_keys=True), encoding="utf-8")

        # A structurally corrupt artifact fails verification instead of raising.
        prediction = Path(manifest["prediction_path"])
        payload = dict(np.load(prediction, allow_pickle=False))
        payload["day_boundary_id"] = payload["day_boundary_id"][:-5]
        np.savez_compressed(prediction, **payload)
        manifest_path.write_text(
            json.dumps(
                {**original, "prediction_sha256": sha256_file(prediction)},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        assert _cmd_verify_run(synthetic_root, spec.run_id) == 1

    def test_dirty_tree_makes_a_run_ineligible(self, synthetic_root: Path) -> None:
        (synthetic_root / "untracked.txt").write_text("dirty", encoding="utf-8")
        manifest = execute_run(
            synthetic_root,
            _classical_spec(),
            artifact_root=synthetic_root / "artifacts" / "fi2010" / "baselines",
            run_data_provider=_provider(synthetic_root),
        )
        assert manifest["eligible_for_confirmatory_report"] is False
        assert "git tree is dirty" in manifest["exclusion_reasons"]

    def test_confirmatory_matrix_refuses_dirty_tree_before_provider(
        self, synthetic_root: Path
    ) -> None:
        (synthetic_root / "untracked.txt").write_text("dirty", encoding="utf-8")
        called = False

        def provider(*_args: object) -> RunData:
            nonlocal called
            called = True
            raise AssertionError("dirty confirmatory matrix must not load data")

        with pytest.raises(RuntimeError, match="dirty Git tree.*untracked.txt"):
            execute_run(
                synthetic_root,
                _classical_spec(),
                artifact_root=synthetic_root / "artifacts" / "fi2010" / "baselines",
                run_data_provider=provider,
                confirmatory_matrix=True,
            )
        assert not called

    def test_completed_manifest_cannot_be_rewritten(self, synthetic_root: Path) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        manifest = execute_run(
            synthetic_root,
            _classical_spec(),
            artifact_root=artifacts,
            run_data_provider=_provider(synthetic_root),
        )
        manifest_path = run_paths(artifacts).runs / f"{manifest['run_id']}.json"
        original = manifest_path.read_bytes()
        with pytest.raises(ValueError, match="completed manifest is immutable"):
            _record_run(synthetic_root, artifacts, manifest)
        assert manifest_path.read_bytes() == original

    def test_completed_skip_requires_verified_confirmatory_provenance(
        self, synthetic_root: Path
    ) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        spec = _classical_spec()
        manifest = execute_run(
            synthetic_root,
            spec,
            artifact_root=artifacts,
            run_data_provider=_provider(synthetic_root),
        )
        assert (
            completed_run_skip_reasons(synthetic_root, spec, manifest, artifact_root=artifacts)
            == []
        )
        for changed, fragment in (
            ({"run_kind": "smoke", "eligible_for_confirmatory_report": False}, "run_kind"),
            ({"code_commit": "0" * 40}, "code commit"),
            ({"configuration_hash": "a" * 64}, "configuration"),
            ({"protocol_sha256": "b" * 64}, "protocol SHA-256"),
            ({"prediction_path": str(artifacts / "predictions" / "missing.npz")}, "prediction"),
        ):
            reasons = completed_run_skip_reasons(
                synthetic_root, spec, {**manifest, **changed}, artifact_root=artifacts
            )
            assert any(fragment in reason for reason in reasons), reasons

    def test_temporal_provenance_rejects_posthoc_commit_claim(self, synthetic_root: Path) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        manifest = execute_run(
            synthetic_root,
            _classical_spec(),
            artifact_root=artifacts,
            run_data_provider=_provider(synthetic_root),
        )
        forged = {
            **manifest,
            "started_utc": "2000-01-01T00:00:00+00:00",
            "completed_utc": "2000-01-01T00:01:00+00:00",
        }
        reasons = code_commit_provenance_reasons(synthetic_root, forged)
        assert "recorded code commit did not exist when this run executed" in reasons

    def test_run_index_records_running_transition_before_data_load(
        self, synthetic_root: Path
    ) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        spec = _classical_spec()
        provider = _provider(synthetic_root)

        def observing_provider(root: Path, requested: RunSpec, config_path: str) -> RunData:
            index = json.loads((artifacts / "run_index.json").read_text(encoding="utf-8"))
            assert index["running"] == [spec.run_id]
            return provider(root, requested, config_path)

        execute_run(
            synthetic_root,
            spec,
            artifact_root=artifacts,
            run_data_provider=observing_provider,
        )

    def test_orphan_last_checkpoint_is_recoverable(self, synthetic_root: Path) -> None:
        artifacts = synthetic_root / "artifacts" / "fi2010" / "baselines"
        checkpoints = run_paths(artifacts).checkpoints
        checkpoints.mkdir(parents=True, exist_ok=True)
        spec = RunSpec(model="mlplob", setup=SETUP_ANCHORED_FORWARD, horizon=10, seed=1337, fold=1)
        config = yaml.safe_load(
            (synthetic_root / "configs" / "experiments" / "fi2010" / "mlplob.yaml").read_text(
                encoding="utf-8"
            )
        )
        frozen = frozen_data_identity(synthetic_root)
        payload = {
            "checkpoint_kind": "last",
            "configuration_hash": configuration_hash(config),
            "data_fingerprint": expected_data_fingerprint(
                synthetic_root, setup=spec.setup, fold=spec.fold
            ),
            "protocol_hash": protocol_sha256(synthetic_root),
            "seed": spec.seed,
            "next_epoch": 2,
            "best_epoch": 1,
            "best_validation_metric": 0.1,
            "patience_counter": 0,
            "best_model_state": None,
            "archive_sha256": frozen["archive_sha256"],
        }
        path = checkpoints / f"{spec.run_id}.last.pt"
        import torch

        torch.save(payload, path)
        result = reconcile_interrupted_artifacts(synthetic_root, artifacts)
        assert result["recoverable"][spec.run_id] == str(path)
        assert result["invalid"] == []

    def test_changed_frozen_data_identity_makes_a_run_ineligible(
        self, synthetic_root: Path
    ) -> None:
        provider = _provider(synthetic_root)

        def tampered(root: Path, spec: RunSpec, config_path: str) -> RunData:
            data = provider(root, spec, config_path)
            return RunData(**{**data.__dict__, "data_fingerprint": "0" * 64})

        manifest = execute_run(
            synthetic_root,
            _classical_spec(),
            artifact_root=synthetic_root / "artifacts" / "fi2010" / "baselines",
            run_data_provider=tampered,
        )
        assert manifest["eligible_for_confirmatory_report"] is False
        assert any(
            "frozen FI-2010 data identity" in reason for reason in manifest["exclusion_reasons"]
        )


def _make_synthetic_predictions(n: int = 200) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 3, size=n).astype(np.int64)
    y_pred = rng.integers(0, 3, size=n).astype(np.int64)
    proba = np.eye(3)[y_pred].astype(np.float64)
    proba = proba + rng.uniform(0, 0.05, size=proba.shape)
    proba /= proba.sum(axis=1, keepdims=True)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": proba,
        "sample_index": np.arange(n, dtype=np.int64),
        "source_file_id": np.full(n, 1, dtype=np.int64),
        "day_boundary_id": np.full(n, 2, dtype=np.int64),
    }


class TestPredictionArtifacts:
    """Round-trip and recomputation guarantees for the persisted arrays."""

    def test_prediction_artifact_roundtrip(self, tmp_path: Path) -> None:
        preds = _make_synthetic_predictions(100)
        path = tmp_path / "test.npz"
        save_prediction_artifact(path, **preds)  # type: ignore[arg-type]
        loaded = load_prediction_artifact(path)
        assert np.array_equal(loaded["y_true"], preds["y_true"])
        assert np.array_equal(loaded["y_pred"], preds["y_pred"])
        assert np.allclose(loaded["probabilities"], preds["probabilities"])
        assert tuple(loaded["class_order"]) == ("up", "stationary", "down")
        assert loaded["day_boundary_id"].dtype == np.int64

    def test_metric_recomputation_from_artifact(self) -> None:
        preds = _make_synthetic_predictions(200)
        direct = classification_metrics(preds["y_true"], preds["y_pred"], preds["probabilities"])
        from_artifact = recompute_metrics_from_artifact(preds)
        for key in ("macro_f1", "mcc", "accuracy", "balanced_accuracy", "nll", "brier", "ece"):
            assert float(direct[key]) == pytest.approx(float(from_artifact[key]), rel=1e-10)
        assert direct["confusion_matrix"] == from_artifact["confusion_matrix"]
