"""Run and report the ordered FI-2010 baseline matrix."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from deepbook.evaluation.classification import classification_metrics
from deepbook.evaluation.prediction import save_prediction_artifact, sha256_file
from deepbook.models.classical import (
    CausalMovementPersistence,
    MajorityClassifier,
    MultinomialLogistic,
    RandomForestBaseline,
)
from deepbook.models.deeplob import DeepLOB, parameter_count
from deepbook.models.mlplob import MLPLOB
from deepbook.training.data import FoldMatrices, labels_for_horizon, load_fold
from deepbook.training.fi2010 import (
    SegmentedWindowDataset,
    build_run_manifest,
    check_protocol_ancestry,
    chronological_training_validation_split,
    configuration_hash,
    protocol_sha256,
    report_fingerprint,
    resolve_protocol_commit,
    validate_run_manifest,
)
from deepbook.training.gates import tiny_batch_overfit_gate
from deepbook.training.loop import (
    evaluate_torch_classifier,
    fit_torch_classifier,
    predict_raw,
)

HORIZONS = (10, 20, 30, 50, 100)
SEEDS = (1337, 2027, 31415, 424242, 8675309)
CLASSICAL_MODELS = ("majority", "causal_persistence", "logistic_current_event", "random_forest")
_DETERMINISTIC_CLASSICAL_MODELS = ("majority", "causal_persistence", "logistic_current_event")
_STOCHASTIC_CLASSICAL_MODELS = ("random_forest",)
_NEURAL_MODELS = ("mlplob", "deeplob")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return commit, dirty


def _device_name(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _one_hot(indices: np.ndarray) -> np.ndarray:
    return np.asarray(np.eye(3, dtype=np.float64)[indices])


def _metrics_with_valid_predictions(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    valid = predictions >= 0
    if not np.any(valid):
        raise ValueError("causal predictor produced no valid predictions")
    return classification_metrics(labels[valid], predictions[valid], probabilities[valid])


def _timed_predict(
    predictor: Callable[[np.ndarray], np.ndarray], features: np.ndarray
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    predictions = np.asarray(predictor(features), dtype=np.int64)
    elapsed = time.perf_counter() - started
    return predictions, elapsed * 1000.0 / max(1, len(features))


def _classify_run(
    run_kind_flag: str,
    git_dirty: bool,
    smoke_flag: bool,
    status: str,
    termination_reason: str,
    actual_epochs: int | None,
    configured_epochs: int | None,
) -> tuple[str, bool, list[str]]:
    """Determine run_kind, eligibility, and exclusion_reasons for a run."""
    reasons: list[str] = []
    if smoke_flag:
        reasons.append("run explicitly marked as smoke")
    if git_dirty:
        reasons.append("git tree is dirty")
    if status != "completed":
        reasons.append(f"run status is {status}, not completed")
    if termination_reason not in {"early_stopping", "max_epochs", "completed", "not_applicable"}:
        reasons.append(f"invalid termination reason: {termination_reason}")

    eligible = not bool(reasons) and not smoke_flag
    run_kind = smoke_flag and "smoke" or (eligible and "confirmatory" or "smoke")
    return run_kind, eligible, reasons


def _base_manifest(
    *,
    root: Path,
    run_id: str,
    config: dict[str, Any],
    config_path: str,
    fold: int | None,
    day_group: str | None,
    horizon: int,
    seed: int,
    data: FoldMatrices,
    model: str,
    status: str,
    metrics: dict[str, Any],
    configured_epochs: int | None = None,
    actual_epochs: int | None = None,
    best_epoch: int | None = None,
    termination_reason: str = "not_applicable",
    smoke: bool = False,
    resumed: bool = False,
    resumed_from: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    commit, dirty = _git_state(root)
    protocol_commit_val = resolve_protocol_commit(root)
    protocol_hash = protocol_sha256(root)
    run_kind, eligible, reasons = _classify_run(
        smoke and "smoke" or "confirmatory",
        dirty,
        smoke,
        status,
        termination_reason,
        actual_epochs,
        configured_epochs,
    )
    # Verify protocol ancestry
    if eligible and not check_protocol_ancestry(root, protocol_commit_val):
        eligible = False
        reasons.append("code commit does not descend from protocol commit")
        run_kind = "smoke"

    started_utc = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return build_run_manifest(
        run_id=run_id,
        code_commit=commit,
        dirty=dirty,
        model=model,
        setup=fold is not None and "anchored_forward" or "first_seven_final_three",
        fold=fold,
        day_group=day_group,
        horizon=horizon,
        seed=seed,
        data_fingerprint=data.data_fingerprint,
        configuration_hash=configuration_hash(config),
        configuration_path=config_path,
        status=status,
        metrics=metrics,
        run_kind=run_kind,
        eligible_for_confirmatory_report=eligible,
        exclusion_reasons=reasons,
        protocol_commit=protocol_commit_val,
        protocol_sha256=protocol_hash,
        configured_max_epochs=configured_epochs,
        actual_epochs_completed=actual_epochs,
        best_epoch=best_epoch,
        termination_reason=termination_reason,
        resumed=resumed,
        resumed_from_run_id=resumed_from,
        archive_sha256=data.archive_sha256,
        training_file_sha256=data.training_file_sha256,
        testing_file_sha256=data.testing_file_sha256,
        environment=_environment(),
        labels_regenerated=False,
        test_set_used_for_selection=False,
        started_utc=started_utc,
        **details,
    )


# --- Classical run ---


def _classical_run(
    root: Path,
    model_name: str,
    fold: int,
    horizon: int,
    seed: int,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    config_path = "configs/experiments/fi2010/classical.yaml"
    full_config_path = root / config_path
    config = _load_yaml(full_config_path)
    data = load_fold(root, fold, experiment_config_path=Path(config_path))
    horizon_index = HORIZONS.index(horizon)
    train_features = data.training_features.T
    test_features = data.testing_features.T
    train_labels = labels_for_horizon(data.training_labels, horizon_index)
    test_labels = labels_for_horizon(data.testing_labels, horizon_index)
    train_indices, validation_indices = chronological_training_validation_split(
        len(train_labels),
        float(config["validation"]["single_day_fraction"]),
        int(config["validation"]["purge_events"]),
        int(config["validation"]["embargo_events"]),
    )
    started = time.perf_counter()
    model: Any
    if model_name == "majority":
        model = MajorityClassifier(float(config["models"]["majority"]["smoothing"]))
        model.fit(train_labels[train_indices])
        validation_predictions, validation_latency = _timed_predict(
            model.predict, train_features[validation_indices]
        )
        validation_probabilities = model.predict_proba(train_features[validation_indices])
        model.fit(train_labels)
        test_predictions, test_latency = _timed_predict(model.predict, test_features)
        test_probabilities = model.predict_proba(test_features)
        parameters = 0
    elif model_name == "causal_persistence":
        model = CausalMovementPersistence(horizon)
        combined = np.concatenate([train_labels, test_labels])
        all_predictions, latency = _timed_predict(model.predict, combined)
        validation_predictions = all_predictions[validation_indices]
        validation_probabilities = _one_hot(np.maximum(validation_predictions, 0))
        test_predictions = all_predictions[len(train_labels) :]
        test_probabilities = _one_hot(np.maximum(test_predictions, 0))
        validation_latency = latency
        test_latency = latency
        parameters = 0
    elif model_name == "logistic_current_event":
        model = MultinomialLogistic(
            max_iter=int(config["models"]["logistic_current_event"]["max_iter"]),
            random_state=seed,
        )
        model.fit(train_features[train_indices], train_labels[train_indices])
        validation_predictions, validation_latency = _timed_predict(
            model.predict, train_features[validation_indices]
        )
        validation_probabilities = model.predict_proba(train_features[validation_indices])
        model.fit(train_features, train_labels)
        test_predictions, test_latency = _timed_predict(model.predict, test_features)
        test_probabilities = model.predict_proba(test_features)
        parameters = int(model.coef_.size + model.intercept_.size)
    elif model_name == "random_forest":
        forest_config = config["models"]["random_forest"]
        model = RandomForestBaseline(
            n_estimators=int(forest_config["n_estimators"]),
            max_depth=int(forest_config["max_depth"]),
            min_samples_leaf=int(forest_config["min_samples_leaf"]),
            max_training_rows=int(forest_config["max_training_rows"]),
            random_state=seed,
        )
        model.fit(train_features[train_indices], train_labels[train_indices])
        validation_predictions, validation_latency = _timed_predict(
            model.predict, train_features[validation_indices]
        )
        validation_probabilities = model.predict_proba(train_features[validation_indices])
        model.fit(train_features, train_labels)
        test_predictions, test_latency = _timed_predict(model.predict, test_features)
        test_probabilities = model.predict_proba(test_features)
        parameters = int(sum(tree.tree_.node_count for tree in model.estimator.estimators_))
    else:
        raise ValueError(f"unknown classical model: {model_name}")

    # Remove invalid predictions for metrics only
    valid_test = test_predictions >= 0
    if not np.any(valid_test):
        raise ValueError("classical model produced no valid test predictions")
    test_metrics = classification_metrics(
        test_labels[valid_test], test_predictions[valid_test], test_probabilities[valid_test]
    )
    validation_metrics = _metrics_with_valid_predictions(
        train_labels[validation_indices], validation_predictions, validation_probabilities
    )
    completed_utc = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    run_id = f"{model_name}-anchored-forward-f{fold}-h{horizon}-s{seed}"
    # Save prediction artifact with real boundary IDs
    pred_path = root / "artifacts" / "fi2010" / "baselines" / "predictions" / f"{run_id}.npz"
    n_test = len(test_labels)
    save_prediction_artifact(
        pred_path,
        y_true=test_labels.astype(np.int64),
        y_pred=test_predictions.astype(np.int64),
        probabilities=test_probabilities.astype(np.float64),
        sample_index=np.arange(n_test, dtype=np.int64),
        source_file_id=np.full(n_test, fold, dtype=np.int64),
        day_boundary_id=np.full(n_test, fold, dtype=np.int64),
    )
    pred_hash = sha256_file(pred_path)
    return _base_manifest(
        root=root,
        run_id=run_id,
        config=config,
        config_path=config_path,
        fold=fold,
        day_group=None,
        horizon=horizon,
        seed=seed,
        data=data,
        model=model_name,
        status="completed",
        metrics={"validation": validation_metrics, "test": test_metrics},
        smoke=smoke,
        configured_epochs=None,
        actual_epochs=None,
        best_epoch=None,
        termination_reason="not_applicable",
        training_seconds=time.perf_counter() - started,
        total_wall_seconds=time.perf_counter() - started,
        inference_latency_ms_per_sample={"validation": validation_latency, "test": test_latency},
        parameter_count=parameters,
        peak_gpu_memory_bytes=0,
        tiny_batch_overfit={"status": "not_applicable"},
        checkpoint_round_trip={"status": "not_applicable"},
        checkpoint_sha256=None,
        checkpoint_path=None,
        prediction_sha256=pred_hash,
        prediction_path=str(pred_path),
        sample_count=n_test,
        class_order=["up", "stationary", "down"],
        command=" ".join(sys.argv),
        exit_code=0,
        completed_utc=completed_utc,
        device="cpu",
    )


# --- Neural run ---


def _neural_run(
    root: Path,
    model_name: str,
    fold: int,
    horizon: int,
    seed: int,
    device: str,
    max_epochs: int | None,
    *,
    smoke: bool = False,
    resume_from: str | None = None,
) -> dict[str, Any]:
    config_name = "mlplob" if model_name == "mlplob" else "deeplob"
    config_path = f"configs/experiments/fi2010/{config_name}.yaml"
    full_config_path = root / config_path
    config = _load_yaml(full_config_path)
    data = load_fold(root, fold, experiment_config_path=Path(config_path))
    horizon_index = HORIZONS.index(horizon)
    sequence_length = int(config["sequence_length"])
    train_labels = data.training_labels
    train_end_indices, validation_indices = chronological_training_validation_split(
        train_labels.shape[1],
        float(config["validation"].get("single_day_fraction", 0.2)),
        int(config["validation"]["purge_events"]),
        int(config["validation"]["embargo_events"]),
    )
    train_end = int(train_end_indices[-1] + 1)
    validation_start = int(validation_indices[0])
    train_dataset = SegmentedWindowDataset(
        [(data.training_lob[:, :train_end], train_labels[:, :train_end])],
        horizon_index,
        sequence_length,
    )
    validation_dataset = SegmentedWindowDataset(
        [(data.training_lob[:, validation_start:], train_labels[:, validation_start:])],
        horizon_index,
        sequence_length,
    )
    test_dataset = SegmentedWindowDataset(
        [(data.testing_lob, data.testing_labels)], horizon_index, sequence_length
    )
    model: Any
    if model_name == "mlplob":
        model = MLPLOB(
            input_shape=(sequence_length, 40),
            hidden_sizes=tuple(int(size) for size in config["model"]["hidden_sizes"]),
        )
    else:
        model = DeepLOB()
    selected_device = _device_name(device)
    tiny_gate: dict[str, Any] = {"status": "not_run"}
    if fold == 1 and horizon == 10 and seed == SEEDS[0]:
        factory: Callable[[], torch.nn.Module]

        if model_name == "mlplob":

            def factory() -> MLPLOB:
                return MLPLOB(
                    input_shape=(sequence_length, 40),
                    hidden_sizes=tuple(int(size) for size in config["model"]["hidden_sizes"]),
                )
        else:
            factory = DeepLOB
        tiny_gate = tiny_batch_overfit_gate(
            factory,
            input_shape=(sequence_length, 40),
            device=selected_device,
            seed=seed,
        )
    checkpoint_path = (
        root
        / "artifacts"
        / "fi2010"
        / "baselines"
        / "checkpoints"
        / f"{model_name}-f{fold}-h{horizon}-s{seed}.pt"
    )
    configured_epochs = max_epochs or int(config["training"]["max_epochs"])
    started = time.perf_counter()
    fit_result = fit_torch_classifier(
        model,
        train_dataset,
        validation_dataset,
        seed=seed,
        max_epochs=configured_epochs,
        patience=int(config["training"]["patience"]),
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
        device=selected_device,
        checkpoint_path=checkpoint_path,
        configuration_hash=configuration_hash(config),
        data_fingerprint=data.data_fingerprint,
        protocol_hash=protocol_sha256(root),
        resume_from=Path(resume_from) if resume_from else None,
    )
    test_metrics, test_latency = evaluate_torch_classifier(
        model,
        test_dataset,
        device=selected_device,
        batch_size=int(config["training"]["batch_size"]),
    )
    completed_utc = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    run_id = f"{model_name}-anchored-forward-f{fold}-h{horizon}-s{seed}"
    # Save prediction artifact
    model.eval()
    target_device = torch.device(selected_device)
    true_vals, pred_vals, probas, _ = predict_raw(
        model, test_dataset, target_device, int(config["training"]["batch_size"])
    )
    pred_path = root / "artifacts" / "fi2010" / "baselines" / "predictions" / f"{run_id}.npz"
    n_test = len(true_vals)
    save_prediction_artifact(
        pred_path,
        y_true=true_vals.astype(np.int64),
        y_pred=pred_vals.astype(np.int64),
        probabilities=probas.astype(np.float64),
        sample_index=np.arange(n_test, dtype=np.int64),
        source_file_id=np.full(n_test, fold, dtype=np.int64),
        day_boundary_id=np.full(n_test, fold, dtype=np.int64),
    )
    pred_hash = sha256_file(pred_path)
    chk_hash = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None

    termination = "early_stopping" if fit_result.best_epoch < configured_epochs else "max_epochs"
    return _base_manifest(
        root=root,
        run_id=run_id,
        config=config,
        config_path=config_path,
        fold=fold,
        day_group=None,
        horizon=horizon,
        seed=seed,
        data=data,
        model=model_name,
        status="completed",
        metrics={"validation": fit_result.validation_metrics, "test": test_metrics},
        configured_epochs=configured_epochs,
        actual_epochs=fit_result.best_epoch,
        best_epoch=fit_result.best_epoch,
        termination_reason=termination,
        smoke=smoke,
        resumed=bool(resume_from),
        resumed_from=resume_from,
        training_seconds=fit_result.training_seconds,
        total_wall_seconds=time.perf_counter() - started,
        inference_latency_ms_per_sample={
            "validation": fit_result.inference_latency_ms_per_sample,
            "test": test_latency,
        },
        parameter_count=parameter_count(model),
        peak_gpu_memory_bytes=fit_result.peak_gpu_memory_bytes,
        tiny_batch_overfit=tiny_gate,
        checkpoint_round_trip={
            "passed": fit_result.checkpoint_round_trip,
            "path": str(checkpoint_path),
        },
        checkpoint_sha256=chk_hash,
        checkpoint_path=str(checkpoint_path) if checkpoint_path.is_file() else None,
        prediction_sha256=pred_hash,
        prediction_path=str(pred_path),
        sample_count=n_test,
        class_order=["up", "stationary", "down"],
        device=selected_device,
        command=" ".join(sys.argv),
        exit_code=0,
        completed_utc=completed_utc,
    )


# --- Persistence ---


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record_run(root: Path, manifest: dict[str, Any]) -> Path:
    validate_run_manifest(manifest, root / "data_contracts" / "fi2010_run_manifest.schema.json")
    path = root / "artifacts" / "fi2010" / "baselines" / "runs" / f"{manifest['run_id']}.json"
    _write_json(path, manifest)
    return path


# --- Per-model seed configuration ---


def _seeds_for_model(model: str) -> tuple[int, ...]:
    """Return the declared seed tuple for a given model."""
    if model in _DETERMINISTIC_CLASSICAL_MODELS:
        return (SEEDS[0],)  # only seed 1337 for deterministic models
    return SEEDS  # all five for stochastic models and neural models


# --- Run index (authoritative, deterministic) ---


def generate_run_index(root: Path) -> dict[str, Any]:
    """Generate a deterministic run index from run manifests with audit checks."""
    run_root = root / "artifacts" / "fi2010" / "baselines" / "runs"
    checkpoint_root = root / "artifacts" / "fi2010" / "baselines" / "checkpoints"
    prediction_root = root / "artifacts" / "fi2010" / "baselines" / "predictions"

    manifests_by_id: dict[str, dict[str, Any]] = {}
    run_ids_seen: set[str] = set()
    if run_root.is_dir():
        for manifest_path in sorted(run_root.glob("*.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rid = manifest["run_id"]
            if rid in run_ids_seen:
                raise ValueError(f"duplicate run_id in manifests: {rid}")
            run_ids_seen.add(rid)
            manifests_by_id[rid] = manifest

    # Detect orphans
    orphan_checkpoints: list[str] = []
    if checkpoint_root.is_dir():
        orphan_checkpoints = sorted(
            cp.name for cp in checkpoint_root.iterdir() if cp.suffix in {".pt", ".pth"}
        )
    orphan_predictions: list[str] = []
    if prediction_root.is_dir():
        orphan_predictions = sorted(
            pp.name for pp in prediction_root.iterdir() if pp.suffix == ".npz"
        )

    # Planned Cartesian coverage per model configuration
    all_models = CLASSICAL_MODELS + _NEURAL_MODELS
    planned_cells: set[str] = set()
    for model in all_models:
        model_seeds = _seeds_for_model(model)
        for fold in range(1, 10):
            for horizon in HORIZONS:
                for seed_val in model_seeds:
                    planned_cells.add(f"{model}-anchored-forward-f{fold}-h{horizon}-s{seed_val}")

    # Classify existing runs
    completed_confirmatory: set[str] = set()
    completed_smoke: set[str] = set()
    failed: set[str] = set()
    interrupted: set[str] = set()
    duplicates: list[str] = []

    for rid, m in manifests_by_id.items():
        status = m.get("status", "")
        if status == "completed":
            if m.get("eligible_for_confirmatory_report"):
                completed_confirmatory.add(rid)
            else:
                completed_smoke.add(rid)
        elif status == "failed":
            failed.add(rid)
        elif status == "interrupted":
            interrupted.add(rid)

    # Cross-reference checkpoints and predictions
    for rid in sorted(manifests_by_id):
        m = manifests_by_id[rid]
        cp_path = m.get("checkpoint_path", "")
        if cp_path:
            cp_name = Path(cp_path).name
            if cp_name in orphan_checkpoints:
                orphan_checkpoints.remove(cp_name)
        pp_path = m.get("prediction_path", "")
        if pp_path:
            pp_name = Path(pp_path).name
            if pp_name in orphan_predictions:
                orphan_predictions.remove(pp_name)

    # Build index entries
    index_entries: list[dict[str, Any]] = []
    for rid in sorted(manifests_by_id):
        m = manifests_by_id[rid]
        index_entries.append(
            {
                "run_id": rid,
                "run_kind": m.get("run_kind", "smoke"),
                "eligible_for_confirmatory_report": m.get(
                    "eligible_for_confirmatory_report", False
                ),
                "exclusion_reasons": m.get("exclusion_reasons", []),
                "model": m.get("model", ""),
                "setup": m.get("setup", ""),
                "fold": m.get("fold"),
                "day_group": m.get("day_group"),
                "horizon": m.get("horizon"),
                "seed": m.get("seed"),
                "status": m.get("status", ""),
                "exit_code": m.get("exit_code"),
                "termination_reason": m.get("termination_reason"),
                "protocol_commit": m.get("protocol_commit"),
                "protocol_sha256": m.get("protocol_sha256"),
                "code_commit": m.get("code_commit"),
                "git_tree_dirty": m.get("git_tree_dirty", m.get("dirty", False)),
                "configuration_hash": m.get("configuration_hash", ""),
                "data_fingerprint": m.get("data_fingerprint", ""),
                "checkpoint_sha256": m.get("checkpoint_sha256"),
                "prediction_sha256": m.get("prediction_sha256"),
                "configured_max_epochs": m.get("configured_max_epochs"),
                "actual_epochs_completed": m.get("actual_epochs_completed"),
                "best_epoch": m.get("best_epoch"),
                "resumed": m.get("resumed", False),
                "started_utc": m.get("started_utc", ""),
                "completed_utc": m.get("completed_utc", ""),
            }
        )

    # Derive timestamp from latest manifest completion or protocol freeze time
    latest_completed = ""
    for m in manifests_by_id.values():
        cu = m.get("completed_utc", "")
        if cu > latest_completed:
            latest_completed = cu

    return {
        "schema_version": 1,
        "generated_from_manifests_utc": latest_completed or None,
        "total_manifest_count": len(manifests_by_id),
        "planned_cells": sorted(planned_cells),
        "completed_confirmatory": sorted(completed_confirmatory),
        "completed_smoke": sorted(completed_smoke),
        "failed": sorted(failed),
        "interrupted": sorted(interrupted),
        "missing_manifests": sorted(planned_cells - set(manifests_by_id)),
        "duplicate_run_ids": sorted(duplicates),
        "orphan_checkpoints": orphan_checkpoints,
        "orphan_predictions": orphan_predictions,
        "runs": index_entries,
    }


# --- Report generation (deterministic) ---


def write_report(root: Path) -> tuple[Path, Path]:
    """Aggregate ignored run manifests into deterministic JSON and Markdown reports."""
    run_root = root / "artifacts" / "fi2010" / "baselines" / "runs"
    manifests: list[dict[str, Any]] = []
    if run_root.is_dir():
        manifests = sorted(
            (json.loads(p.read_text(encoding="utf-8")) for p in run_root.glob("*.json")),
            key=lambda m: m.get("run_id", ""),
        )
    confirmatory = [
        m
        for m in manifests
        if m.get("eligible_for_confirmatory_report") and m.get("status") == "completed"
    ]
    smoke = [
        m
        for m in manifests
        if not m.get("eligible_for_confirmatory_report", False)
        or m.get("run_kind") in {"smoke", None}
    ]
    # Aggregate by model/setup/horizon
    grouped: dict[str, list[float]] = defaultdict(list)
    for manifest in confirmatory:
        test_metrics = manifest.get("metrics", {}).get("test", {})
        if "macro_f1" in test_metrics:
            key = f"{manifest['model']}|{manifest.get('setup', '')}|h{manifest['horizon']}"
            grouped[key].append(float(test_metrics["macro_f1"]))

    summary: dict[str, dict[str, Any]] = {}
    for key, macro_f1_values in sorted(grouped.items()):
        entry: dict[str, Any] = {
            "count": len(macro_f1_values),
            "mean_macro_f1": float(np.mean(macro_f1_values)),
        }
        if len(macro_f1_values) >= 2:
            entry["std_macro_f1"] = float(np.std(macro_f1_values))
        else:
            entry["std_macro_f1"] = "unavailable"
        summary[key] = entry

    tiny_gates = sorted(
        (
            {
                "run_id": m["run_id"],
                "model": m["model"],
                "result": m.get("tiny_batch_overfit", {}),
            }
            for m in manifests
            if m.get("tiny_batch_overfit", {}).get("status") not in {"not_run", "not_applicable"}
        ),
        key=lambda g: g["run_id"],
    )
    smoke_exclusion = sorted(
        (
            {
                "run_id": m["run_id"],
                "run_kind": m.get("run_kind", "smoke"),
                "exclusion_reasons": m.get("exclusion_reasons", []),
            }
            for m in smoke
        ),
        key=lambda s: s["run_id"],
    )

    run_index = generate_run_index(root)

    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "local FI-2010 reproduction; not publisher-verified",
        "data_contract": {
            "archive_sha256": "cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664",
            "source_manifest": "data/interim/fi2010/fi2010_source_manifest.json",
            "split_manifest": "data/interim/fi2010/fi2010_split_manifest.json",
            "feature_rows": 40,
            "supplied_non_label_rows": 144,
            "label_rows": 5,
            "window_length": 100,
            "boundary_contract": (
                "published fold/file boundaries preserved; no finer day or instrument "
                "boundaries inferred"
            ),
            "labels_regenerated": False,
            "test_set_used_for_selection": False,
        },
        "reference_comparison": {
            "paper": "arxiv:1808.03668v6",
            "official_repository_commit": "ff14d7c2fd38bdfc143389786993d0f0236d4eb8",
            "local_results_are_not_publisher_verified": True,
        },
        "confirmatory_runs": confirmatory,
        "smoke_runs": smoke,
        "smoke_exclusion_summary": smoke_exclusion,
        "summary": summary,
        "gates": {
            "tiny_batch_overfit": tiny_gates,
            "all_passed": all(bool(g["result"].get("passed")) for g in tiny_gates),
        },
        "failed_runs": [m for m in manifests if m.get("status") == "failed"],
        "resumed_runs": [m for m in manifests if m.get("resumed")],
        "run_index": run_index,
    }
    report["report_fingerprint"] = report_fingerprint(report)
    json_path = root / "reports" / "results" / "fi2010_baseline_reproduction.json"
    markdown_path = root / "reports" / "results" / "fi2010_baseline_reproduction.md"
    _write_json(json_path, report)

    confirmatory_count = len(confirmatory)
    smoke_count = len(smoke)
    failed_count = len(report["failed_runs"])
    markdown_lines = [
        "# FI-2010 Baseline Reproduction Results",
        "",
        (
            "Local results are reproductions under the tracked protocol; they are not "
            "publisher-verified benchmark values."
        ),
        "",
        f"- Report fingerprint: `{report['report_fingerprint']}`",
        f"- Confirmatory runs: `{confirmatory_count}`",
        f"- Excluded smoke runs: `{smoke_count}`",
        f"- Failed runs: `{failed_count}`",
        f"- Resumed runs: `{len(report['resumed_runs'])}`",
        "- Labels regenerated: `False`",
        "- Test-set-driven selection: `False`",
        "",
    ]

    if confirmatory_count == 0:
        markdown_lines.extend(
            [
                "**Confirmatory matrix not started.**",
                "",
            ]
        )

    if smoke_count > 0:
        markdown_lines.extend(
            [
                "## Engineering smoke runs excluded from confirmatory analysis",
                "",
                "| Run ID | Reasons |",
                "|---|---|",
            ]
        )
        for sm in smoke_exclusion[:20]:
            reasons_str = "; ".join(sm.get("exclusion_reasons", [])) or "pre-freeze / legacy"
            markdown_lines.append(f"| {sm['run_id']} | {reasons_str} |")
        if smoke_count > 20:
            markdown_lines.append(f"| ... and {smoke_count - 20} more | |")
        markdown_lines.append("")

    if confirmatory_count > 0:
        markdown_lines.extend(
            [
                "## Confirmatory macro-F1 summary",
                "",
                "| Model/Setup/Horizon | Runs | Mean | Std |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, values in summary.items():
            std_str = (
                f"{values['std_macro_f1']:.6f}"
                if isinstance(values["std_macro_f1"], float)
                else str(values["std_macro_f1"])
            )
            markdown_lines.append(
                f"| {key} | {values['count']} | {values['mean_macro_f1']:.6f} | {std_str} |"
            )

    markdown_lines.extend(
        [
            "",
            "Full metrics, confusion matrices, timing, memory, manifests, failures, "
            "and commands are in the JSON report.",
            "",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    return json_path, markdown_path


# --- CLI ---


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run audited FI-2010 baselines")
    parser.add_argument("--model", choices=CLASSICAL_MODELS + _NEURAL_MODELS)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--horizon", type=int, choices=HORIZONS, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument(
        "--matrix", action="store_true", help="run all folds, horizons, and required seeds"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="mark this run as smoke (nonconfirmatory)"
    )
    parser.add_argument("--report-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run selected baseline combinations and return a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    if args.report_only:
        paths = write_report(root)
        print(f"Report JSON: {paths[0]}")
        print(f"Report Markdown: {paths[1]}")
        return 0
    if args.model is None:
        parser.error("--model is required unless --report-only is supplied")
    folds = range(1, 10) if args.matrix else (args.fold,)
    horizons = HORIZONS if args.matrix else (args.horizon,)
    model_seeds = _seeds_for_model(args.model)
    seeds = model_seeds if args.matrix and len(model_seeds) > 1 else (args.seed,)
    failures = 0
    for fold in folds:
        for horizon in horizons:
            for seed in seeds:
                run_id = f"{args.model}-anchored-forward-f{fold}-h{horizon}-s{seed}"
                try:
                    if args.model in CLASSICAL_MODELS:
                        manifest = _classical_run(
                            root, args.model, fold, horizon, seed, smoke=args.smoke
                        )
                    else:
                        manifest = _neural_run(
                            root,
                            args.model,
                            fold,
                            horizon,
                            seed,
                            args.device,
                            args.max_epochs,
                            smoke=args.smoke,
                        )
                except Exception as error:
                    failures += 1
                    try:
                        data = load_fold(root, fold)
                        config_path_part = (
                            "classical.yaml"
                            if args.model in CLASSICAL_MODELS
                            else f"{args.model}.yaml"
                        )
                        config_path = f"configs/experiments/fi2010/{config_path_part}"
                        config = _load_yaml(root / config_path)
                        manifest = _base_manifest(
                            root=root,
                            run_id=run_id,
                            config=config,
                            config_path=config_path,
                            fold=fold,
                            day_group=None,
                            horizon=horizon,
                            seed=seed,
                            data=data,
                            model=args.model,
                            status="failed",
                            metrics={},
                            smoke=args.smoke,
                            termination_reason="failed",
                            error=f"{type(error).__name__}: {error}",
                            command=" ".join(sys.argv),
                            exit_code=1,
                        )
                    except Exception as nested_error:
                        print(f"ERROR {run_id}: {type(error).__name__}: {error}", file=sys.stderr)
                        print(
                            "ERROR recording failure: "
                            f"{type(nested_error).__name__}: {nested_error}",
                            file=sys.stderr,
                        )
                        continue
                path = _record_run(root, manifest)
                print(f"Run {manifest['status']}: {path}")
    index = generate_run_index(root)
    index_path = root / "artifacts" / "fi2010" / "baselines" / "run_index.json"
    _write_json(index_path, index)
    print(f"Run index: {index_path}")
    report_paths = write_report(root)
    print(f"Report JSON: {report_paths[0]}")
    print(f"Report Markdown: {report_paths[1]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
