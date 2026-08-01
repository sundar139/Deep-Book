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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from deepbook.evaluation.classification import classification_metrics
from deepbook.evaluation.prediction import (
    load_prediction_artifact,
    save_prediction_artifact,
    sha256_file,
)
from deepbook.models.classical import (
    CausalMovementPersistence,
    MajorityClassifier,
    MultinomialLogistic,
    RandomForestBaseline,
)
from deepbook.models.deeplob import DeepLOB, parameter_count
from deepbook.models.mlplob import MLPLOB
from deepbook.training.data import labels_for_horizon, load_day_group, load_fold
from deepbook.training.fi2010 import (
    DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
    SETUP_ANCHORED_FORWARD,
    SETUP_FIRST_SEVEN_FINAL_THREE,
    SegmentedWindowDataset,
    build_run_manifest,
    check_protocol_ancestry,
    chronological_training_validation_split,
    combined_testing_sha256,
    configuration_hash,
    expected_archive_sha256,
    expected_data_fingerprint,
    frozen_data_identity,
    protocol_sha256,
    report_fingerprint,
    resolve_protocol_commit,
    validate_run_manifest,
)
from deepbook.training.gates import tiny_batch_overfit_gate
from deepbook.training.loop import fit_torch_classifier, predict_raw

HORIZONS = (10, 20, 30, 50, 100)
SEEDS = (1337, 2027, 31415, 424242, 8675309)
CLASSICAL_MODELS = ("majority", "causal_persistence", "logistic_current_event", "random_forest")
_DETERMINISTIC_CLASSICAL_MODELS = ("majority", "causal_persistence", "logistic_current_event")
_NEURAL_MODELS = ("mlplob", "deeplob")
ALL_MODELS = CLASSICAL_MODELS + _NEURAL_MODELS
SETUPS = (SETUP_ANCHORED_FORWARD, SETUP_FIRST_SEVEN_FINAL_THREE)
CLASS_ORDER = ["up", "stationary", "down"]


# --- Run identity -----------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """One planned matrix cell."""

    model: str
    setup: str
    horizon: int
    seed: int
    fold: int | None = None
    day_group: str | None = None

    @property
    def cell(self) -> str:
        """Return the setup-qualified cell label used for run identifiers."""
        if self.setup == SETUP_ANCHORED_FORWARD:
            return f"anchored-forward-f{self.fold}"
        return f"first-seven-final-three-{str(self.day_group).replace('_', '-')}"

    @property
    def run_id(self) -> str:
        """Return the deterministic run identifier for this cell."""
        return f"{self.model}-{self.cell}-h{self.horizon}-s{self.seed}"


def _seeds_for_model(model: str) -> tuple[int, ...]:
    """Return the declared seed tuple for a given model."""
    if model in _DETERMINISTIC_CLASSICAL_MODELS:
        return (SEEDS[0],)
    return SEEDS


def planned_run_specs(root: Path) -> tuple[RunSpec, ...]:
    """Return every planned matrix cell, derived from the frozen configuration."""
    frozen = frozen_data_identity(root)
    folds = tuple(sorted(int(fold) for fold in frozen["folds"]))
    day_groups = tuple(sorted(str(group) for group in frozen["day_groups"]))
    specs: list[RunSpec] = []
    for model in ALL_MODELS:
        for seed in _seeds_for_model(model):
            for horizon in HORIZONS:
                for fold in folds:
                    specs.append(
                        RunSpec(
                            model=model,
                            setup=SETUP_ANCHORED_FORWARD,
                            horizon=horizon,
                            seed=seed,
                            fold=fold,
                        )
                    )
                for day_group in day_groups:
                    specs.append(
                        RunSpec(
                            model=model,
                            setup=SETUP_FIRST_SEVEN_FINAL_THREE,
                            horizon=horizon,
                            seed=seed,
                            day_group=day_group,
                        )
                    )
    return tuple(sorted(specs, key=lambda spec: spec.run_id))


# --- Run data ---------------------------------------------------------------


@dataclass(frozen=True)
class SourceSegment:
    """One independent audited test source file with its own boundary identity."""

    day_index: int
    source_fold: int
    file_sha256: str
    lob: np.ndarray
    features: np.ndarray
    labels: np.ndarray

    @property
    def observation_count(self) -> int:
        """Return the observation count of this segment."""
        return int(self.labels.shape[1])


@dataclass(frozen=True)
class RunData:
    """Setup-independent training matrix plus independent test segments."""

    setup: str
    fold: int | None
    day_group: str | None
    training_lob: np.ndarray
    training_features: np.ndarray
    training_labels: np.ndarray
    test_segments: tuple[SourceSegment, ...]
    archive_sha256: str
    training_file_sha256: str
    testing_file_sha256: str
    testing_file_sha256_by_day: dict[str, str] = field(default_factory=dict)
    data_fingerprint: str = ""

    @property
    def day_index_map(self) -> dict[str, dict[str, Any]]:
        """Return the persisted mapping from day identifier to audited source file."""
        return {
            str(segment.day_index): {
                "source_fold": segment.source_fold,
                "file_sha256": segment.file_sha256,
                "observations": segment.observation_count,
            }
            for segment in self.test_segments
        }


def build_run_data(root: Path, spec: RunSpec, config_path: str) -> RunData:
    """Load audited matrices for one run, identically shaped for either setup."""
    if spec.setup == SETUP_ANCHORED_FORWARD:
        if spec.fold is None:
            raise ValueError("anchored_forward requires a fold")
        fold_data = load_fold(root, spec.fold, experiment_config_path=Path(config_path))
        day_index = int(frozen_data_identity(root)["folds"][spec.fold]["testing_day_index"])
        segment = SourceSegment(
            day_index=day_index,
            source_fold=spec.fold,
            file_sha256=fold_data.testing_file_sha256,
            lob=fold_data.testing_lob,
            features=fold_data.testing_features,
            labels=fold_data.testing_labels,
        )
        return RunData(
            setup=spec.setup,
            fold=spec.fold,
            day_group=None,
            training_lob=fold_data.training_lob,
            training_features=fold_data.training_features,
            training_labels=fold_data.training_labels,
            test_segments=(segment,),
            archive_sha256=fold_data.archive_sha256,
            training_file_sha256=fold_data.training_file_sha256,
            testing_file_sha256=fold_data.testing_file_sha256,
            testing_file_sha256_by_day={str(day_index): fold_data.testing_file_sha256},
            data_fingerprint=fold_data.data_fingerprint,
        )
    group = load_day_group(root, str(spec.day_group), experiment_config_path=Path(config_path))
    segments = tuple(
        SourceSegment(
            day_index=day.day_index,
            source_fold=day.source_fold,
            file_sha256=day.file_sha256,
            lob=day.lob,
            features=day.features,
            labels=day.labels,
        )
        for day in group.test_days
    )
    return RunData(
        setup=spec.setup,
        fold=None,
        day_group=group.day_group,
        training_lob=group.training_lob,
        training_features=group.training_features,
        training_labels=group.training_labels,
        test_segments=segments,
        archive_sha256=group.archive_sha256,
        training_file_sha256=group.training_file_sha256,
        testing_file_sha256=group.testing_file_sha256,
        testing_file_sha256_by_day=group.testing_file_sha256_by_day,
        data_fingerprint=group.data_fingerprint,
    )


RunDataProvider = Callable[[Path, RunSpec, str], RunData]


# --- Environment ------------------------------------------------------------


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _one_hot(indices: np.ndarray) -> np.ndarray:
    return np.asarray(np.eye(3, dtype=np.float64)[indices])


def _timed_predict(
    predictor: Callable[[np.ndarray], np.ndarray], features: np.ndarray
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    predictions = np.asarray(predictor(features), dtype=np.int64)
    elapsed = time.perf_counter() - started
    return predictions, elapsed * 1000.0 / max(1, len(features))


def _metrics_with_valid_predictions(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    valid = predictions >= 0
    if not np.any(valid):
        raise ValueError("predictor produced no valid predictions")
    return classification_metrics(labels[valid], predictions[valid], probabilities[valid])


# --- Confirmatory eligibility ----------------------------------------------


@dataclass(frozen=True)
class EligibilityContext:
    """Everything eligibility depends on, gathered after artifacts exist."""

    model: str
    status: str
    termination_reason: str
    smoke: bool
    git_dirty: bool
    protocol_ancestry_ok: bool
    protocol_hash_matches: bool
    configuration_hash_matches: bool
    data_fingerprint_matches: bool
    archive_sha256_matches: bool
    prediction_valid: bool
    checkpoint_valid: bool
    configured_max_epochs: int | None
    actual_epochs_completed: int | None
    best_epoch: int | None


def classify_run(context: EligibilityContext) -> tuple[str, bool, list[str]]:
    """Return run_kind, eligibility, and every exclusion reason for one run."""
    reasons: list[str] = []
    if context.smoke:
        reasons.append("run explicitly marked as smoke")
    if context.git_dirty:
        reasons.append("git tree is dirty")
    if context.status != "completed":
        reasons.append(f"run status is {context.status}, not completed")
    if not context.protocol_ancestry_ok:
        reasons.append("code commit does not descend from protocol commit")
    if not context.protocol_hash_matches:
        reasons.append("protocol hash does not match the frozen contract")
    if not context.configuration_hash_matches:
        reasons.append("configuration hash does not match its configuration file")
    if not context.data_fingerprint_matches:
        reasons.append("data fingerprint does not match the frozen FI-2010 data identity")
    if not context.archive_sha256_matches:
        reasons.append("archive digest does not match the frozen authoritative archive")
    if not context.prediction_valid:
        reasons.append("prediction artifact is missing or invalid")

    is_neural = context.model in _NEURAL_MODELS
    if is_neural:
        if context.termination_reason not in {"early_stopping", "max_epochs"}:
            reasons.append(f"invalid neural termination reason: {context.termination_reason}")
        if not context.checkpoint_valid:
            reasons.append("best-model checkpoint is missing or invalid")
        configured = context.configured_max_epochs
        actual = context.actual_epochs_completed
        best = context.best_epoch
        if configured is None or configured < 1:
            reasons.append("configured_max_epochs must be a positive integer for neural runs")
        elif actual is None or actual < 1:
            reasons.append("actual_epochs_completed must be at least 1 for neural runs")
        elif actual > configured:
            reasons.append(
                f"actual_epochs_completed={actual} exceeds configured_max_epochs={configured}"
            )
        elif best is None or best < 1:
            reasons.append("best_epoch must be at least 1 for neural runs")
        elif best > actual:
            reasons.append(f"best_epoch={best} exceeds actual_epochs_completed={actual}")
    else:
        if context.termination_reason != "not_applicable":
            reasons.append(
                f"classical termination reason must be not_applicable, "
                f"got {context.termination_reason}"
            )
        if context.configured_max_epochs is not None:
            reasons.append("classical runs must record a null configured_max_epochs")
        if context.actual_epochs_completed is not None:
            reasons.append("classical runs must record a null actual_epochs_completed")
        if context.best_epoch is not None:
            reasons.append("classical runs must record a null best_epoch")

    eligible = not reasons
    run_kind = "confirmatory" if eligible else "smoke"
    return run_kind, eligible, reasons


def _prediction_is_valid(path: Path | None, expected_samples: int | None) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        payload = load_prediction_artifact(path)
    except (ValueError, OSError):
        return False
    if expected_samples is not None and int(payload["y_true"].shape[0]) != expected_samples:
        return False
    return int(payload["y_true"].shape[0]) > 0


def _checkpoint_is_valid(path: Path | None, recorded_digest: str | None) -> bool:
    if path is None or not path.is_file() or not recorded_digest:
        return False
    return sha256_file(path) == recorded_digest


# --- Prediction assembly ----------------------------------------------------


@dataclass
class PredictionBundle:
    """Per-sample prediction arrays with real source and day identifiers."""

    y_true: list[np.ndarray] = field(default_factory=list)
    y_pred: list[np.ndarray] = field(default_factory=list)
    probabilities: list[np.ndarray] = field(default_factory=list)
    sample_index: list[np.ndarray] = field(default_factory=list)
    source_file_id: list[np.ndarray] = field(default_factory=list)
    day_boundary_id: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        segment: SourceSegment,
        true_values: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        """Append one independently evaluated segment, keeping its identity."""
        count = int(true_values.shape[0])
        self.y_true.append(np.asarray(true_values, dtype=np.int64))
        self.y_pred.append(np.asarray(predictions, dtype=np.int64))
        self.probabilities.append(np.asarray(probabilities, dtype=np.float64))
        self.sample_index.append(np.arange(count, dtype=np.int64))
        self.source_file_id.append(np.full(count, segment.source_fold, dtype=np.int64))
        self.day_boundary_id.append(np.full(count, segment.day_index, dtype=np.int64))

    def concatenate(self) -> dict[str, np.ndarray]:
        """Concatenate completed per-day arrays into one ordered artifact payload."""
        if not self.y_true:
            raise ValueError("no prediction segments were produced")
        return {
            "y_true": np.concatenate(self.y_true),
            "y_pred": np.concatenate(self.y_pred),
            "probabilities": np.concatenate(self.probabilities),
            "sample_index": np.concatenate(self.sample_index),
            "source_file_id": np.concatenate(self.source_file_id),
            "day_boundary_id": np.concatenate(self.day_boundary_id),
        }


def _per_day_metrics(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    per_day: dict[str, Any] = {}
    for day in sorted(set(payload["day_boundary_id"].tolist())):
        mask = (payload["day_boundary_id"] == day) & (payload["y_pred"] >= 0)
        if not np.any(mask):
            continue
        per_day[str(int(day))] = classification_metrics(
            payload["y_true"][mask], payload["y_pred"][mask], payload["probabilities"][mask]
        )
    return per_day


# --- Classical and neural fitting ------------------------------------------


def _fit_classical(
    model_name: str, config: dict[str, Any], seed: int, horizon: int
) -> tuple[Any, int]:
    if model_name == "majority":
        return MajorityClassifier(float(config["models"]["majority"]["smoothing"])), 0
    if model_name == "causal_persistence":
        return CausalMovementPersistence(horizon), 0
    if model_name == "logistic_current_event":
        return (
            MultinomialLogistic(
                max_iter=int(config["models"]["logistic_current_event"]["max_iter"]),
                random_state=seed,
            ),
            0,
        )
    if model_name == "random_forest":
        forest = config["models"]["random_forest"]
        return (
            RandomForestBaseline(
                n_estimators=int(forest["n_estimators"]),
                max_depth=int(forest["max_depth"]),
                min_samples_leaf=int(forest["min_samples_leaf"]),
                max_training_rows=int(forest["max_training_rows"]),
                random_state=seed,
            ),
            0,
        )
    raise ValueError(f"unknown classical model: {model_name}")


def _classical_predictions(
    model: Any, model_name: str, segment_labels: np.ndarray, segment_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict one independent segment without reading any other segment."""
    if model_name == "causal_persistence":
        predictions, latency = _timed_predict(model.predict, segment_labels)
        return predictions, _one_hot(np.maximum(predictions, 0)), latency
    predictions, latency = _timed_predict(model.predict, segment_features)
    return predictions, model.predict_proba(segment_features), latency


# --- Orchestration ----------------------------------------------------------


@dataclass(frozen=True)
class RunPaths:
    """Ignored artifact locations for one run."""

    runs: Path
    predictions: Path
    checkpoints: Path
    index: Path


def run_paths(artifact_root: Path) -> RunPaths:
    """Return the ignored artifact layout rooted at one baselines directory."""
    return RunPaths(
        runs=artifact_root / "runs",
        predictions=artifact_root / "predictions",
        checkpoints=artifact_root / "checkpoints",
        index=artifact_root / "run_index.json",
    )


def default_artifact_root(root: Path) -> Path:
    """Return the repository's ignored baseline artifact root."""
    return root / "artifacts" / "fi2010" / "baselines"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record_run(root: Path, artifact_root: Path, manifest: dict[str, Any]) -> Path:
    validate_run_manifest(manifest, root / "data_contracts" / "fi2010_run_manifest.schema.json")
    path = run_paths(artifact_root).runs / f"{manifest['run_id']}.json"
    _write_json(path, manifest)
    return path


def execute_run(
    root: Path,
    spec: RunSpec,
    *,
    artifact_root: Path | None = None,
    device: str = "cpu",
    max_epochs: int | None = None,
    smoke: bool = False,
    resume_from: Path | None = None,
    run_data_provider: RunDataProvider | None = None,
) -> dict[str, Any]:
    """Execute one matrix cell end to end and return its validated manifest.

    This is the single production orchestration path. The CLI and the integration
    tests both call it; only ``artifact_root`` and ``run_data_provider`` are
    injected, so no production behavior is reimplemented anywhere else.
    """
    artifacts = artifact_root if artifact_root is not None else default_artifact_root(root)
    paths = run_paths(artifacts)
    provider = run_data_provider if run_data_provider is not None else build_run_data
    config_name = "classical" if spec.model in CLASSICAL_MODELS else spec.model
    config_path = f"configs/experiments/fi2010/{config_name}.yaml"
    config = _load_yaml(root / config_path)
    started_utc = _utc_now()
    started = time.perf_counter()
    protocol_hash_at_start = protocol_sha256(root)

    data = provider(root, spec, config_path)
    horizon_index = HORIZONS.index(spec.horizon)
    if spec.model in CLASSICAL_MODELS:
        outcome = _run_classical(data, spec, config, horizon_index)
    else:
        outcome = _run_neural(
            root,
            data,
            spec,
            config,
            horizon_index,
            device=_device_name(device),
            max_epochs=max_epochs,
            checkpoint_root=paths.checkpoints,
            resume_from=resume_from,
        )

    payload = outcome["predictions"]
    prediction_path = paths.predictions / f"{spec.run_id}.npz"
    save_prediction_artifact(
        prediction_path,
        y_true=payload["y_true"],
        y_pred=payload["y_pred"],
        probabilities=payload["probabilities"],
        sample_index=payload["sample_index"],
        source_file_id=payload["source_file_id"],
        day_boundary_id=payload["day_boundary_id"],
    )
    prediction_digest = sha256_file(prediction_path)
    sample_count = int(payload["y_true"].shape[0])
    best_checkpoint_path: Path | None = outcome.get("best_checkpoint_path")
    last_checkpoint_path: Path | None = outcome.get("last_checkpoint_path")
    best_digest = (
        sha256_file(best_checkpoint_path)
        if best_checkpoint_path is not None and best_checkpoint_path.is_file()
        else None
    )
    last_digest = (
        sha256_file(last_checkpoint_path)
        if last_checkpoint_path is not None and last_checkpoint_path.is_file()
        else None
    )

    commit, dirty = _git_state(root)
    protocol_commit_value = resolve_protocol_commit(root)
    protocol_hash = protocol_sha256(root)
    configuration_digest = configuration_hash(config)
    expected_fingerprint = expected_data_fingerprint(
        root, setup=spec.setup, fold=spec.fold, day_group=spec.day_group
    )
    context = EligibilityContext(
        model=spec.model,
        status="completed",
        termination_reason=str(outcome["termination_reason"]),
        smoke=smoke,
        git_dirty=dirty,
        protocol_ancestry_ok=check_protocol_ancestry(root, protocol_commit_value),
        # Both re-read from disk after the run, so a mid-run edit to the frozen
        # contract or the experiment configuration invalidates eligibility.
        protocol_hash_matches=protocol_hash_at_start == protocol_hash,
        configuration_hash_matches=(
            configuration_digest == configuration_hash(_load_yaml(root / config_path))
        ),
        data_fingerprint_matches=data.data_fingerprint == expected_fingerprint,
        archive_sha256_matches=data.archive_sha256 == expected_archive_sha256(root),
        prediction_valid=_prediction_is_valid(prediction_path, sample_count),
        checkpoint_valid=(
            True
            if spec.model in CLASSICAL_MODELS
            else _checkpoint_is_valid(best_checkpoint_path, best_digest)
        ),
        configured_max_epochs=outcome.get("configured_max_epochs"),
        actual_epochs_completed=outcome.get("actual_epochs_completed"),
        best_epoch=outcome.get("best_epoch"),
    )
    run_kind, eligible, reasons = classify_run(context)

    manifest = build_run_manifest(
        run_id=spec.run_id,
        code_commit=commit,
        dirty=dirty,
        model=spec.model,
        setup=spec.setup,
        fold=spec.fold,
        day_group=spec.day_group,
        horizon=spec.horizon,
        seed=spec.seed,
        data_fingerprint=data.data_fingerprint,
        configuration_hash=configuration_digest,
        configuration_path=config_path,
        status="completed",
        metrics=outcome["metrics"],
        run_kind=run_kind,
        eligible_for_confirmatory_report=eligible,
        exclusion_reasons=reasons,
        protocol_commit=protocol_commit_value,
        protocol_sha256=protocol_hash,
        configured_max_epochs=outcome.get("configured_max_epochs"),
        actual_epochs_completed=outcome.get("actual_epochs_completed"),
        best_epoch=outcome.get("best_epoch"),
        termination_reason=str(outcome["termination_reason"]),
        resumed=resume_from is not None,
        resumed_from_run_id=str(resume_from) if resume_from is not None else None,
        started_utc=started_utc,
        completed_utc=_utc_now(),
        archive_sha256=data.archive_sha256,
        training_file_sha256=data.training_file_sha256,
        testing_file_sha256=data.testing_file_sha256,
        testing_file_sha256_by_day=data.testing_file_sha256_by_day,
        day_index_map=data.day_index_map,
        environment=_environment(),
        labels_regenerated=False,
        test_set_used_for_selection=False,
        prediction_path=str(prediction_path),
        prediction_sha256=prediction_digest,
        sample_count=sample_count,
        class_order=list(CLASS_ORDER),
        best_checkpoint_path=str(best_checkpoint_path) if best_checkpoint_path else None,
        best_checkpoint_sha256=best_digest,
        last_checkpoint_path=str(last_checkpoint_path) if last_checkpoint_path else None,
        last_checkpoint_sha256=last_digest,
        parameter_count=int(outcome["parameter_count"]),
        training_seconds=float(outcome["training_seconds"]),
        total_wall_seconds=time.perf_counter() - started,
        inference_latency_ms_per_sample=outcome["latency"],
        peak_gpu_memory_bytes=int(outcome.get("peak_gpu_memory_bytes", 0)),
        tiny_batch_overfit=outcome.get("tiny_batch_overfit", {"status": "not_applicable"}),
        checkpoint_round_trip=outcome.get("checkpoint_round_trip", {"status": "not_applicable"}),
        device=str(outcome.get("device", "cpu")),
        command=" ".join(sys.argv),
        exit_code=0,
    )
    _record_run(root, artifacts, manifest)
    return manifest


def failed_run_manifest(root: Path, spec: RunSpec, error: BaseException) -> dict[str, Any]:
    """Build an auditable manifest for a run that could not complete.

    Every field comes from the frozen contract and Git, so a failure is still
    recorded even when the audited matrices could not be loaded at all.
    """
    frozen = frozen_data_identity(root)
    config_name = "classical" if spec.model in CLASSICAL_MODELS else spec.model
    config_path = f"configs/experiments/fi2010/{config_name}.yaml"
    config = _load_yaml(root / config_path)
    commit, dirty = _git_state(root)
    if spec.setup == SETUP_ANCHORED_FORWARD:
        entry = frozen["folds"][int(spec.fold or 0)]
        training_digest = str(entry["training_file_sha256"])
        testing_by_day = {str(entry["testing_day_index"]): str(entry["testing_file_sha256"])}
    else:
        group = frozen["day_groups"][str(spec.day_group)]
        training_digest = str(group["training_file_sha256"])
        testing_by_day = {
            str(day["day_index"]): str(day["file_sha256"]) for day in group["test_days"]
        }
    now = _utc_now()
    return build_run_manifest(
        run_id=spec.run_id,
        code_commit=commit,
        dirty=dirty,
        model=spec.model,
        setup=spec.setup,
        fold=spec.fold,
        day_group=spec.day_group,
        horizon=spec.horizon,
        seed=spec.seed,
        data_fingerprint=expected_data_fingerprint(
            root, setup=spec.setup, fold=spec.fold, day_group=spec.day_group
        ),
        configuration_hash=configuration_hash(config),
        configuration_path=config_path,
        status="failed",
        metrics={"error": {"type": type(error).__name__}},
        run_kind="smoke",
        eligible_for_confirmatory_report=False,
        exclusion_reasons=["run status is failed, not completed"],
        protocol_commit=resolve_protocol_commit(root),
        protocol_sha256=protocol_sha256(root),
        termination_reason="failed",
        resumed=False,
        started_utc=now,
        completed_utc=now,
        archive_sha256=str(frozen["archive_sha256"]),
        training_file_sha256=training_digest,
        testing_file_sha256=combined_testing_sha256(testing_by_day),
        testing_file_sha256_by_day=testing_by_day,
        parameter_count=None,
        device="cpu",
        environment=_environment(),
        error=f"{type(error).__name__}: {error}",
        command=" ".join(sys.argv),
        exit_code=1,
    )


def _run_classical(
    data: RunData, spec: RunSpec, config: dict[str, Any], horizon_index: int
) -> dict[str, Any]:
    train_features = data.training_features.T
    train_labels = labels_for_horizon(data.training_labels, horizon_index)
    train_indices, validation_indices = chronological_training_validation_split(
        len(train_labels),
        float(config["validation"]["single_day_fraction"]),
        int(config["validation"]["purge_events"]),
        int(config["validation"]["embargo_events"]),
    )
    started = time.perf_counter()
    model, _ = _fit_classical(spec.model, config, spec.seed, spec.horizon)

    if spec.model == "majority":
        model.fit(train_labels[train_indices])
    elif spec.model != "causal_persistence":
        model.fit(train_features[train_indices], train_labels[train_indices])
    validation_predictions, validation_latency = (
        _timed_predict(model.predict, train_labels[validation_indices])
        if spec.model == "causal_persistence"
        else _timed_predict(model.predict, train_features[validation_indices])
    )
    validation_probabilities = (
        _one_hot(np.maximum(validation_predictions, 0))
        if spec.model == "causal_persistence"
        else model.predict_proba(train_features[validation_indices])
    )
    validation_metrics = _metrics_with_valid_predictions(
        train_labels[validation_indices], validation_predictions, validation_probabilities
    )
    # Refit on all training-only observations before touching any test segment.
    if spec.model == "majority":
        model.fit(train_labels)
    elif spec.model != "causal_persistence":
        model.fit(train_features, train_labels)

    bundle = PredictionBundle()
    test_latency = 0.0
    parameters = 0
    for segment in data.test_segments:
        segment_labels = labels_for_horizon(segment.labels, horizon_index)
        segment_features = segment.features.T
        predictions, probabilities, latency = _classical_predictions(
            model, spec.model, segment_labels, segment_features
        )
        test_latency = max(test_latency, latency)
        bundle.add(segment, segment_labels, predictions, probabilities)
    if spec.model == "logistic_current_event":
        parameters = int(model.coef_.size + model.intercept_.size)
    elif spec.model == "random_forest":
        parameters = int(sum(tree.tree_.node_count for tree in model.estimator.estimators_))

    payload = bundle.concatenate()
    valid = payload["y_pred"] >= 0
    if not np.any(valid):
        raise ValueError("classical model produced no valid test predictions")
    test_metrics = classification_metrics(
        payload["y_true"][valid], payload["y_pred"][valid], payload["probabilities"][valid]
    )
    return {
        "predictions": payload,
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
            "test_by_day": _per_day_metrics(payload),
        },
        "termination_reason": "not_applicable",
        "configured_max_epochs": None,
        "actual_epochs_completed": None,
        "best_epoch": None,
        "parameter_count": parameters,
        "training_seconds": time.perf_counter() - started,
        "latency": {"validation": validation_latency, "test": test_latency},
        "device": "cpu",
    }


def _run_neural(
    root: Path,
    data: RunData,
    spec: RunSpec,
    config: dict[str, Any],
    horizon_index: int,
    *,
    device: str,
    max_epochs: int | None,
    checkpoint_root: Path,
    resume_from: Path | None,
) -> dict[str, Any]:
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
    if spec.model == "mlplob":
        model: Any = MLPLOB(
            input_shape=(sequence_length, 40),
            hidden_sizes=tuple(int(size) for size in config["model"]["hidden_sizes"]),
        )
    else:
        model = DeepLOB()

    tiny_gate: dict[str, Any] = {"status": "not_run"}
    if spec.fold == 1 and spec.horizon == 10 and spec.seed == SEEDS[0]:
        factory: Callable[[], torch.nn.Module]
        if spec.model == "mlplob":

            def factory() -> MLPLOB:
                return MLPLOB(
                    input_shape=(sequence_length, 40),
                    hidden_sizes=tuple(int(size) for size in config["model"]["hidden_sizes"]),
                )
        else:
            factory = DeepLOB
        tiny_gate = tiny_batch_overfit_gate(
            factory, input_shape=(sequence_length, 40), device=device, seed=spec.seed
        )

    best_checkpoint_path = checkpoint_root / f"{spec.run_id}.best.pt"
    last_checkpoint_path = checkpoint_root / f"{spec.run_id}.last.pt"
    configured_epochs = max_epochs or int(config["training"]["max_epochs"])
    fit_result = fit_torch_classifier(
        model,
        train_dataset,
        validation_dataset,
        seed=spec.seed,
        max_epochs=configured_epochs,
        patience=int(config["training"]["patience"]),
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
        device=device,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        configuration_hash=configuration_hash(config),
        data_fingerprint=data.data_fingerprint,
        protocol_hash=protocol_sha256(root),
        resume_from=resume_from,
    )

    bundle = PredictionBundle()
    batch_size = int(config["training"]["batch_size"])
    target_device = torch.device(device)
    model.eval()
    test_latency = 0.0
    for segment in data.test_segments:
        segment_dataset = SegmentedWindowDataset(
            [(segment.lob, segment.labels)], horizon_index, sequence_length
        )
        true_values, predictions, probabilities, latency = predict_raw(
            model, segment_dataset, target_device, batch_size
        )
        test_latency = max(test_latency, latency)
        bundle.add(segment, true_values, predictions, probabilities)

    payload = bundle.concatenate()
    test_metrics = classification_metrics(
        payload["y_true"], payload["y_pred"], payload["probabilities"]
    )
    return {
        "predictions": payload,
        "metrics": {
            "validation": fit_result.validation_metrics,
            "test": test_metrics,
            "test_by_day": _per_day_metrics(payload),
        },
        "termination_reason": fit_result.termination_reason,
        "configured_max_epochs": configured_epochs,
        "actual_epochs_completed": fit_result.actual_epochs_completed,
        "best_epoch": fit_result.best_epoch,
        "parameter_count": parameter_count(model),
        "training_seconds": fit_result.training_seconds,
        "latency": {
            "validation": fit_result.inference_latency_ms_per_sample,
            "test": test_latency,
        },
        "peak_gpu_memory_bytes": fit_result.peak_gpu_memory_bytes,
        "tiny_batch_overfit": tiny_gate,
        "checkpoint_round_trip": {
            "passed": fit_result.checkpoint_round_trip,
            "best_path": str(best_checkpoint_path),
        },
        "best_checkpoint_path": best_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
        "device": device,
    }


# --- Run index --------------------------------------------------------------


def logical_identity(manifest: dict[str, Any]) -> str:
    """Return the logical matrix identity of a manifest, independent of run_id."""
    return configuration_hash(
        {
            "model": manifest.get("model", ""),
            "setup": manifest.get("setup", ""),
            "fold": manifest.get("fold"),
            "day_group": manifest.get("day_group"),
            "horizon": manifest.get("horizon"),
            "seed": manifest.get("seed"),
            "configuration_hash": manifest.get("configuration_hash", ""),
            "data_fingerprint": manifest.get("data_fingerprint", ""),
            "run_kind": manifest.get("run_kind", ""),
        }
    )


def _load_manifests(runs_root: Path) -> list[dict[str, Any]]:
    if not runs_root.is_dir():
        return []
    manifests = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(runs_root.glob("*.json"))
    ]
    return sorted(manifests, key=lambda manifest: str(manifest.get("run_id", "")))


def duplicate_logical_identities(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every logical identity claimed by more than one manifest."""
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        by_identity[logical_identity(manifest)].append(manifest)
    conflicts: list[dict[str, Any]] = []
    for identity, group in sorted(by_identity.items()):
        if len(group) < 2:
            continue
        first = group[0]
        conflicts.append(
            {
                "logical_identity": identity,
                "model": first.get("model", ""),
                "setup": first.get("setup", ""),
                "fold": first.get("fold"),
                "day_group": first.get("day_group"),
                "horizon": first.get("horizon"),
                "seed": first.get("seed"),
                "run_kind": first.get("run_kind", ""),
                "run_ids": sorted(str(manifest.get("run_id", "")) for manifest in group),
            }
        )
    return conflicts


def generate_run_index(root: Path, artifact_root: Path | None = None) -> dict[str, Any]:
    """Generate a deterministic run index from manifests and frozen planning only."""
    artifacts = artifact_root if artifact_root is not None else default_artifact_root(root)
    paths = run_paths(artifacts)
    manifests = _load_manifests(paths.runs)
    manifests_by_id: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        manifests_by_id[str(manifest["run_id"])] = manifest

    orphan_checkpoints: list[str] = []
    if paths.checkpoints.is_dir():
        orphan_checkpoints = sorted(
            item.name for item in paths.checkpoints.iterdir() if item.suffix in {".pt", ".pth"}
        )
    orphan_predictions: list[str] = []
    if paths.predictions.is_dir():
        orphan_predictions = sorted(
            item.name for item in paths.predictions.iterdir() if item.suffix == ".npz"
        )
    for manifest in manifests:
        for key, pool in (
            ("best_checkpoint_path", orphan_checkpoints),
            ("last_checkpoint_path", orphan_checkpoints),
            ("checkpoint_path", orphan_checkpoints),
            ("prediction_path", orphan_predictions),
        ):
            value = manifest.get(key)
            if value:
                name = Path(str(value)).name
                if name in pool:
                    pool.remove(name)

    specs = planned_run_specs(root)
    planned_cells = sorted(spec.run_id for spec in specs)
    conflicts = duplicate_logical_identities(manifests)
    conflicted_ids = {run_id for conflict in conflicts for run_id in conflict["run_ids"]}

    completed_confirmatory: list[str] = []
    completed_smoke: list[str] = []
    failed: list[str] = []
    interrupted: list[str] = []
    for run_id, manifest in sorted(manifests_by_id.items()):
        status = manifest.get("status", "")
        if status == "completed":
            if manifest.get("eligible_for_confirmatory_report") and run_id not in conflicted_ids:
                completed_confirmatory.append(run_id)
            else:
                completed_smoke.append(run_id)
        elif status == "failed":
            failed.append(run_id)
        elif status == "interrupted":
            interrupted.append(run_id)

    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for spec in specs:
        totals["planned_by_model"][spec.model] += 1
        totals["planned_by_setup"][spec.setup] += 1
        totals["planned_by_horizon"][f"h{spec.horizon}"] += 1
        totals["planned_by_seed"][f"s{spec.seed}"] += 1
        group = f"fold_{spec.fold}" if spec.fold is not None else f"day_group_{spec.day_group}"
        totals["planned_by_fold_or_day_group"][group] += 1
    for manifest in manifests:
        totals["manifests_by_status"][str(manifest.get("status", ""))] += 1
        totals["manifests_by_setup"][str(manifest.get("setup", ""))] += 1
        totals["manifests_by_model"][str(manifest.get("model", ""))] += 1
        eligibility = (
            "eligible" if manifest.get("eligible_for_confirmatory_report") else "not_eligible"
        )
        totals["manifests_by_eligibility"][eligibility] += 1

    entries = [
        {
            "run_id": run_id,
            "run_kind": manifest.get("run_kind", "smoke"),
            "eligible_for_confirmatory_report": bool(
                manifest.get("eligible_for_confirmatory_report", False)
            ),
            "exclusion_reasons": manifest.get("exclusion_reasons", []),
            "logical_identity": logical_identity(manifest),
            "duplicate_logical_identity": run_id in conflicted_ids,
            "model": manifest.get("model", ""),
            "setup": manifest.get("setup", ""),
            "fold": manifest.get("fold"),
            "day_group": manifest.get("day_group"),
            "horizon": manifest.get("horizon"),
            "seed": manifest.get("seed"),
            "status": manifest.get("status", ""),
            "exit_code": manifest.get("exit_code"),
            "termination_reason": manifest.get("termination_reason"),
            "protocol_commit": manifest.get("protocol_commit"),
            "protocol_sha256": manifest.get("protocol_sha256"),
            "code_commit": manifest.get("code_commit"),
            "git_tree_dirty": manifest.get("git_tree_dirty", manifest.get("dirty", False)),
            "configuration_hash": manifest.get("configuration_hash", ""),
            "data_fingerprint": manifest.get("data_fingerprint", ""),
            "best_checkpoint_sha256": manifest.get("best_checkpoint_sha256"),
            "prediction_sha256": manifest.get("prediction_sha256"),
            "sample_count": manifest.get("sample_count"),
            "configured_max_epochs": manifest.get("configured_max_epochs"),
            "actual_epochs_completed": manifest.get("actual_epochs_completed"),
            "best_epoch": manifest.get("best_epoch"),
            "resumed": manifest.get("resumed", False),
            "started_utc": manifest.get("started_utc", ""),
            "completed_utc": manifest.get("completed_utc", ""),
        }
        for run_id, manifest in sorted(manifests_by_id.items())
    ]

    latest_completed = ""
    for manifest in manifests:
        completed = str(manifest.get("completed_utc", ""))
        if completed > latest_completed:
            latest_completed = completed

    return {
        "schema_version": 1,
        "generated_from_manifests_utc": latest_completed or None,
        "total_manifest_count": len(manifests_by_id),
        "planned_cell_count": len(planned_cells),
        "planned_cells": planned_cells,
        "planned_totals": {
            key: dict(sorted(value.items())) for key, value in sorted(totals.items())
        },
        "completed_confirmatory": sorted(completed_confirmatory),
        "completed_smoke": sorted(completed_smoke),
        "failed": sorted(failed),
        "interrupted": sorted(interrupted),
        "missing_manifests": sorted(set(planned_cells) - set(manifests_by_id)),
        "duplicate_logical_identities": conflicts,
        "duplicate_run_id_count": len(conflicted_ids),
        "orphan_checkpoints": orphan_checkpoints,
        "orphan_predictions": orphan_predictions,
        "runs": entries,
    }


def write_run_index(root: Path, artifact_root: Path | None = None) -> Path:
    """Write the deterministic run index atomically and return its path."""
    artifacts = artifact_root if artifact_root is not None else default_artifact_root(root)
    index = generate_run_index(root, artifacts)
    path = run_paths(artifacts).index
    _write_json(path, index)
    return path


# --- Report -----------------------------------------------------------------

INCOMPLETE_COVERAGE = "INCOMPLETE â€” no reference conclusion"


def _expected_coverage(root: Path) -> dict[str, set[tuple[str, int]]]:
    expected: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for spec in planned_run_specs(root):
        cell = f"fold{spec.fold}" if spec.setup == SETUP_ANCHORED_FORWARD else str(spec.day_group)
        group = f"{spec.model}|{spec.setup}|h{spec.horizon}"
        expected[group].add((cell, spec.seed))
    return expected


def write_report(root: Path, artifact_root: Path | None = None) -> tuple[Path, Path]:
    """Aggregate ignored run manifests into deterministic JSON and Markdown reports."""
    artifacts = artifact_root if artifact_root is not None else default_artifact_root(root)
    manifests = _load_manifests(run_paths(artifacts).runs)
    conflicts = duplicate_logical_identities(manifests)
    conflicted_ids = {run_id for conflict in conflicts for run_id in conflict["run_ids"]}
    confirmatory = [
        manifest
        for manifest in manifests
        if manifest.get("eligible_for_confirmatory_report")
        and manifest.get("status") == "completed"
        and str(manifest.get("run_id", "")) not in conflicted_ids
    ]
    confirmatory_ids = {str(manifest.get("run_id", "")) for manifest in confirmatory}
    smoke = [
        manifest
        for manifest in manifests
        if str(manifest.get("run_id", "")) not in confirmatory_ids
    ]

    grouped: dict[str, list[float]] = defaultdict(list)
    observed: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for manifest in confirmatory:
        test_metrics = manifest.get("metrics", {}).get("test", {})
        if "macro_f1" not in test_metrics:
            continue
        group = f"{manifest.get('model')}|{manifest.get('setup')}|h{manifest.get('horizon')}"
        grouped[group].append(float(test_metrics["macro_f1"]))
        cell = (
            f"fold{manifest.get('fold')}"
            if manifest.get("setup") == SETUP_ANCHORED_FORWARD
            else str(manifest.get("day_group"))
        )
        observed[group].add((cell, int(manifest.get("seed", 0))))

    expected = _expected_coverage(root)
    summary: dict[str, dict[str, Any]] = {}
    for group, values in sorted(grouped.items()):
        required = expected.get(group, set())
        complete = bool(required) and required.issubset(observed[group])
        entry: dict[str, Any] = {
            "count": len(values),
            "expected_cell_count": len(required),
            "observed_cell_count": len(observed[group]),
            "coverage_complete": complete,
            "mean_macro_f1": float(np.mean(values)),
            "std_macro_f1": float(np.std(values)) if len(values) >= 2 else "unavailable",
            "reference_comparison": (
                "eligible for reference comparison" if complete else INCOMPLETE_COVERAGE
            ),
        }
        summary[group] = entry

    tiny_gates = sorted(
        (
            {
                "run_id": manifest["run_id"],
                "model": manifest["model"],
                "result": manifest.get("tiny_batch_overfit", {}),
            }
            for manifest in manifests
            if manifest.get("tiny_batch_overfit", {}).get("status")
            not in {"not_run", "not_applicable"}
        ),
        key=lambda gate: str(gate["run_id"]),
    )
    smoke_exclusion = sorted(
        (
            {
                "run_id": manifest["run_id"],
                "run_kind": manifest.get("run_kind", "smoke"),
                "exclusion_reasons": manifest.get("exclusion_reasons", []),
            }
            for manifest in smoke
        ),
        key=lambda item: str(item["run_id"]),
    )
    by_setup = {
        setup: sorted(
            str(manifest.get("run_id", ""))
            for manifest in confirmatory
            if manifest.get("setup") == setup
        )
        for setup in SETUPS
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "local FI-2010 reproduction; not publisher-verified",
        "data_contract": {
            "archive_sha256": expected_archive_sha256(root),
            "source_manifest": "data/interim/fi2010/fi2010_source_manifest.json",
            "split_manifest": "data/interim/fi2010/fi2010_split_manifest.json",
            "frozen_data_identity": "configs/references/fi2010_frozen_data_identity.yaml",
            "feature_rows": 40,
            "supplied_non_label_rows": 144,
            "label_rows": 5,
            "window_length": 100,
            "boundary_contract": (
                "published fold/file boundaries preserved; no finer day or instrument "
                "boundaries inferred; each test day is windowed independently"
            ),
            "labels_regenerated": False,
            "test_set_used_for_selection": False,
        },
        "reference_comparison": {
            "paper": "arxiv:1808.03668v6",
            "official_repository_commit": "ff14d7c2fd38bdfc143389786993d0f0236d4eb8",
            "local_results_are_not_publisher_verified": True,
            "policy": (
                "A published reference value is only compared once the matching cell "
                "has complete fold or day-group and seed coverage for that model, "
                "setup, and horizon. Single-cell results are never compared with "
                "multi-fold published means."
            ),
        },
        "confirmatory_runs": confirmatory,
        "confirmatory_run_ids_by_setup": by_setup,
        "smoke_runs": smoke,
        "smoke_exclusion_summary": smoke_exclusion,
        "duplicate_logical_identities": conflicts,
        "summary": summary,
        "gates": {
            "tiny_batch_overfit": tiny_gates,
            "all_passed": all(bool(gate["result"].get("passed")) for gate in tiny_gates),
        },
        "failed_runs": [m for m in manifests if m.get("status") == "failed"],
        "resumed_runs": [m for m in manifests if m.get("resumed")],
        "run_index": generate_run_index(root, artifacts),
    }
    report["report_fingerprint"] = report_fingerprint(report)
    json_path = root / "reports" / "results" / "fi2010_baseline_reproduction.json"
    markdown_path = root / "reports" / "results" / "fi2010_baseline_reproduction.md"
    _write_json(json_path, report)
    write_run_index(root, artifacts)

    lines = [
        "# FI-2010 Baseline Reproduction Results",
        "",
        (
            "Local results are reproductions under the tracked protocol; they are not "
            "publisher-verified benchmark values."
        ),
        "",
        f"- Report fingerprint: `{report['report_fingerprint']}`",
        f"- Confirmatory runs: `{len(confirmatory)}`",
        f"- Excluded smoke runs: `{len(smoke)}`",
        f"- Failed runs: `{len(report['failed_runs'])}`",
        f"- Resumed runs: `{len(report['resumed_runs'])}`",
        f"- Duplicate logical identities: `{len(conflicts)}`",
        "- Labels regenerated: `False`",
        "- Test-set-driven selection: `False`",
        "",
    ]
    for setup in SETUPS:
        lines.append(f"- Confirmatory runs, `{setup}`: `{len(by_setup[setup])}`")
    lines.append("")

    if not confirmatory:
        lines.extend(["**Confirmatory matrix not started.**", ""])

    if conflicts:
        lines.extend(
            [
                "## Duplicate logical identities (excluded from aggregation)",
                "",
                "| Model | Setup | Cell | Horizon | Seed | Run IDs |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for conflict in conflicts:
            cell = conflict["fold"] if conflict["fold"] is not None else conflict["day_group"]
            lines.append(
                f"| {conflict['model']} | {conflict['setup']} | {cell} | "
                f"{conflict['horizon']} | {conflict['seed']} | "
                f"{', '.join(conflict['run_ids'])} |"
            )
        lines.append("")

    if smoke:
        lines.extend(
            [
                "## Engineering smoke runs excluded from confirmatory analysis",
                "",
                "| Run ID | Reasons |",
                "|---|---|",
            ]
        )
        for item in smoke_exclusion[:20]:
            reasons = "; ".join(item.get("exclusion_reasons", [])) or "pre-freeze / legacy"
            lines.append(f"| {item['run_id']} | {reasons} |")
        if len(smoke) > 20:
            lines.append(f"| ... and {len(smoke) - 20} more | |")
        lines.append("")

    if confirmatory:
        lines.extend(
            [
                "## Confirmatory macro-F1 summary",
                "",
                "| Model/Setup/Horizon | Runs | Coverage | Mean | Std | Reference |",
                "|---|---:|---|---:|---:|---|",
            ]
        )
        for key, entry in summary.items():
            std = entry["std_macro_f1"]
            std_text = f"{std:.6f}" if isinstance(std, float) else str(std)
            coverage = f"{entry['observed_cell_count']}/{entry['expected_cell_count']}"
            lines.append(
                f"| {key} | {entry['count']} | {coverage} | "
                f"{entry['mean_macro_f1']:.6f} | {std_text} | {entry['reference_comparison']} |"
            )

    lines.extend(
        [
            "",
            (
                "Full metrics, confusion matrices, timing, memory, manifests, failures, "
                "and commands are in the JSON report."
            ),
            "",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


# --- CLI --------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run audited FI-2010 baselines")
    parser.add_argument("--model", choices=ALL_MODELS)
    parser.add_argument("--setup", choices=SETUPS, default=SETUP_ANCHORED_FORWARD)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--day-group", default=DAY_GROUP_FIRST_SEVEN_FINAL_THREE)
    parser.add_argument("--horizon", type=int, choices=HORIZONS, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument(
        "--matrix", action="store_true", help="run all folds, horizons, and required seeds"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="mark this run as smoke (nonconfirmatory)"
    )
    parser.add_argument("--report-only", action="store_true")
    return parser


def _selected_specs(args: argparse.Namespace, root: Path) -> list[RunSpec]:
    if args.matrix:
        return [
            spec
            for spec in planned_run_specs(root)
            if spec.model == args.model and spec.setup == args.setup
        ]
    if args.setup == SETUP_ANCHORED_FORWARD:
        return [
            RunSpec(
                model=args.model,
                setup=args.setup,
                horizon=args.horizon,
                seed=args.seed,
                fold=args.fold,
            )
        ]
    return [
        RunSpec(
            model=args.model,
            setup=args.setup,
            horizon=args.horizon,
            seed=args.seed,
            day_group=args.day_group,
        )
    ]


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
    failures = 0
    for spec in _selected_specs(args, root):
        try:
            manifest = execute_run(
                root,
                spec,
                device=args.device,
                max_epochs=args.max_epochs,
                smoke=args.smoke,
                resume_from=Path(args.resume_from) if args.resume_from else None,
            )
        except Exception as error:  # noqa: BLE001 - failures are recorded, not raised
            failures += 1
            print(f"ERROR {spec.run_id}: {type(error).__name__}: {error}", file=sys.stderr)
            try:
                _record_run(
                    root, default_artifact_root(root), failed_run_manifest(root, spec, error)
                )
            except Exception as nested:  # noqa: BLE001 - reported, never fatal
                print(
                    f"ERROR recording failure for {spec.run_id}: {type(nested).__name__}: {nested}",
                    file=sys.stderr,
                )
            continue
        print(f"Run {manifest['status']}: {spec.run_id}")
    print(f"Run index: {write_run_index(root)}")
    report_paths = write_report(root)
    print(f"Report JSON: {report_paths[0]}")
    print(f"Report Markdown: {report_paths[1]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
