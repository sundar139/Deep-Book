"""Training, alignment, cache, and checkpoint helpers for FI-2010."""

from __future__ import annotations

import hashlib
import json
import os
import random
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def configuration_hash(configuration: Any) -> str:
    """Hash canonical JSON configuration without relying on mapping order."""
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def report_fingerprint(report: Any) -> str:
    """Return the stable fingerprint used to identify a complete report."""
    if not isinstance(report, dict):
        return configuration_hash(report)
    stable_report = {
        key: value
        for key, value in report.items()
        if key not in {"generated_utc", "report_fingerprint"}
    }
    return configuration_hash(stable_report)


def chronological_training_validation_split(
    observation_count: int,
    validation_fraction: float,
    purge_events: int,
    embargo_events: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one training matrix chronologically with a non-overlapping gap."""
    if observation_count <= 0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("observation_count must be positive and validation_fraction in (0,1)")
    if purge_events < 0 or embargo_events < 0:
        raise ValueError("purge_events and embargo_events must be non-negative")
    boundary = int(observation_count * (1.0 - validation_fraction))
    train_end = boundary - purge_events
    validation_start = boundary + embargo_events
    if train_end <= 0 or validation_start >= observation_count:
        raise ValueError("purge/embargo leaves no training or validation observations")
    return np.arange(train_end, dtype=np.int64), np.arange(
        validation_start, observation_count, dtype=np.int64
    )


def build_run_manifest(
    *,
    run_id: str,
    code_commit: str,
    dirty: bool,
    model: str,
    setup: str,
    fold: int | None,
    horizon: int,
    seed: int,
    data_fingerprint: str,
    configuration_hash: str,
    status: str,
    metrics: dict[str, Any],
    run_kind: str = "smoke",
    eligible_for_confirmatory_report: bool = False,
    exclusion_reasons: list[str] | None = None,
    configured_max_epochs: int | None = None,
    actual_epochs_completed: int | None = None,
    started_utc: str = "",
    **details: Any,
) -> dict[str, Any]:
    """Build the schema-shaped record used for every planned or completed run."""
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "run_kind": run_kind,
        "eligible_for_confirmatory_report": eligible_for_confirmatory_report,
        "exclusion_reasons": exclusion_reasons or [],
        "code_commit": code_commit,
        "git_tree_dirty": dirty,
        "model": model,
        "setup": setup,
        "fold": fold,
        "horizon": horizon,
        "seed": seed,
        "data_fingerprint": data_fingerprint,
        "configuration_hash": configuration_hash,
        "configured_max_epochs": configured_max_epochs,
        "actual_epochs_completed": actual_epochs_completed,
        "status": status,
        "started_utc": started_utc,
        "metrics": metrics,
    }
    manifest.update(details)
    return manifest


def validate_run_manifest(manifest: dict[str, Any], schema_path: Path) -> None:
    """Validate a run record against the tracked JSON schema."""
    try:
        import jsonschema
    except ImportError as error:
        raise RuntimeError("jsonschema is required to validate run manifests") from error
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)


def window_count(observation_count: int, sequence_length: int) -> int:
    """Return valid causal window count for one independent segment."""
    if observation_count < 0 or sequence_length <= 0:
        raise ValueError("observation_count must be non-negative and sequence_length positive")
    return max(0, observation_count - sequence_length + 1)


def label_to_index(labels: np.ndarray) -> np.ndarray:
    """Map supplied FI-2010 labels 1,2,3 to model indices 0,1,2."""
    values = np.asarray(labels, dtype=np.int64)
    if np.any((values < 1) | (values > 3)):
        raise ValueError("source labels must be 1, 2, or 3")
    return values - 1


def source_label(indices: np.ndarray) -> np.ndarray:
    """Map model indices 0,1,2 back to supplied labels 1,2,3."""
    values = np.asarray(indices, dtype=np.int64)
    if np.any((values < 0) | (values > 2)):
        raise ValueError("model labels must be 0, 1, or 2")
    return values + 1


def make_windows(
    lob: np.ndarray, labels: np.ndarray, horizon_index: int, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize small windows for tests; real training uses SegmentedWindowDataset."""
    features = np.asarray(lob, dtype=np.float32)
    source_labels = np.asarray(labels)
    if features.ndim != 2 or features.shape[0] != 40:
        raise ValueError("lob must have shape (40, observations)")
    if source_labels.ndim != 2 or source_labels.shape[1] != features.shape[1]:
        raise ValueError("labels must have shape (5, observations)")
    if horizon_index not in range(5):
        raise ValueError("horizon_index must be in range 0..4")
    count = window_count(features.shape[1], sequence_length)
    windows = np.stack(
        [features[:, start : start + sequence_length].T for start in range(count)], axis=0
    )
    targets = label_to_index(source_labels[horizon_index, sequence_length - 1 :])
    return windows, targets


class SegmentedWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy causal windows over independent source segments."""

    def __init__(
        self,
        segments: list[tuple[np.ndarray, np.ndarray]],
        horizon_index: int,
        sequence_length: int,
    ) -> None:
        if horizon_index not in range(5):
            raise ValueError("horizon_index must be in range 0..4")
        self.segments = segments
        self.horizon_index = horizon_index
        self.sequence_length = sequence_length
        self._ends: list[int] = []
        total = 0
        for lob, labels in segments:
            if lob.shape != (40, labels.shape[1]) or labels.shape[0] != 5:
                raise ValueError("each segment must contain lob (40,N) and labels (5,N)")
            total += window_count(lob.shape[1], sequence_length)
            self._ends.append(total)

    def __len__(self) -> int:
        """Return the number of valid windows across all segments."""
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one channel-first window and its horizon target."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment_index = bisect_right(self._ends, index)
        previous = self._ends[segment_index - 1] if segment_index else 0
        start = index - previous
        lob, labels = self.segments[segment_index]
        window = lob[:, start : start + self.sequence_length].T
        target = labels[self.horizon_index, start + self.sequence_length - 1]
        return torch.from_numpy(
            np.ascontiguousarray(window[None, :, :], dtype=np.float32)
        ), torch.tensor(int(target) - 1, dtype=torch.long)


@dataclass(frozen=True)
class ModelingCacheSpec:
    """All source and transformation inputs that identify a matrix cache."""

    archive_sha256: str
    source_matrix_sha256: str
    benchmark_variant: str
    normalization: str
    feature_rows: int
    parser_version: str
    sequence_length: int
    horizon: int
    configuration_hash: str
    dtype: str

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical cache identity fields."""
        return self.__dict__.copy()

    @property
    def key(self) -> str:
        """Return the content-addressed cache key."""
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def modeling_cache_path(root: Path, spec: ModelingCacheSpec) -> Path:
    """Return the ignored cache path for one underlying matrix, not windows."""
    return root / "fi2010" / "modeling" / f"{spec.key}.npz"


def save_matrix_cache(
    path: Path, lob: np.ndarray, labels: np.ndarray, spec: ModelingCacheSpec
) -> None:
    """Write an underlying matrix cache atomically with a JSON metadata sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    np.savez_compressed(temporary, lob=lob, labels=labels)
    generated = temporary.with_suffix(temporary.suffix + ".npz")
    generated.replace(path)
    metadata = path.with_suffix(".json")
    metadata_tmp = metadata.with_name(f"{metadata.name}.{os.getpid()}.tmp")
    metadata_tmp.write_text(json.dumps(spec.as_dict(), sort_keys=True) + "\n", encoding="utf-8")
    metadata_tmp.replace(metadata)


def load_matrix_cache(path: Path, spec: ModelingCacheSpec) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a matrix cache before reuse."""
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata != spec.as_dict():
        raise ValueError("modeling cache metadata does not match requested specification")
    with np.load(path, allow_pickle=False) as archive:
        lob = np.asarray(archive["lob"])
        labels = np.asarray(archive["labels"])
    if lob.shape != (spec.feature_rows, labels.shape[1]) or labels.shape[0] != 5:
        raise ValueError("modeling cache arrays have invalid shapes")
    if not np.isfinite(lob).all() or not np.isfinite(labels).all():
        raise ValueError("modeling cache contains non-finite values")
    return lob, labels


def seed_everything(seed: int) -> dict[str, Any]:
    """Set reproducibility controls and return the settings that were applied."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    deterministic_enabled = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        deterministic_enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "torch_deterministic_algorithms": deterministic_enabled,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def save_checkpoint_atomic(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    configuration_hash: str,
    data_fingerprint: str,
    best_validation_metric: float,
) -> None:
    """Atomically save a complete resumable training state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "configuration_hash": configuration_hash,
        "data_fingerprint": data_fingerprint,
        "best_validation_metric": best_validation_metric,
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    """Restore a checkpoint and return its reproducibility metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return {key: payload[key] for key in payload if key not in {"model", "optimizer"}}
