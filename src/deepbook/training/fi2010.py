"""Training, alignment, cache, and checkpoint helpers for FI-2010."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

# ponytail: frozen contract files — paths relative to repo root
_PROTOCOL_CONTRACT_PATHS = (
    "reports/protocol/fi2010_baseline_reproduction.md",
    "configs/references/deeplob_fi2010.yaml",
    "configs/references/fi2010_frozen_data_identity.yaml",
    "configs/experiments/fi2010/classical.yaml",
    "configs/experiments/fi2010/mlplob.yaml",
    "configs/experiments/fi2010/deeplob.yaml",
)

FROZEN_DATA_IDENTITY_PATH = "configs/references/fi2010_frozen_data_identity.yaml"
DAY_GROUP_FIRST_SEVEN_FINAL_THREE = "days_8_9_10"
SETUP_ANCHORED_FORWARD = "anchored_forward"
SETUP_FIRST_SEVEN_FINAL_THREE = "first_seven_final_three"
GIT_CLOCK_SKEW_TOLERANCE_SECONDS = 300


def resolve_protocol_commit(root: Path) -> str:
    """Return the latest Git commit touching any frozen protocol/config/reference file.

    Returns an empty string when ``root`` is not a Git work tree or Git is
    unavailable. An empty commit can never satisfy :func:`check_protocol_ancestry`,
    so a run without resolvable provenance is excluded rather than crashing.
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", *_PROTOCOL_CONTRACT_PATHS],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return ""
    return result.stdout.strip()


def protocol_sha256(root: Path) -> str:
    """Return a deterministic canonical SHA-256 of the complete frozen contract files."""
    lines: list[str] = []
    for rel in sorted(_PROTOCOL_CONTRACT_PATHS):
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(f"protocol contract file missing: {rel}")
        lines.append(f"{rel}:{p.read_bytes().hex()}")
    payload = "\n".join(lines)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_protocol_ancestry(root: Path, protocol_commit: str) -> bool:
    """Return True when HEAD descends from the given protocol commit."""
    if not protocol_commit or len(protocol_commit) != 40:
        return False
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, "HEAD"],
        cwd=root,
        capture_output=True,
    )
    return result.returncode == 0


def git_commit_timestamp(root: Path, commit: str) -> str | None:
    """Return a commit's committer timestamp, or ``None`` when it is unknown."""
    result = subprocess.run(
        ["git", "show", "-s", "--format=%cI", commit],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip()


def git_commit_tree(root: Path, commit: str) -> str | None:
    """Return the tree object recorded by a commit, or ``None`` when unknown."""
    result = subprocess.run(
        ["git", "rev-parse", f"{commit}^{{tree}}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip()


def check_code_commit_ancestry(root: Path, code_commit: str) -> bool:
    """Return True only when the recorded code commit is in current HEAD ancestry."""
    if not code_commit or len(code_commit) != 40:
        return False
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", code_commit, "HEAD"],
        cwd=root,
        capture_output=True,
    )
    return result.returncode == 0


def _parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def code_commit_provenance_reasons(
    root: Path,
    manifest: dict[str, Any],
    *,
    tolerance_seconds: int = GIT_CLOCK_SKEW_TOLERANCE_SECONDS,
) -> list[str]:
    """Return temporal, ancestry, and tree-integrity failures for a run manifest."""
    reasons: list[str] = []
    commit = str(manifest.get("code_commit", ""))
    actual_timestamp = git_commit_timestamp(root, commit)
    actual_tree = git_commit_tree(root, commit)
    if actual_timestamp is None or actual_tree is None:
        reasons.append("recorded code commit is unknown")
        return reasons
    if not check_code_commit_ancestry(root, commit):
        reasons.append("recorded code commit is not in current repository ancestry")
    recorded_timestamp = str(manifest.get("code_commit_timestamp", ""))
    if recorded_timestamp != actual_timestamp:
        reasons.append("recorded code commit timestamp does not match the repository")
    if str(manifest.get("code_commit_tree", "")) != actual_tree:
        reasons.append("recorded Git tree hash does not match the code commit")
    started = _parse_utc(str(manifest.get("started_utc", "")))
    completed = _parse_utc(str(manifest.get("completed_utc", "")))
    commit_time = _parse_utc(actual_timestamp)
    if started is None or completed is None or commit_time is None:
        reasons.append("run timestamps or code commit timestamp are invalid")
    else:
        tolerance = timedelta(seconds=tolerance_seconds)
        if commit_time > started + tolerance or commit_time > completed + tolerance:
            reasons.append("recorded code commit did not exist when this run executed")
    return reasons


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
        if key not in {"generated_utc", "report_fingerprint", "generated_from_manifests_utc"}
    }
    return configuration_hash(stable_report)


def frozen_data_identity(root: Path) -> dict[str, Any]:
    """Load the tracked frozen FI-2010 data identity contract."""
    path = root / FROZEN_DATA_IDENTITY_PATH
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"frozen data identity must be a mapping: {path}")
    return value


def anchored_fold_fingerprint(archive_sha256: str, fold: int, training: str, testing: str) -> str:
    """Return the canonical Setup 1 data fingerprint for one anchored fold."""
    return configuration_hash(
        {
            "archive_sha256": archive_sha256,
            "fold": int(fold),
            "training_file_sha256": training,
            "testing_file_sha256": testing,
        }
    )


def day_group_fingerprint(
    archive_sha256: str, day_group: str, training: str, testing_by_day: dict[str, str]
) -> str:
    """Return the canonical Setup 2 data fingerprint for one day group."""
    return configuration_hash(
        {
            "archive_sha256": archive_sha256,
            "day_group": day_group,
            "training_file_sha256": training,
            "testing_file_sha256_by_day": dict(sorted(testing_by_day.items())),
        }
    )


def combined_testing_sha256(testing_by_day: dict[str, str]) -> str:
    """Return one canonical digest over several ordered per-day testing file digests."""
    payload = "\n".join(f"{day}:{testing_by_day[day]}" for day in sorted(testing_by_day, key=int))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expected_data_fingerprint(
    root: Path, *, setup: str, fold: int | None = None, day_group: str | None = None
) -> str:
    """Return the frozen expected data fingerprint for one setup cell."""
    frozen = frozen_data_identity(root)
    archive = str(frozen["archive_sha256"])
    if setup == SETUP_ANCHORED_FORWARD:
        if fold is None:
            raise ValueError("anchored_forward requires a fold")
        entry = frozen["folds"][int(fold)]
        return anchored_fold_fingerprint(
            archive,
            int(fold),
            str(entry["training_file_sha256"]),
            str(entry["testing_file_sha256"]),
        )
    if setup == SETUP_FIRST_SEVEN_FINAL_THREE:
        if day_group is None:
            raise ValueError("first_seven_final_three requires a day_group")
        entry = frozen["day_groups"][str(day_group)]
        testing_by_day = {
            str(day["day_index"]): str(day["file_sha256"]) for day in entry["test_days"]
        }
        return day_group_fingerprint(
            archive, str(day_group), str(entry["training_file_sha256"]), testing_by_day
        )
    raise ValueError(f"unknown setup: {setup}")


def expected_archive_sha256(root: Path) -> str:
    """Return the frozen authoritative FI-2010 archive digest."""
    return str(frozen_data_identity(root)["archive_sha256"])


def epoch_shuffle_seed(base_seed: int, epoch: int) -> int:
    """Return the shuffle seed for one epoch as a pure function of base seed and epoch.

    Uninterrupted epoch ``e`` and resumed epoch ``e`` must receive the identical
    sample order, so this must never depend on prior generator consumption or on
    whether training was resumed.
    """
    if epoch < 1:
        raise ValueError("epoch must be positive")
    digest = hashlib.sha256(f"{int(base_seed)}:{int(epoch)}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


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
    best_epoch: int | None = None,
    termination_reason: str = "not_applicable",
    protocol_commit: str = "",
    protocol_sha256: str = "",
    configuration_path: str = "",
    resumed: bool = False,
    resumed_from_run_id: str | None = None,
    day_group: str | None = None,
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
        "protocol_commit": protocol_commit,
        "protocol_sha256": protocol_sha256,
        "code_commit": code_commit,
        "git_tree_dirty": dirty,
        "model": model,
        "setup": setup,
        "fold": fold,
        "day_group": day_group,
        "horizon": horizon,
        "seed": seed,
        "data_fingerprint": data_fingerprint,
        "configuration_hash": configuration_hash,
        "configuration_path": configuration_path,
        "configured_max_epochs": configured_max_epochs,
        "actual_epochs_completed": actual_epochs_completed,
        "best_epoch": best_epoch,
        "termination_reason": termination_reason,
        "status": status,
        "resumed": resumed,
        "resumed_from_run_id": resumed_from_run_id,
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


def _write_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_states() -> dict[str, Any]:
    states: dict[str, Any] = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda_rng"] = torch.cuda.get_rng_state_all()
    return states


def save_checkpoint_atomic(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    configuration_hash: str,
    data_fingerprint: str,
    best_validation_metric: float,
    best_epoch: int = 0,
    patience_counter: int = 0,
    protocol_hash: str = "",
) -> None:
    """Atomically save the validation-selected best-model checkpoint.

    This checkpoint is for final evaluation and test prediction only. Resuming an
    interrupted run must use the last-state checkpoint written by
    :func:`save_training_state_atomic`, never this one.
    """
    payload: dict[str, Any] = {
        "checkpoint_kind": "best",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "configuration_hash": configuration_hash,
        "data_fingerprint": data_fingerprint,
        "best_validation_metric": best_validation_metric,
        "best_epoch": best_epoch or epoch,
        "patience_counter": patience_counter,
        "protocol_hash": protocol_hash,
        **_rng_states(),
    }
    _write_torch_atomic(path, payload)


def save_training_state_atomic(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    completed_epoch: int,
    seed: int,
    configuration_hash: str,
    data_fingerprint: str,
    protocol_hash: str,
    best_model_state: dict[str, torch.Tensor] | None,
    best_epoch: int,
    best_validation_metric: float,
    patience_counter: int,
    best_checkpoint_path: str | None = None,
    scaler_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
) -> None:
    """Atomically save the exact end-of-epoch state needed to resume training.

    Written after every completed epoch. ``next_epoch`` is the epoch a resumed run
    must start from; the shuffle order for that epoch is derived from
    ``seed`` and the epoch index by :func:`epoch_shuffle_seed`, so no generator
    consumption history has to be replayed.
    """
    payload: dict[str, Any] = {
        "checkpoint_kind": "last",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": completed_epoch,
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "seed": seed,
        "shuffle_seed_base": seed,
        "configuration_hash": configuration_hash,
        "data_fingerprint": data_fingerprint,
        "protocol_hash": protocol_hash,
        "best_model_state": best_model_state,
        "best_epoch": best_epoch,
        "best_checkpoint_path": best_checkpoint_path,
        "best_validation_metric": best_validation_metric,
        "patience_counter": patience_counter,
        "scaler_state": scaler_state,
        "scheduler_state": scheduler_state,
        **_rng_states(),
    }
    _write_torch_atomic(path, payload)


def load_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    """Restore a checkpoint and return its reproducibility metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    # Restore RNG state if present
    if "python_rng" in payload:
        random.setstate(payload["python_rng"])
    if "numpy_rng" in payload:
        np.random.set_state(payload["numpy_rng"])
    if "torch_rng" in payload:
        torch.random.set_rng_state(payload["torch_rng"])
    if "cuda_rng" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return {key: payload[key] for key in payload if key not in {"model", "optimizer"}}


def load_training_state(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    """Restore a last-state resume checkpoint and return its metadata."""
    metadata = load_checkpoint(path, model, optimizer)
    kind = metadata.get("checkpoint_kind")
    if kind != "last":
        raise ValueError(
            f"resume requires a last-state checkpoint, got checkpoint_kind={kind!r}; "
            "the best-model checkpoint holds stale weights and must not be resumed from"
        )
    return metadata
