"""Audited FI-2010 fold loading for baseline runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from deepbook.data.fi2010 import PARSER_VERSION
from deepbook.data.fi2010.dataset import ParsedMatrix, discover_matrices, parse_matrix
from deepbook.training.fi2010 import (
    DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
    ModelingCacheSpec,
    anchored_fold_fingerprint,
    combined_testing_sha256,
    configuration_hash,
    day_group_fingerprint,
    frozen_data_identity,
    label_to_index,
    load_matrix_cache,
    modeling_cache_path,
    save_matrix_cache,
)


@dataclass(frozen=True)
class FoldMatrices:
    """Cached, parsed training/testing matrices for one audited fold."""

    fold: int
    training_lob: np.ndarray
    training_features: np.ndarray
    training_labels: np.ndarray
    testing_lob: np.ndarray
    testing_features: np.ndarray
    testing_labels: np.ndarray
    archive_sha256: str
    training_file_sha256: str
    testing_file_sha256: str
    data_fingerprint: str


@dataclass(frozen=True)
class DayMatrices:
    """One independent audited test-day source file."""

    day_index: int
    source_fold: int
    lob: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    file_sha256: str
    observation_count: int


@dataclass(frozen=True)
class DayGroupMatrices:
    """Cumulative training matrix plus several independent audited test days."""

    day_group: str
    training_lob: np.ndarray
    training_features: np.ndarray
    training_labels: np.ndarray
    test_days: tuple[DayMatrices, ...]
    archive_sha256: str
    training_file_sha256: str
    testing_file_sha256: str
    testing_file_sha256_by_day: dict[str, str]
    data_fingerprint: str


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _cache_one(
    *,
    root: Path,
    extraction_root: Path,
    record: dict[str, Any],
    experiment_configuration_hash: str,
    archive_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    spec = ModelingCacheSpec(
        archive_sha256=archive_sha256,
        source_matrix_sha256=str(record["file_sha256"]),
        benchmark_variant=str(record["benchmark_variant"]),
        normalization=str(record["normalization"]),
        feature_rows=144,
        parser_version=PARSER_VERSION,
        sequence_length=100,
        horizon=0,
        configuration_hash=experiment_configuration_hash,
        dtype="float32",
    )
    path = modeling_cache_path(root / "data" / "interim", spec)
    try:
        return load_matrix_cache(path, spec)
    except FileNotFoundError:
        pass
    matrices = discover_matrices(extraction_root, [str(record["source_member_path"])])
    if len(matrices) != 1:
        raise ValueError(f"audited matrix could not be discovered: {record['source_member_path']}")
    parsed: ParsedMatrix = parse_matrix(matrices[0], extraction_root)
    lob = np.asarray(parsed.values[:144], dtype=np.float32)
    labels = np.asarray(parsed.labels, dtype=np.int8)
    save_matrix_cache(path, lob, labels, spec)
    return lob, labels


@dataclass(frozen=True)
class _SplitContext:
    """Resolved audited split records and paths for one experiment configuration."""

    archive_sha256: str
    extraction_root: Path
    experiment_hash: str
    by_fold_role: dict[tuple[int, str], dict[str, Any]]


def _split_context(root: Path, experiment_config_path: Path) -> _SplitContext:
    experiment = _read_yaml(root / experiment_config_path)
    source_config = _read_yaml(root / str(experiment["source_config"]))
    archive_sha256 = str(source_config["source"]["archive_sha256"])
    variant = str(experiment["benchmark_variant"])
    normalization = str(experiment["normalization"])
    split_path = root / "data" / "interim" / "fi2010" / "fi2010_split_manifest.json"
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    by_fold_role = {
        (int(record["fold_identifier"]), str(record["role"])): record
        for record in split_manifest["splits"]
        if record["benchmark_variant"] == variant and record["normalization"] == normalization
    }
    return _SplitContext(
        archive_sha256=archive_sha256,
        extraction_root=root / "data" / "raw" / "fi2010" / "extracted" / archive_sha256,
        experiment_hash=configuration_hash(experiment),
        by_fold_role=by_fold_role,
    )


def _require_record(context: _SplitContext, fold: int, role: str) -> dict[str, Any]:
    record = context.by_fold_role.get((int(fold), role))
    if record is None:
        raise ValueError(f"audited split manifest has no {role} record for fold {fold}")
    return record


def load_day_group(
    root: Path,
    day_group: str = DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
    *,
    experiment_config_path: Path = Path("configs/experiments/fi2010/classical.yaml"),
) -> DayGroupMatrices:
    """Load the first-seven/final-three setup from audited files only.

    Training is the single cumulative days 1-7 matrix. Each of days 8, 9, and 10
    is a separate audited daily file, kept separate so windows never cross a day
    boundary and no test observation is duplicated.
    """
    frozen = frozen_data_identity(root)
    group = frozen["day_groups"].get(str(day_group))
    if group is None:
        raise ValueError(f"frozen contract declares no day group: {day_group}")
    context = _split_context(root, experiment_config_path)
    if context.archive_sha256 != str(frozen["archive_sha256"]):
        raise ValueError("archive digest does not match the frozen data identity contract")

    training_record = _require_record(context, int(group["training_fold"]), "training")
    if str(training_record["file_sha256"]) != str(group["training_file_sha256"]):
        raise ValueError("days 1-7 training file digest does not match the frozen contract")
    if int(training_record["observation_count"]) != int(group["training_observations"]):
        raise ValueError("days 1-7 training observation count does not match the frozen contract")
    train_lob, train_labels = _cache_one(
        root=root,
        extraction_root=context.extraction_root,
        record=training_record,
        experiment_configuration_hash=context.experiment_hash,
        archive_sha256=context.archive_sha256,
    )

    test_days: list[DayMatrices] = []
    seen_days: set[int] = set()
    seen_digests: set[str] = set()
    for day in group["test_days"]:
        day_index = int(day["day_index"])
        if day_index in seen_days:
            raise ValueError(f"frozen contract repeats test day {day_index}")
        record = _require_record(context, int(day["source_fold"]), "testing")
        digest = str(record["file_sha256"])
        if digest != str(day["file_sha256"]):
            raise ValueError(f"test day {day_index} file digest does not match the frozen contract")
        if digest in seen_digests:
            raise ValueError(f"test day {day_index} reuses an already-selected source file")
        if int(record["observation_count"]) != int(day["observations"]):
            raise ValueError(f"test day {day_index} observation count does not match the contract")
        lob, labels = _cache_one(
            root=root,
            extraction_root=context.extraction_root,
            record=record,
            experiment_configuration_hash=context.experiment_hash,
            archive_sha256=context.archive_sha256,
        )
        seen_days.add(day_index)
        seen_digests.add(digest)
        test_days.append(
            DayMatrices(
                day_index=day_index,
                source_fold=int(day["source_fold"]),
                lob=lob[:40],
                features=np.asarray(lob, dtype=np.float32),
                labels=labels,
                file_sha256=digest,
                observation_count=int(record["observation_count"]),
            )
        )

    testing_by_day = {str(day.day_index): day.file_sha256 for day in test_days}
    return DayGroupMatrices(
        day_group=str(day_group),
        training_lob=train_lob[:40],
        training_features=np.asarray(train_lob, dtype=np.float32),
        training_labels=train_labels,
        test_days=tuple(sorted(test_days, key=lambda day: day.day_index)),
        archive_sha256=context.archive_sha256,
        training_file_sha256=str(training_record["file_sha256"]),
        testing_file_sha256=combined_testing_sha256(testing_by_day),
        testing_file_sha256_by_day=testing_by_day,
        data_fingerprint=day_group_fingerprint(
            context.archive_sha256,
            str(day_group),
            str(training_record["file_sha256"]),
            testing_by_day,
        ),
    )


def load_fold(
    root: Path,
    fold: int,
    *,
    experiment_config_path: Path = Path("configs/experiments/fi2010/classical.yaml"),
) -> FoldMatrices:
    """Load one fold exclusively from the audited split manifest."""
    context = _split_context(root, experiment_config_path)
    archive_sha256 = context.archive_sha256
    by_role = {role: _require_record(context, fold, role) for role in ("training", "testing")}
    extraction_root = context.extraction_root
    experiment_hash = context.experiment_hash
    train_lob, train_labels = _cache_one(
        root=root,
        extraction_root=extraction_root,
        record=by_role["training"],
        experiment_configuration_hash=experiment_hash,
        archive_sha256=archive_sha256,
    )
    test_lob, test_labels = _cache_one(
        root=root,
        extraction_root=extraction_root,
        record=by_role["testing"],
        experiment_configuration_hash=experiment_hash,
        archive_sha256=archive_sha256,
    )
    train_features = np.asarray(train_lob, dtype=np.float32)
    test_features = np.asarray(test_lob, dtype=np.float32)
    fingerprint = anchored_fold_fingerprint(
        archive_sha256,
        fold,
        str(by_role["training"]["file_sha256"]),
        str(by_role["testing"]["file_sha256"]),
    )
    return FoldMatrices(
        fold=fold,
        training_lob=train_lob[:40],
        training_features=train_features,
        training_labels=train_labels,
        testing_lob=test_lob[:40],
        testing_features=test_features,
        testing_labels=test_labels,
        archive_sha256=archive_sha256,
        training_file_sha256=str(by_role["training"]["file_sha256"]),
        testing_file_sha256=str(by_role["testing"]["file_sha256"]),
        data_fingerprint=fingerprint,
    )


def labels_for_horizon(labels: np.ndarray, horizon_index: int) -> np.ndarray:
    """Extract one audited source-label row and map it to model indices."""
    if horizon_index not in range(5):
        raise ValueError("horizon_index must be in range 0..4")
    return label_to_index(np.asarray(labels[horizon_index], dtype=np.int8))
