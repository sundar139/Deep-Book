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
    ModelingCacheSpec,
    configuration_hash,
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


def load_fold(
    root: Path,
    fold: int,
    *,
    experiment_config_path: Path = Path("configs/experiments/fi2010/classical.yaml"),
) -> FoldMatrices:
    """Load one fold exclusively from the audited split manifest."""
    experiment_path = root / experiment_config_path
    experiment = _read_yaml(experiment_path)
    source_config_path = root / str(experiment["source_config"])
    source_config = _read_yaml(source_config_path)
    archive_sha256 = str(source_config["source"]["archive_sha256"])
    variant = str(experiment["benchmark_variant"])
    normalization = str(experiment["normalization"])
    split_path = root / "data" / "interim" / "fi2010" / "fi2010_split_manifest.json"
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in split_manifest["splits"]
        if int(record["fold_identifier"]) == fold
        and record["benchmark_variant"] == variant
        and record["normalization"] == normalization
    ]
    by_role = {str(record["role"]): record for record in records}
    if set(by_role) != {"training", "testing"}:
        raise ValueError(f"audited fold {fold} does not have one training and one testing record")
    extraction_root = root / "data" / "raw" / "fi2010" / "extracted" / archive_sha256
    experiment_hash = configuration_hash(experiment)
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
    payload = {
        "archive_sha256": archive_sha256,
        "fold": fold,
        "training_file_sha256": by_role["training"]["file_sha256"],
        "testing_file_sha256": by_role["testing"]["file_sha256"],
    }
    fingerprint = configuration_hash(payload)
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
