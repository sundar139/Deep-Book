"""Deterministic per-sample prediction artifact I/O and validation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

CLASS_ORDER = ("up", "stationary", "down")
_PROBABILITY_TOLERANCE = 1e-8


def save_prediction_artifact(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    sample_index: np.ndarray,
    source_file_id: np.ndarray,
    day_boundary_id: np.ndarray,
) -> None:
    """Atomically write a validated per-sample prediction artifact."""
    true_vals = np.asarray(y_true, dtype=np.int64)
    pred_vals = np.asarray(y_pred, dtype=np.int64)
    proba = np.asarray(probabilities, dtype=np.float64)

    if true_vals.ndim != 1 or pred_vals.ndim != 1 or proba.ndim != 2:
        raise ValueError("y_true/y_pred must be 1-D; probabilities must be 2-D")
    n = true_vals.shape[0]
    if pred_vals.shape[0] != n or proba.shape[0] != n:
        raise ValueError("array sample counts must match")
    if proba.shape[1] != 3:
        raise ValueError("probabilities must have 3 columns (up, stationary, down)")
    if np.any((true_vals < 0) | (true_vals > 2)) or np.any((pred_vals < 0) | (pred_vals > 2)):
        raise ValueError("model indices must be 0, 1, or 2")
    if not np.isfinite(proba).all() or np.any(proba < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=_PROBABILITY_TOLERANCE):
        raise ValueError("probability rows must sum to 1.0")

    sample_idx = np.asarray(sample_index, dtype=np.int64)
    src_id = np.asarray(source_file_id, dtype=np.int64)
    day_id = np.asarray(day_boundary_id, dtype=np.int64)
    for arr in (sample_idx, src_id, day_id):
        if arr.shape != (n,):
            raise ValueError("auxiliary arrays must match sample count")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        tmp,
        y_true=true_vals,
        y_pred=pred_vals,
        probabilities=proba,
        sample_index=sample_idx,
        source_file_id=src_id,
        day_boundary_id=day_id,
        class_order=np.array(list(CLASS_ORDER)),
    )
    tmp.replace(path)


def load_prediction_artifact(path: Path) -> dict[str, np.ndarray]:
    """Load and validate a prediction artifact."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "y_true",
            "y_pred",
            "probabilities",
            "sample_index",
            "source_file_id",
            "day_boundary_id",
            "class_order",
        }
        missing = required - set(archive.keys())
        if missing:
            raise ValueError(f"prediction artifact missing keys: {sorted(missing)}")
        result: dict[str, np.ndarray] = {}
        for key in required:
            result[key] = np.asarray(archive[key])
    n = result["y_true"].shape[0]
    for key in ("y_pred", "probabilities", "sample_index", "source_file_id", "day_boundary_id"):
        if result[key].shape[0] != n:
            raise ValueError(f"prediction artifact sample count mismatch: {key}")
    if tuple(result["class_order"]) != CLASS_ORDER:
        raise ValueError(
            f"prediction artifact class_order {tuple(result['class_order'])} != {CLASS_ORDER}"
        )
    return result


def sha256_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def recompute_metrics_from_artifact(predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    """Recompute all metrics from a loaded prediction artifact."""
    from deepbook.evaluation.classification import classification_metrics

    return classification_metrics(
        predictions["y_true"],
        predictions["y_pred"],
        predictions["probabilities"],
    )
