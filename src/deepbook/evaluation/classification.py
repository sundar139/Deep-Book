"""Classification metrics with fixed FI-2010 class ordering."""

from __future__ import annotations

from typing import Any

import numpy as np

CLASS_COUNT = 3
CLASS_ORDER = ("up", "stationary", "down")


def _as_labels(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels.size == 0 or np.any((labels < 0) | (labels >= CLASS_COUNT)):
        raise ValueError("labels must be non-empty model indices 0, 1, or 2")
    return labels


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Compute deterministic multiclass metrics for classes up, stationary, down."""
    true = _as_labels(y_true)
    predicted = _as_labels(y_pred)
    if predicted.shape != true.shape:
        raise ValueError("predictions and labels must have the same shape")
    proba = np.asarray(probabilities, dtype=np.float64)
    if proba.shape != (true.size, CLASS_COUNT):
        raise ValueError("probabilities must have shape (sample_count, 3)")
    if not np.isfinite(proba).all() or np.any(proba < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("probability rows must sum to one")

    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(confusion, (true, predicted), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision_denominator = confusion.sum(axis=0).astype(np.float64)
    recall_denominator = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.zeros(CLASS_COUNT, dtype=np.float64),
        where=precision_denominator != 0,
    )
    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.zeros(CLASS_COUNT, dtype=np.float64),
        where=recall_denominator != 0,
    )
    f1_denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        f1_denominator,
        out=np.zeros(CLASS_COUNT, dtype=np.float64),
        where=f1_denominator != 0,
    )
    total = float(true.size)
    predicted_counts = confusion.sum(axis=0).astype(np.float64)
    true_counts = confusion.sum(axis=1).astype(np.float64)
    numerator = total * float(true_positive.sum()) - float(np.dot(predicted_counts, true_counts))
    denominator = float(
        np.sqrt(
            (total * total - float(np.dot(predicted_counts, predicted_counts)))
            * (total * total - float(np.dot(true_counts, true_counts)))
        )
    )
    mcc = numerator / denominator if denominator else 0.0

    clipped = np.clip(proba, 1e-15, 1.0)
    nll = float(-np.mean(np.log(clipped[np.arange(true.size), true])))
    one_hot = np.eye(CLASS_COUNT, dtype=np.float64)[true]
    brier = float(np.mean(np.sum(np.square(proba - one_hot), axis=1)))
    confidence = np.max(proba, axis=1)
    correctness = (predicted == true).astype(np.float64)
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10.0
        upper = (bin_index + 1) / 10.0
        mask = (confidence >= lower) & (
            confidence <= upper if bin_index == 9 else confidence < upper
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(confidence[mask].mean()) - float(correctness[mask].mean())
            )

    return {
        "class_order": list(CLASS_ORDER),
        "sample_count": int(true.size),
        "accuracy": float(np.mean(correctness)),
        "macro_f1": float(f1.mean()),
        "mcc": float(mcc),
        "balanced_accuracy": float(recall.mean()),
        "classwise_precision": precision.tolist(),
        "classwise_recall": recall.tolist(),
        "classwise_f1": f1.tolist(),
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
        "confusion_matrix": confusion.tolist(),
    }
