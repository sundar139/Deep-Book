from __future__ import annotations

import numpy as np

from deepbook.evaluation.classification import classification_metrics


def test_classification_metrics_match_hand_calculation() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2])
    predictions = np.array([0, 1, 1, 0, 2, 2])
    probabilities = (
        np.eye(3, dtype=np.float64)[predictions] * 0.9
        + (1.0 - np.eye(3, dtype=np.float64)[predictions]) * 0.05
    )

    metrics = classification_metrics(labels, predictions, probabilities)

    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 1, 1], [0, 1, 1]]
    assert metrics["accuracy"] == 4 / 6
    assert 0.0 < metrics["macro_f1"] < 1.0
    assert -1.0 <= metrics["mcc"] <= 1.0
    assert metrics["mcc"] == 0.5
    assert metrics["classwise_precision"] == [1.0, 0.5, 0.5]
    assert metrics["classwise_recall"] == [1.0, 0.5, 0.5]
    assert metrics["nll"] > 0.0
    assert 0.0 <= metrics["brier"] <= 2.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_probability_rows_are_validated_and_clipped() -> None:
    labels = np.array([0, 1, 2])
    predictions = np.array([0, 1, 2])
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    metrics = classification_metrics(labels, predictions, probabilities)
    assert metrics["nll"] >= 0.0
    assert metrics["brier"] == 0.0
    assert metrics["ece"] == 0.0
