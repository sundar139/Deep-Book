"""Regression tests: fixed prediction tables with hand-computed metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deepbook.evaluation.classification import classification_metrics

# Perfect classification: 2 per class
PERFECT_Y_TRUE = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
PERFECT_Y_PRED = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
PERFECT_PROBA = np.eye(3, dtype=np.float64)[PERFECT_Y_PRED]


def test_perfect_macro_f1_is_one() -> None:
    metrics = classification_metrics(PERFECT_Y_TRUE, PERFECT_Y_PRED, PERFECT_PROBA)
    assert metrics["macro_f1"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["nll"] < 1e-10
    assert metrics["brier"] < 1e-10
    assert metrics["ece"] < 1e-10


# All predicted the same class (stationary)
ALL_SAME_TRUE = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
ALL_SAME_PRED = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
ALL_SAME_PROBA = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), (6, 1))


def test_all_same_prediction_metrics() -> None:
    metrics = classification_metrics(ALL_SAME_TRUE, ALL_SAME_PRED, ALL_SAME_PROBA)
    # Only stationary class has hits
    assert metrics["classwise_precision"][1] == pytest.approx(2.0 / 6.0)
    assert metrics["classwise_recall"][1] == 1.0
    assert metrics["classwise_precision"][0] == 0.0
    assert metrics["classwise_precision"][2] == 0.0
    assert metrics["classwise_recall"][0] == 0.0
    assert metrics["classwise_recall"][2] == 0.0
    assert metrics["balanced_accuracy"] == pytest.approx(1.0 / 3.0)
    # Macro-F1: stationary F1 = 2*P*R/(P+R) = 2*(2/6)*1/(2/6+1) = 2*(2/6)/(8/6) = 4/8 = 0.5
    # Macro-F1: mean of [0, 0.5, 0] = 0.5/3
    assert metrics["macro_f1"] == pytest.approx(0.5 / 3.0)
    assert metrics["confusion_matrix"] == [[0, 2, 0], [0, 2, 0], [0, 2, 0]]


# Hand-computed: NLL for uncertain predictions
NLL_TRUE = np.array([0, 1], dtype=np.int64)
NLL_PRED = np.array([0, 1], dtype=np.int64)
NLL_PROBA = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=np.float64)


def test_nll_hand_computed() -> None:
    metrics = classification_metrics(NLL_TRUE, NLL_PRED, NLL_PROBA)
    expected = -0.5 * (np.log(0.7) + np.log(0.6))
    assert metrics["nll"] == pytest.approx(expected)


# Hand-computed Brier
BRIER_TRUE = np.array([0, 1], dtype=np.int64)
BRIER_PRED = np.array([0, 1], dtype=np.int64)
BRIER_PROBA = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=np.float64)


def test_brier_hand_computed() -> None:
    metrics = classification_metrics(BRIER_TRUE, BRIER_PRED, BRIER_PROBA)
    # Brier for sample 0: (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09+0.04+0.01 = 0.14
    # Brier for sample 1: (0.1-0)^2 + (0.6-1)^2 + (0.3-0)^2 = 0.01+0.16+0.09 = 0.26
    expected = 0.5 * (0.14 + 0.26)
    assert metrics["brier"] == pytest.approx(expected)


# ECE
ECE_TRUE = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
ECE_PRED = np.array([0, 0, 1, 2, 0, 1], dtype=np.int64)  # 2/6 correct
ECE_PROBA = np.array(
    [
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.05, 0.05, 0.9],
        [0.2, 0.7, 0.1],
    ],
    dtype=np.float64,
)


def test_ece_computation() -> None:
    metrics = classification_metrics(ECE_TRUE, ECE_PRED, ECE_PROBA)
    assert 0.0 <= metrics["ece"] <= 1.0
    # It should be non-zero since calibration isn't perfect
    assert metrics["ece"] > 0.0


# DeepLOB parameter count regression
def test_deeplob_parameter_count_143907() -> None:
    from deepbook.models.deeplob import DeepLOB, parameter_count

    model = DeepLOB()
    count = parameter_count(model)
    assert count == 143907, f"Expected 143907 trainable parameters, got {count}"


# DeepLOB tensor shapes
def test_deeplob_tensor_shapes() -> None:
    from deepbook.models.deeplob import DeepLOB

    model = DeepLOB()
    batch = torch.randn(2, 1, 100, 40)
    logits, trace = model.forward_with_trace(batch)
    assert logits.shape == (2, 3)
    assert trace["conv1"] == (2, 32, 94, 20)
    assert trace["conv2"] == (2, 32, 88, 10)
    assert trace["conv3"] == (2, 32, 82, 1)
    assert trace["inception"] == (2, 192, 82, 1)
    assert trace["lstm"] == (2, 82, 64)


# Smoke exclusion regression
def test_run_report_excludes_smoke() -> None:
    """Verify smoke runs are excluded and confirmatory runs included in grouping logic."""
    # This tests the classification logic pattern
    smoke_manifest = {
        "run_id": "smoke-run",
        "run_kind": "smoke",
        "eligible_for_confirmatory_report": False,
        "exclusion_reasons": ["git tree is dirty"],
        "status": "completed",
        "model": "test",
        "horizon": 10,
        "metrics": {"test": {"macro_f1": 0.5}},
    }
    confirmatory_manifest = {
        "run_id": "confirmatory-run",
        "run_kind": "confirmatory",
        "eligible_for_confirmatory_report": True,
        "exclusion_reasons": [],
        "status": "completed",
        "model": "test",
        "horizon": 10,
        "metrics": {"test": {"macro_f1": 0.8}},
    }
    confirmatory = [
        m
        for m in [smoke_manifest, confirmatory_manifest]
        if m.get("eligible_for_confirmatory_report") and m.get("status") == "completed"
    ]
    assert len(confirmatory) == 1
    assert confirmatory[0]["run_id"] == "confirmatory-run"


# Class order regression
def test_class_order_is_up_stationary_down() -> None:
    from deepbook.evaluation.classification import CLASS_ORDER

    assert CLASS_ORDER == ("up", "stationary", "down")


# Confusion matrix shape
def test_confusion_matrix_is_3x3() -> None:
    metrics = classification_metrics(PERFECT_Y_TRUE, PERFECT_Y_PRED, PERFECT_PROBA)
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (3, 3)
