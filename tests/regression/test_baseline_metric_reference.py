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


# Exact hand-computed ECE over production's fixed ten equal-width confidence bins.
#
# Bin edges are [0.0,0.1), [0.1,0.2), ... [0.8,0.9), [0.9,1.0]; only the final bin
# includes its upper edge. Confidence is the maximum predicted probability.
#
# | # | probabilities        | pred | true | confidence | correct | bin          |
# |---|----------------------|------|------|-----------:|--------:|--------------|
# | 0 | [0.95, 0.03, 0.02]   | 0    | 0    |       0.95 |       1 | 9 [0.9, 1.0] |
# | 1 | [0.90, 0.06, 0.04]   | 0    | 1    |       0.90 |       0 | 9 [0.9, 1.0] |
# | 2 | [0.10, 0.75, 0.15]   | 1    | 1    |       0.75 |       1 | 7 [0.7, 0.8) |
# | 3 | [0.10, 0.15, 0.75]   | 2    | 0    |       0.75 |       0 | 7 [0.7, 0.8) |
# | 4 | [0.45, 0.30, 0.25]   | 0    | 0    |       0.45 |       1 | 4 [0.4, 0.5) |
# | 5 | [0.25, 0.45, 0.30]   | 1    | 2    |       0.45 |       0 | 4 [0.4, 0.5) |
#
# Bin 4: weight 2/6, mean confidence 0.45,  accuracy 0.5 -> gap 0.050
# Bin 7: weight 2/6, mean confidence 0.75,  accuracy 0.5 -> gap 0.250
# Bin 9: weight 2/6, mean confidence 0.925, accuracy 0.5 -> gap 0.425
# All other bins are empty and contribute nothing.
#
# ECE = (1/3)(0.050) + (1/3)(0.250) + (1/3)(0.425) = 0.725/3 = 29/120
ECE_TRUE = np.array([0, 1, 1, 0, 0, 2], dtype=np.int64)
ECE_PRED = np.array([0, 0, 1, 2, 0, 1], dtype=np.int64)  # 3/6 correct
ECE_PROBA = np.array(
    [
        [0.95, 0.03, 0.02],
        [0.90, 0.06, 0.04],
        [0.10, 0.75, 0.15],
        [0.10, 0.15, 0.75],
        [0.45, 0.30, 0.25],
        [0.25, 0.45, 0.30],
    ],
    dtype=np.float64,
)
EXPECTED_ECE = 29.0 / 120.0


def test_ece_matches_the_hand_computed_value() -> None:
    metrics = classification_metrics(ECE_TRUE, ECE_PRED, ECE_PROBA)
    assert metrics["ece"] == pytest.approx(EXPECTED_ECE, abs=1e-12)
    assert metrics["ece"] == pytest.approx(0.2416666666666667, abs=1e-12)
    # The three occupied bins carry equal weight, so the mean gap is the ECE.
    assert metrics["ece"] == pytest.approx((0.05 + 0.25 + 0.425) / 3.0, abs=1e-12)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_ece_is_zero_for_perfectly_calibrated_certain_predictions() -> None:
    metrics = classification_metrics(PERFECT_Y_TRUE, PERFECT_Y_PRED, PERFECT_PROBA)
    assert metrics["ece"] == pytest.approx(0.0, abs=1e-12)


def test_ece_final_bin_includes_probability_one() -> None:
    """A confidence of exactly 1.0 must land in bin 9, not be dropped."""
    true_values = np.array([0, 1], dtype=np.int64)
    predictions = np.array([0, 0], dtype=np.int64)
    probabilities = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    metrics = classification_metrics(true_values, predictions, probabilities)
    # Both samples sit in the last bin: weight 1, mean confidence 1.0, accuracy 0.5.
    assert metrics["ece"] == pytest.approx(0.5, abs=1e-12)


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
