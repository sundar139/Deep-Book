from __future__ import annotations

import numpy as np

from deepbook.models.classical import (
    CausalMovementPersistence,
    MajorityClassifier,
    MultinomialLogistic,
    RandomForestBaseline,
)


def test_majority_uses_training_labels_only_and_returns_probabilities() -> None:
    model = MajorityClassifier(smoothing=1e-6).fit(np.array([0, 0, 1, 2, 0]))
    assert model.predict(np.zeros(4, dtype=np.float32)).tolist() == [0, 0, 0, 0]
    probabilities = model.predict_proba(np.zeros((4, 2), dtype=np.float32))
    assert probabilities.shape == (4, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_causal_persistence_uses_horizon_lag() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
    predictions = CausalMovementPersistence(horizon=2).predict(labels)
    assert predictions.tolist() == [-1, -1, 0, 1, 2, 0]


def test_logistic_coefficients_and_probabilities_have_three_classes() -> None:
    rng = np.random.default_rng(1337)
    x = rng.normal(size=(30, 4))
    y = np.repeat(np.arange(3), 10)
    model = MultinomialLogistic(max_iter=100).fit(x, y)
    assert model.coef_.shape == (3, 4)
    assert model.intercept_.shape == (3,)
    assert model.predict_proba(x[:3]).shape == (3, 3)


def test_tree_baseline_is_bounded_and_probabilistic() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 4))
    y = np.repeat(np.arange(3), 10)
    model = RandomForestBaseline(n_estimators=5, max_training_rows=20, random_state=7).fit(x, y)
    assert model.training_rows_ == 20
    assert model.predict_proba(x[:3]).shape == (3, 3)
