"""Small scikit-learn FI-2010 classification baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class MajorityClassifier:
    """Training-label majority predictor with smoothed probabilities."""

    def __init__(self, smoothing: float = 1e-6) -> None:
        if smoothing < 0.0:
            raise ValueError("smoothing must be non-negative")
        self.smoothing = float(smoothing)
        self.class_probabilities_: np.ndarray | None = None
        self.class_: int | None = None

    def fit(self, labels: np.ndarray) -> MajorityClassifier:
        """Fit class frequencies using only the supplied training labels."""
        values = np.asarray(labels, dtype=np.int64)
        if values.ndim != 1 or values.size == 0 or np.any((values < 0) | (values > 2)):
            raise ValueError("labels must be non-empty model indices")
        counts = np.bincount(values, minlength=3).astype(np.float64) + self.smoothing
        self.class_probabilities_ = counts / counts.sum()
        self.class_ = int(np.argmax(counts))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict the fitted majority class for each feature row."""
        if self.class_ is None:
            raise RuntimeError("classifier is not fitted")
        return np.full(len(features), self.class_, dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return the fitted smoothed class distribution for each row."""
        if self.class_probabilities_ is None:
            raise RuntimeError("classifier is not fitted")
        return np.repeat(self.class_probabilities_[None, :], len(features), axis=0)


class CausalMovementPersistence:
    """Persist a supplied movement only after its horizon interval ends."""

    def __init__(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)

    def predict(self, labels: np.ndarray) -> np.ndarray:
        """Return prior model-index labels at the conservative horizon lag."""
        values = np.asarray(labels, dtype=np.int64)
        if values.ndim != 1 or np.any((values < 0) | (values > 2)):
            raise ValueError("labels must be one-dimensional model indices")
        predictions = np.full(values.shape, -1, dtype=np.int64)
        if values.size > self.horizon:
            predictions[self.horizon :] = values[: -self.horizon]
        return predictions


class MultinomialLogistic:
    """Fixed multinomial logistic regression with probability output."""

    def __init__(self, max_iter: int = 200, random_state: int = 1337) -> None:
        self.estimator = LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> MultinomialLogistic:
        """Fit the multinomial estimator on the supplied training rows."""
        self.estimator.fit(features, labels)
        return self

    @property
    def coef_(self) -> np.ndarray:
        """Return fitted class coefficients."""
        return np.asarray(self.estimator.coef_)

    @property
    def intercept_(self) -> np.ndarray:
        """Return fitted class intercepts."""
        return np.asarray(self.estimator.intercept_)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict model-index classes for feature rows."""
        return np.asarray(self.estimator.predict(features), dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return probabilities in the fixed three-class order."""
        return _three_class_probabilities(
            self.estimator.predict_proba(features), self.estimator.classes_
        )


class RandomForestBaseline:
    """Bounded chronological RandomForest baseline with class probabilities."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 18,
        min_samples_leaf: int = 2,
        max_training_rows: int | None = 100_000,
        random_state: int = 1337,
    ) -> None:
        self.max_training_rows = max_training_rows
        self.random_state = random_state
        self.estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=1,
        )
        self.training_rows_: int | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> RandomForestBaseline:
        """Fit on a deterministic chronological prefix of training rows."""
        if self.max_training_rows is None:
            selected = np.arange(len(labels))
        else:
            selected = np.arange(min(len(labels), self.max_training_rows))
        self.estimator.fit(features[selected], labels[selected])
        self.training_rows_ = int(selected.size)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict model-index classes for feature rows."""
        return np.asarray(self.estimator.predict(features), dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return probabilities in the fixed three-class order."""
        return _three_class_probabilities(
            self.estimator.predict_proba(features), self.estimator.classes_
        )


def _three_class_probabilities(probabilities: Any, classes: Any) -> np.ndarray:
    result = np.zeros((len(probabilities), 3), dtype=np.float64)
    for column, label in enumerate(np.asarray(classes, dtype=np.int64)):
        result[:, int(label)] = np.asarray(probabilities)[:, column]
    return result
