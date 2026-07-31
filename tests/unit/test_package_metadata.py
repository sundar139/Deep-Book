"""Tests for package metadata (version, import)."""

from __future__ import annotations


def test_package_imports() -> None:
    import deepbook

    assert deepbook is not None


def test_package_has_version() -> None:
    import deepbook

    assert isinstance(deepbook.__version__, str)
    assert len(deepbook.__version__) > 0
    # version should be parseable as semver-like
    parts = deepbook.__version__.split(".")
    assert len(parts) >= 2


def test_package_exposes_validation() -> None:
    from deepbook.validation import invariants

    assert hasattr(invariants, "check_book")
    assert hasattr(invariants, "InvariantViolation")


def test_package_exposes_eval() -> None:
    from deepbook.eval import stats

    assert hasattr(stats, "paired_bootstrap_ci")
    assert hasattr(stats, "diebold_mariano")


def test_package_exposes_features() -> None:
    from deepbook.features import hawkes

    assert hasattr(hawkes, "simulate_hawkes")
    assert hasattr(hawkes, "fit_hawkes_mle")
