"""Tests for package metadata (version, import)."""

from __future__ import annotations


def test_package_imports() -> None:
    import deepbook

    assert deepbook is not None


def test_package_has_version() -> None:
    import deepbook

    assert isinstance(deepbook.__version__, str)
    assert len(deepbook.__version__) > 0
    parts = deepbook.__version__.split(".")
    assert len(parts) >= 2


def test_version_via_importlib_matches() -> None:
    from importlib.metadata import version

    import deepbook

    assert version("deepbook") == deepbook.__version__
