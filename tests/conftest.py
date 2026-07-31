"""Shared test configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _skip_missing_data(items, data_root: Path) -> None:
    """Skip data-marked tests only when their local data root is absent."""
    if data_root.exists():
        return
    for item in items:
        if "data" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="required local research data is unavailable"))


def pytest_collection_modifyitems(config, items):
    """Apply the missing-data skip policy to collected tests."""
    data_root = Path(os.environ.get("DEEPBOOK_DATA", ROOT / "data"))
    _skip_missing_data(items, data_root)
