"""Shared test configuration."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config, items):
    """Auto-skip data-dependent tests when the data isn't present."""
    data_root = Path(os.environ.get("DEEPBOOK_DATA", ROOT / "data"))
    for item in items:
        if "data" in item.keywords and not data_root.exists():
            item.add_marker(
                item.config.getoption("-m", "")  # ponytail: simple skip reason
            )
            import pytest

            item.add_marker(pytest.mark.skip(reason=f"no data at {data_root}"))
