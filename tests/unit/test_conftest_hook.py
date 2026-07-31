"""Tests for the data-marker skip hook."""

from __future__ import annotations

from pathlib import Path

from conftest import _skip_missing_data


class FakeItem:
    """Minimal collected-item shape used by the hook."""

    def __init__(self, *keywords: str) -> None:
        self.keywords = set(keywords)
        self.markers: list[object] = []

    def add_marker(self, marker: object) -> None:
        self.markers.append(marker)


def test_data_marker_skipped_when_data_absent(tmp_path: Path) -> None:
    item = FakeItem("data")

    _skip_missing_data([item], tmp_path / "missing")

    assert len(item.markers) == 1
    assert item.markers[0].mark.name == "skip"  # type: ignore[attr-defined]


def test_normal_test_not_skipped(tmp_path: Path) -> None:
    item = FakeItem()

    _skip_missing_data([item], tmp_path / "missing")

    assert item.markers == []


def test_data_marker_not_skipped_when_data_present(tmp_path: Path) -> None:
    item = FakeItem("data")
    data_root = tmp_path / "data"
    data_root.mkdir()

    _skip_missing_data([item], data_root)

    assert item.markers == []


def test_hook_handles_mixed_collection_without_error(tmp_path: Path) -> None:
    data_item = FakeItem("data")
    ordinary_item = FakeItem()

    _skip_missing_data([data_item, ordinary_item], tmp_path / "missing")

    assert len(data_item.markers) == 1
    assert ordinary_item.markers == []
