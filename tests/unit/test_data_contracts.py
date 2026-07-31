"""Tests for JSON Schema data contracts.

Validates valid/invalid examples for raw events, book snapshots,
and experiment manifests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[2]


def _load_schema(name: str) -> dict:
    path = ROOT / "data_contracts" / name
    return json.loads(path.read_text())


def _validator(name: str) -> Draft202012Validator:
    schema = _load_schema(name)
    return Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# Raw event schema
# ---------------------------------------------------------------------------

RAW_SCHEMA = "raw_event.schema.json"


def _valid_raw_event(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "venue": "binance",
        "instrument": "BTC-USDT",
        "exchange_timestamp": "2026-07-30T12:00:00Z",
        "receive_timestamp": "2026-07-30T12:00:01Z",
        "event_type": "bid_liquidity_addition",
        "price": 65000.0,
        "quantity": 0.5,
    }
    base.update(overrides)
    return base


def test_valid_raw_event_passes() -> None:
    v = _validator(RAW_SCHEMA)
    v.validate(_valid_raw_event())


def test_valid_trade_event_passes() -> None:
    v = _validator(RAW_SCHEMA)
    v.validate(
        _valid_raw_event(
            event_type="buyer_initiated_trade",
            price=65000.0,
            quantity=1.0,
            trade_id="trade_001",
        )
    )


def test_minimal_event_with_required_only_passes() -> None:
    v = _validator(RAW_SCHEMA)
    v.validate(
        {
            "schema_version": 1,
            "venue": "coinbase",
            "instrument": "ETH-USD",
            "exchange_timestamp": "2026-07-30T12:00:00Z",
            "receive_timestamp": "2026-07-30T12:00:00Z",
            "event_type": "snapshot",
            "is_snapshot": True,
        }
    )


@pytest.mark.parametrize(
    "overrides,missing_key",
    [
        ({"schema_version": None}, "schema_version"),
        ({"venue": ""}, "venue"),
        ({"event_type": "nonexistent_event"}, "event_type"),
    ],
)
def test_invalid_raw_events_rejected(overrides: dict, missing_key: str) -> None:
    v = _validator(RAW_SCHEMA)
    data = _valid_raw_event()
    data.update(overrides)
    with pytest.raises(ValidationError):
        v.validate(data)


def test_trade_event_without_price_rejected() -> None:
    v = _validator(RAW_SCHEMA)
    data = _valid_raw_event(event_type="buyer_initiated_trade")
    del data["price"]
    with pytest.raises(ValidationError):
        v.validate(data)


def test_negative_price_rejected() -> None:
    v = _validator(RAW_SCHEMA)
    data = _valid_raw_event(price=-1.0)
    with pytest.raises(ValidationError):
        v.validate(data)


def test_depth_event_without_price_rejected() -> None:
    """Non-snapshot depth events require a price."""
    v = _validator(RAW_SCHEMA)
    data = _valid_raw_event(event_type="depth_update")
    del data["price"]
    with pytest.raises(ValidationError):
        v.validate(data)


def test_snapshot_without_price_accepted() -> None:
    """Snapshots are exempt from the price requirement."""
    v = _validator(RAW_SCHEMA)
    v.validate(
        {
            "schema_version": 1,
            "venue": "binance",
            "instrument": "BTC-USDT",
            "exchange_timestamp": "2026-07-30T12:00:00Z",
            "receive_timestamp": "2026-07-30T12:00:00Z",
            "event_type": "snapshot",
            "is_snapshot": True,
        }
    )


def test_connection_event_without_price_accepted() -> None:
    """Connection/gap/resync events do not require a price."""
    v = _validator(RAW_SCHEMA)
    v.validate(
        {
            "schema_version": 1,
            "venue": "binance",
            "instrument": "BTC-USDT",
            "exchange_timestamp": "2026-07-30T12:00:00Z",
            "receive_timestamp": "2026-07-30T12:00:00Z",
            "event_type": "connection_event",
        }
    )


# ---------------------------------------------------------------------------
# Book snapshot schema
# ---------------------------------------------------------------------------

BOOK_SCHEMA = "book_snapshot.schema.json"


def _valid_book_snapshot(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "venue": "binance",
        "instrument": "BTC-USDT",
        "exchange_timestamp": "2026-07-30T12:00:00Z",
        "receive_timestamp": "2026-07-30T12:00:00Z",
        "sequence_id": 1000,
        "bids": [
            {"price": 65000.0, "quantity": 1.0},
            {"price": 64999.0, "quantity": 2.0},
        ],
        "asks": [
            {"price": 65001.0, "quantity": 0.5},
            {"price": 65002.0, "quantity": 1.5},
        ],
    }
    base.update(overrides)
    return base


def test_valid_book_snapshot_passes() -> None:
    v = _validator(BOOK_SCHEMA)
    v.validate(_valid_book_snapshot())


def test_book_with_gap_flag_passes() -> None:
    v = _validator(BOOK_SCHEMA)
    v.validate(_valid_book_snapshot(has_gap=True, has_corruption=False))


def test_book_without_bids_rejected() -> None:
    v = _validator(BOOK_SCHEMA)
    data = _valid_book_snapshot()
    del data["bids"]
    with pytest.raises(ValidationError):
        v.validate(data)


def test_book_with_empty_bids_rejected() -> None:
    v = _validator(BOOK_SCHEMA)
    data = _valid_book_snapshot(bids=[])
    with pytest.raises(ValidationError):
        v.validate(data)


def test_book_with_negative_quantity_rejected() -> None:
    v = _validator(BOOK_SCHEMA)
    data = _valid_book_snapshot()
    data["bids"][0]["quantity"] = -1.0
    with pytest.raises(ValidationError):
        v.validate(data)


# ---------------------------------------------------------------------------
# Experiment manifest schema
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA = "experiment_manifest.schema.json"


def _valid_manifest(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "experiment_id": "test-exp-001",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": "a" * 40,
        "git_tree_dirty": False,
        "command": "python -m deepbook.experiment --config test.yaml",
        "random_seeds": [42],
        "status": "completed",
    }
    base.update(overrides)
    return base


def test_valid_manifest_passes() -> None:
    v = _validator(MANIFEST_SCHEMA)
    v.validate(_valid_manifest())


def test_manifest_without_experiment_id_rejected() -> None:
    v = _validator(MANIFEST_SCHEMA)
    data = _valid_manifest()
    del data["experiment_id"]
    with pytest.raises(ValidationError):
        v.validate(data)


def test_manifest_empty_seeds_rejected() -> None:
    v = _validator(MANIFEST_SCHEMA)
    data = _valid_manifest(random_seeds=[])
    with pytest.raises(ValidationError):
        v.validate(data)


def test_manifest_invalid_status_rejected() -> None:
    v = _validator(MANIFEST_SCHEMA)
    data = _valid_manifest(status="fictional_status")
    with pytest.raises(ValidationError):
        v.validate(data)


def test_manifest_planned_status_accepted() -> None:
    v = _validator(MANIFEST_SCHEMA)
    v.validate(_valid_manifest(status="planned"))


def test_example_manifest_validates() -> None:
    """The committed example manifest must validate against the schema."""
    v = _validator(MANIFEST_SCHEMA)
    example = json.loads((ROOT / "experiments" / "examples" / "example_manifest.json").read_text())
    v.validate(example)
