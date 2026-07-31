"""Property-based tests for JSON Schema contract invariants.

Uses Hypothesis to generate valid/invalid instances and verify
that schemas correctly accept/reject them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[2]


def _load_schema(name: str) -> dict:
    return json.loads((ROOT / "data_contracts" / name).read_text())


# ---------------------------------------------------------------------------
# Strategy: valid raw events
# ---------------------------------------------------------------------------

_EVENTS_REQUIRING_PRICE = {
    "buyer_initiated_trade",
    "seller_initiated_trade",
    "bid_liquidity_addition",
    "bid_liquidity_removal",
    "ask_liquidity_addition",
    "ask_liquidity_removal",
    "depth_update",
    "top_of_book_update",
}

_EVENTS_REQUIRING_QUANTITY = {"buyer_initiated_trade", "seller_initiated_trade"}


@settings(max_examples=50)
@given(
    event_type=st.sampled_from(sorted(_EVENTS_REQUIRING_PRICE)),
)
def test_events_requiring_price_rejected_without_it(event_type: str) -> None:
    v = Draft202012Validator(_load_schema("raw_event.schema.json"))
    data = {
        "schema_version": 1,
        "venue": "binance",
        "instrument": "BTC-USDT",
        "exchange_timestamp": "2026-07-30T12:00:00Z",
        "receive_timestamp": "2026-07-30T12:00:00Z",
        "event_type": event_type,
    }
    if event_type in _EVENTS_REQUIRING_QUANTITY:
        data["quantity"] = 1.0
    with pytest.raises(ValidationError):
        v.validate(data)


@settings(max_examples=30)
@given(
    event_type=st.sampled_from(["snapshot", "connection_event", "gap", "resync"]),
)
def test_non_depth_events_accepted_without_price(event_type: str) -> None:
    v = Draft202012Validator(_load_schema("raw_event.schema.json"))
    data = {
        "schema_version": 1,
        "venue": "binance",
        "instrument": "BTC-USDT",
        "exchange_timestamp": "2026-07-30T12:00:00Z",
        "receive_timestamp": "2026-07-30T12:00:00Z",
        "event_type": event_type,
    }
    # snapshot needs is_snapshot flag
    if event_type == "snapshot":
        data["is_snapshot"] = True
    try:
        v.validate(data)
    except ValidationError as e:
        pytest.fail(f"Valid event type {event_type} without price rejected: {e}")


# ---------------------------------------------------------------------------
# Strategy: valid book snapshots
# ---------------------------------------------------------------------------

_level_st = st.fixed_dictionaries(
    {
        "price": st.floats(min_value=0.01, max_value=1e8),
        "quantity": st.floats(min_value=0.0, max_value=1e8),
    }
)


@settings(max_examples=30)
@given(
    st.fixed_dictionaries(
        {
            "schema_version": st.just(1),
            "venue": st.just("binance"),
            "instrument": st.just("BTC-USDT"),
            "exchange_timestamp": st.just("2026-07-30T12:00:00Z"),
            "receive_timestamp": st.just("2026-07-30T12:00:00Z"),
            "sequence_id": st.integers(min_value=0),
            "bids": st.lists(_level_st, min_size=1, max_size=20),
            "asks": st.lists(_level_st, min_size=1, max_size=20),
        }
    )
)
def test_generated_valid_book_snapshots_pass(data: dict) -> None:
    v = Draft202012Validator(_load_schema("book_snapshot.schema.json"))
    try:
        v.validate(data)
    except ValidationError as e:
        pytest.fail(f"Valid-looking book snapshot rejected: {e}")


# ---------------------------------------------------------------------------
# Strategy: valid experiment manifests
# ---------------------------------------------------------------------------


@settings(max_examples=30)
@given(
    st.fixed_dictionaries(
        {
            "schema_version": st.just(1),
            "experiment_id": st.text(
                min_size=3,
                max_size=64,
                alphabet=st.characters(whitelist_categories=("L", "N", "P")),
            ),
            "created_utc": st.just("2026-07-30T12:00:00Z"),
            "git_commit": st.just("a" * 40),
            "git_tree_dirty": st.booleans(),
            "command": st.text(min_size=1, max_size=200),
            "random_seeds": st.lists(
                st.integers(min_value=0, max_value=2**31 - 1), min_size=1, max_size=10
            ),
            "status": st.sampled_from(["planned", "running", "completed", "failed", "aborted"]),
        }
    )
)
def test_generated_valid_manifests_pass(data: dict) -> None:
    v = Draft202012Validator(_load_schema("experiment_manifest.schema.json"))
    try:
        v.validate(data)
    except ValidationError as e:
        pytest.fail(f"Valid-looking manifest rejected: {e}")
