"""Tests for src/validation/invariants.py.

Note the pattern used throughout: every check is tested twice -- once on clean
data (must pass) and once on *deliberately corrupted* data (must fail). A check
that has never been observed to fail is not a check, it is decoration.
"""

from __future__ import annotations

import numpy as np
import pytest

from deepbook.validation.invariants import (
    InvariantViolation,
    check_book,
    check_chronological,
    check_execution_accounting,
    check_horizon_unbiased,
    check_no_lookahead_fills,
    check_normalization_fit_on_train_only,
    check_purged_split,
    check_timestamps,
    class_balance,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Book structure
# --------------------------------------------------------------------------- #


def test_clean_book_passes(synthetic_book):
    bid_px, bid_sz, ask_px, ask_sz, _ = synthetic_book
    assert check_book(bid_px, bid_sz, ask_px, ask_sz).ok


def test_crossed_book_is_detected(synthetic_book):
    bid_px, bid_sz, ask_px, ask_sz, _ = synthetic_book
    bid_px = bid_px.copy()
    bid_px[17, 0] = ask_px[17, 0] + 0.05  # cross the book
    rep = check_book(bid_px, bid_sz, ask_px, ask_sz)
    assert not rep.ok and any("crossed" in v for v in rep.violations)


def test_non_monotone_levels_are_detected(synthetic_book):
    bid_px, bid_sz, ask_px, ask_sz, _ = synthetic_book
    ask_px = ask_px.copy()
    ask_px[42, 3] = ask_px[42, 2] - 0.02  # level 3 cheaper than level 2
    rep = check_book(bid_px, bid_sz, ask_px, ask_sz)
    assert not rep.ok and any("monotone" in v for v in rep.violations)


def test_empty_touch_is_detected(synthetic_book):
    bid_px, bid_sz, ask_px, ask_sz, _ = synthetic_book
    bid_sz = bid_sz.copy()
    bid_sz[5, 0] = 0.0
    rep = check_book(bid_px, bid_sz, ask_px, ask_sz)
    assert not rep.ok


def test_report_raises_with_context(synthetic_book):
    bid_px, bid_sz, ask_px, ask_sz, _ = synthetic_book
    bid_px = bid_px.copy()
    bid_px[0, 0] = ask_px[0, 0] + 1.0
    with pytest.raises(InvariantViolation, match="BTCUSDT"):
        check_book(bid_px, bid_sz, ask_px, ask_sz).raise_if_failed("BTCUSDT 2026-07-01")


def test_out_of_order_timestamps_detected(synthetic_book):
    *_, ts = synthetic_book
    assert check_timestamps(ts).ok
    bad = ts.copy()
    bad[100], bad[101] = bad[101], bad[100]
    assert not check_timestamps(bad).ok


# --------------------------------------------------------------------------- #
# Split hygiene -- the leakage guards
# --------------------------------------------------------------------------- #


def test_purged_split_accepts_a_correctly_gapped_split():
    window, horizon, embargo = 128, 20, 50
    gap = window + horizon + embargo
    train = np.arange(0, 5000)
    test = np.arange(5000 + gap, 8000)
    assert check_purged_split(train, test, window, horizon, embargo).ok


def test_purged_split_rejects_an_adjacent_split():
    """The naive split: train ends where test begins. Every overlapping window
    straddling the boundary leaks the test period's future into training.
    """
    window, horizon = 128, 20
    train = np.arange(0, 5000)
    test = np.arange(5000, 8000)
    rep = check_purged_split(train, test, window, horizon, embargo=0)
    assert not rep.ok and "within" in rep.violations[0]


def test_purged_split_rejects_overlapping_indices():
    train = np.arange(0, 5000)
    test = np.arange(4900, 8000)
    rep = check_purged_split(train, test, 10, 5)
    assert not rep.ok


def test_chronological_check_rejects_shuffled_splits(rng):
    idx = np.arange(10000)
    rng.shuffle(idx)
    train, test = idx[:8000], idx[8000:]
    assert not check_chronological(train, test).ok
    assert check_chronological(np.arange(8000), np.arange(8000, 10000)).ok


def test_normalization_must_be_fit_on_train_only(rng):
    full = rng.normal(0, 1, size=10000)
    train = full[:8000]

    good = {"mean": train.mean(), "std": train.std()}
    recomputed = {"mean": train.mean(), "std": train.std()}
    assert check_normalization_fit_on_train_only(good, recomputed).ok

    leaky = {"mean": full.mean(), "std": full.std()}  # fit before the split
    assert not check_normalization_fit_on_train_only(leaky, recomputed).ok


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def test_class_balance_sums_to_one(rng):
    y = rng.integers(0, 3, size=1000)
    b = class_balance(y)
    assert np.isclose(b.sum(), 1.0) and b.size == 3


def test_horizon_bias_is_detected():
    """Fixed threshold: the 'stationary' class shrinks as the horizon grows."""
    biased = {
        10: np.array([0.20, 0.60, 0.20]),
        50: np.array([0.35, 0.30, 0.35]),
        100: np.array([0.42, 0.16, 0.42]),
    }
    assert not check_horizon_unbiased(biased, tol=0.10).ok

    unbiased = {
        10: np.array([0.33, 0.34, 0.33]),
        50: np.array([0.34, 0.32, 0.34]),
        100: np.array([0.35, 0.31, 0.34]),
    }
    assert check_horizon_unbiased(unbiased, tol=0.10).ok


# --------------------------------------------------------------------------- #
# Execution accounting
# --------------------------------------------------------------------------- #


def test_execution_accounting_identity_holds():
    qty = np.array([200.0, 300.0, 500.0])
    px = np.array([100.02, 100.05, 100.10])
    arrival = 100.00
    vwap = float(np.dot(qty, px) / qty.sum())
    bps = (vwap - arrival) / arrival * 1e4
    assert check_execution_accounting(qty, px, 1000.0, arrival, bps).ok


def test_under_execution_is_detected():
    """The agent that 'beats TWAP' by quietly not finishing the order."""
    qty = np.array([200.0, 300.0])
    px = np.array([100.01, 100.02])
    vwap = float(np.dot(qty, px) / qty.sum())
    bps = (vwap - 100.0) / 100.0 * 1e4
    rep = check_execution_accounting(qty, px, 1000.0, 100.0, bps)
    assert not rep.ok and "not completed" in rep.violations[0]


def test_reward_metric_drift_is_detected():
    qty = np.array([500.0, 500.0])
    px = np.array([100.05, 100.05])
    rep = check_execution_accounting(qty, px, 1000.0, 100.0, reported_shortfall_bps=1.0)
    assert not rep.ok and "shortfall" in rep.violations[-1]


def test_lookahead_fill_is_detected():
    decisions = np.array([1.0, 2.0, 3.0])
    good_fills = decisions + 0.01
    assert check_no_lookahead_fills(decisions, good_fills, min_latency_s=0.005).ok

    same_tick = decisions.copy()  # matched against the tick that triggered it
    rep = check_no_lookahead_fills(decisions, same_tick, min_latency_s=0.005)
    assert not rep.ok
