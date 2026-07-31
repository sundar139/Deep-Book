"""Invariants for LOB data and for the train/test machinery.

These are deliberately importable from *both* the test suite and the pipeline
itself: run them as assertions in CI on a fixture, and as sampled runtime checks
on every batch of reconstructed data. A silent book-reconstruction bug is the
kind of thing that produces a beautiful, completely fake result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


class InvariantViolation(AssertionError):
    """Raised when data violates a structural guarantee it must satisfy."""


@dataclass
class Report:
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, msg: str) -> None:
        self.violations.append(msg)

    def raise_if_failed(self, context: str = "") -> None:
        if self.violations:
            head = f"{context}: " if context else ""
            raise InvariantViolation(head + "; ".join(self.violations[:20]))


# --------------------------------------------------------------------------- #
# Order-book structure
# --------------------------------------------------------------------------- #


def check_book(
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    allow_empty_levels: bool = True,
) -> Report:
    """Structural checks on a batch of order-book snapshots.

    All arrays are (n_snapshots, n_levels), level 0 = best.

    ``allow_empty_levels`` covers LOBSTER's dummy padding: when fewer levels are
    occupied than requested, the empty slots are filled with placeholder values.
    Those rows are excluded from the monotonicity check rather than silently
    passed.
    """
    rep = Report()
    arrays = {"bid_px": bid_px, "bid_sz": bid_sz, "ask_px": ask_px, "ask_sz": ask_sz}

    shapes = {k: np.asarray(v).shape for k, v in arrays.items()}
    if len(set(shapes.values())) != 1:
        rep.add(f"shape mismatch {shapes}")
        return rep

    for name, arr in arrays.items():
        arr = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(arr)):
            rep.add(f"{name} contains non-finite values")

    bid_px = np.asarray(bid_px, dtype=float)
    ask_px = np.asarray(ask_px, dtype=float)
    bid_sz = np.asarray(bid_sz, dtype=float)
    ask_sz = np.asarray(ask_sz, dtype=float)

    # 1. No crossed or locked book.
    crossed = bid_px[:, 0] >= ask_px[:, 0]
    if np.any(crossed):
        rep.add(
            f"crossed/locked book in {int(crossed.sum())} snapshots "
            f"(first at index {int(np.argmax(crossed))})"
        )

    # 2. Price monotonicity away from the touch.
    live = np.ones_like(bid_px, dtype=bool)
    if allow_empty_levels:
        live = (bid_sz > 0) & (ask_sz > 0)

    def _mono(px: np.ndarray, decreasing: bool, label: str) -> None:
        d = np.diff(px, axis=1)
        mask = live[:, 1:] & live[:, :-1]
        bad = (d >= 0) if decreasing else (d <= 0)
        bad = bad & mask
        if np.any(bad):
            rep.add(
                f"{label} prices not strictly monotone in {int(bad.any(axis=1).sum())} snapshots"
            )

    _mono(bid_px, decreasing=True, label="bid")
    _mono(ask_px, decreasing=False, label="ask")

    # 3. Sizes non-negative; best level must be populated.
    if np.any(bid_sz < 0) or np.any(ask_sz < 0):
        rep.add("negative sizes present")
    if np.any(bid_sz[:, 0] <= 0) or np.any(ask_sz[:, 0] <= 0):
        rep.add("empty best bid or best ask")

    # 4. Spread sanity: strictly positive and not absurd.
    spread = ask_px[:, 0] - bid_px[:, 0]
    mid = 0.5 * (ask_px[:, 0] + bid_px[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(mid > 0, spread / mid, np.inf)
    if np.any(rel > 0.10):
        rep.add(f"relative spread > 10% in {int((rel > 0.10).sum())} snapshots")

    return rep


def check_timestamps(ts: Sequence[float], allow_equal: bool = True) -> Report:
    """Event timestamps must be non-decreasing. Out-of-order events silently
    corrupt every causal feature you compute downstream.
    """
    rep = Report()
    t = np.asarray(ts, dtype=float)
    if not np.all(np.isfinite(t)):
        rep.add("non-finite timestamps")
        return rep
    d = np.diff(t)
    bad = (d < 0) if allow_equal else (d <= 0)
    if np.any(bad):
        rep.add(
            f"{int(bad.sum())} out-of-order timestamps (first at index {int(np.argmax(bad)) + 1})"
        )
    return rep


# --------------------------------------------------------------------------- #
# Split hygiene
# --------------------------------------------------------------------------- #


def check_purged_split(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    window: int,
    horizon: int,
    embargo: int = 0,
) -> Report:
    """Verify that no training sample's *input window or label window* overlaps a
    test sample's.

    A sample at index ``i`` reads inputs over ``[i - window + 1, i]`` and its
    label reads the future over ``[i, i + horizon]``. Two samples therefore
    share information whenever their indices are within ``window + horizon``.
    Anything closer than that across the split boundary is leakage, and it is
    the single most common reason a LOB model's reported metrics collapse when
    someone else reruns them.
    """
    rep = Report()
    tr = np.sort(np.asarray(train_idx))
    te = np.sort(np.asarray(test_idx))
    if tr.size == 0 or te.size == 0:
        rep.add("empty split")
        return rep

    overlap = np.intersect1d(tr, te)
    if overlap.size:
        rep.add(f"{overlap.size} indices appear in both train and test")

    gap_needed = window + horizon + embargo
    # Any train index within gap_needed of any test index is contaminated.
    pos = np.searchsorted(te, tr)
    left = np.clip(pos - 1, 0, te.size - 1)
    right = np.clip(pos, 0, te.size - 1)
    dist = np.minimum(np.abs(tr - te[left]), np.abs(tr - te[right]))
    bad = dist < gap_needed
    if np.any(bad):
        rep.add(
            f"{int(bad.sum())} train samples within {gap_needed} steps of a test sample "
            f"(need purge={window + horizon}, embargo={embargo})"
        )
    return rep


def check_chronological(train_idx: np.ndarray, test_idx: np.ndarray) -> Report:
    """Test must come strictly after train. Never shuffle a financial split."""
    rep = Report()
    tr, te = np.asarray(train_idx), np.asarray(test_idx)
    if tr.size and te.size and tr.max() >= te.min():
        rep.add(f"train max index {int(tr.max())} >= test min index {int(te.min())}")
    return rep


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def class_balance(labels: Sequence[int], n_classes: int = 3) -> np.ndarray:
    y = np.asarray(labels).astype(int).ravel()
    counts = np.bincount(y - y.min() if y.min() < 0 else y, minlength=n_classes)[:n_classes]
    return counts / max(counts.sum(), 1)


def check_horizon_unbiased(balances_by_horizon: dict[int, np.ndarray], tol: float = 0.10) -> Report:
    """The horizon-bias check.

    With a fixed threshold, longer horizons produce more |return| > theta events,
    so the 'stationary' class shrinks as k grows and cross-horizon numbers stop
    being comparable. After scaling theta with k, the class proportions should be
    approximately stable. This test is what makes your multi-horizon table mean
    something.
    """
    rep = Report()
    ks = sorted(balances_by_horizon)
    if len(ks) < 2:
        return rep
    mat = np.stack([np.asarray(balances_by_horizon[k], dtype=float) for k in ks])
    spread = mat.max(axis=0) - mat.min(axis=0)
    for c, s in enumerate(spread):
        if s > tol:
            rep.add(f"class {c} proportion varies by {s:.3f} across horizons {ks} (tol={tol})")
    return rep


def check_normalization_fit_on_train_only(
    train_stats: dict[str, np.ndarray],
    recomputed_from_train: dict[str, np.ndarray],
    rtol: float = 1e-6,
) -> Report:
    """Assert that the scaler actually stored training statistics -- not statistics
    of the full dataset. Fit a scaler on the training slice, then recompute mean
    and std from that same slice by hand and compare. If someone later moves the
    ``.fit()`` call above the split, this test fails loudly.
    """
    rep = Report()
    for key, expected in recomputed_from_train.items():
        got = train_stats.get(key)
        if got is None:
            rep.add(f"missing statistic '{key}'")
            continue
        if not np.allclose(np.asarray(got), np.asarray(expected), rtol=rtol, atol=1e-9):
            rep.add(f"statistic '{key}' does not match train-only recomputation")
    return rep


# --------------------------------------------------------------------------- #
# Execution accounting
# --------------------------------------------------------------------------- #


def check_execution_accounting(
    fills_qty: Sequence[float],
    fills_price: Sequence[float],
    target_qty: float,
    arrival_mid: float,
    reported_shortfall_bps: float,
    tol_bps: float = 1e-6,
) -> Report:
    """Conservation check for the execution backtest.

    Two failures this catches, both of which silently make an agent look
    brilliant: (1) the fills do not sum to the parent order, so the agent is
    quietly under-executing and dodging cost; (2) the reported shortfall does not
    equal the shortfall implied by the actual fills, meaning the reward and the
    metric have drifted apart.
    """
    rep = Report()
    q = np.asarray(fills_qty, dtype=float)
    p = np.asarray(fills_price, dtype=float)
    if q.shape != p.shape:
        rep.add("fill qty/price shape mismatch")
        return rep

    total = float(q.sum())
    if not np.isclose(total, target_qty, rtol=1e-9, atol=1e-9):
        rep.add(f"filled {total} vs target {target_qty} -- parent order not completed")

    if total == 0 or arrival_mid <= 0:
        rep.add("degenerate execution (zero quantity or non-positive arrival mid)")
        return rep

    vwap = float(np.dot(q, p) / total)
    implied = (vwap - arrival_mid) / arrival_mid * 1e4
    if not np.isclose(implied, reported_shortfall_bps, atol=tol_bps):
        rep.add(f"reported shortfall {reported_shortfall_bps:.6f} bps != implied {implied:.6f} bps")
    return rep


def check_no_lookahead_fills(
    decision_ts: Sequence[float],
    fill_ts: Sequence[float],
    min_latency_s: float = 0.0,
) -> Report:
    """Every fill must occur strictly after the decision that caused it, by at
    least the modelled latency. This is the test that catches the classic
    replay-backtest bug where the simulator matches an order against the very
    tick that triggered it.
    """
    rep = Report()
    d = np.asarray(decision_ts, dtype=float)
    f = np.asarray(fill_ts, dtype=float)
    if d.shape != f.shape:
        rep.add("timestamp shape mismatch")
        return rep
    bad = f < (d + min_latency_s)
    if np.any(bad):
        rep.add(f"{int(bad.sum())} fills occur before decision + latency ({min_latency_s}s)")
    return rep
