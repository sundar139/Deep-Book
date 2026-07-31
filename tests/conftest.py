"""Shared fixtures and test-tier configuration.

Test tiers (see pytest.ini):
  fast     -- pure functions, invariants, contracts. < 30 s total. Runs on every commit.
  slow     -- statistical validation on larger synthetic samples. Runs on every push.
  gpu      -- requires CUDA. Runs locally / nightly only.
  data     -- requires real downloaded data on disk. Skipped in CI by design.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20260730


@pytest.fixture(autouse=True)
def _deterministic():
    """Every test starts from the same RNG state. Non-determinism in a test
    suite is indistinguishable from a real bug, and you will waste days.
    """
    np.random.seed(SEED)
    yield


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def synthetic_book(rng):
    """A well-formed 10-level book: 2000 snapshots, tick size 0.01, mid random walk.

    Returns (bid_px, bid_sz, ask_px, ask_sz, ts). Use this to test features and
    invariants without touching real data -- fast, hermetic, and it lets you
    inject specific corruptions to prove your checks actually fire.
    """
    n, levels, tick = 2000, 10, 0.01
    mid = 100.0 + np.cumsum(rng.normal(0, 0.01, size=n))
    half_spread = tick * rng.integers(1, 3, size=n)

    best_bid = np.round((mid - half_spread) / tick) * tick
    best_ask = best_bid + tick * rng.integers(1, 4, size=n)

    offsets = np.arange(levels) * tick
    bid_px = best_bid[:, None] - offsets[None, :]
    ask_px = best_ask[:, None] + offsets[None, :]
    bid_sz = rng.integers(1, 500, size=(n, levels)).astype(float)
    ask_sz = rng.integers(1, 500, size=(n, levels)).astype(float)
    ts = np.sort(rng.uniform(0, 3600, size=n))
    return bid_px, bid_sz, ask_px, ask_sz, ts


@pytest.fixture
def data_dir() -> Path:
    return Path(os.environ.get("DEEPBOOK_DATA", ROOT / "data"))


def pytest_collection_modifyitems(config, items):
    """Auto-skip data-dependent tests when the data isn't present, instead of
    failing CI for a reason that has nothing to do with the commit.
    """
    data_root = Path(os.environ.get("DEEPBOOK_DATA", ROOT / "data"))
    for item in items:
        if "data" in item.keywords and not data_root.exists():
            item.add_marker(pytest.mark.skip(reason=f"no data at {data_root}"))
