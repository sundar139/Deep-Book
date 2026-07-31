"""Tests for src/eval/stats.py.

The critical class of test here is *calibration under the null*: when there is
genuinely no difference, does the test reject at roughly the nominal rate? A
statistical tool that over-rejects will manufacture findings for you, and you
will not notice, because the finding will be exactly what you hoped for.
"""

from __future__ import annotations

import numpy as np
import pytest

from deepbook.eval.stats import (
    benjamini_hochberg,
    compare_paired,
    cscv_pbo,
    diebold_mariano,
    mcnemar_test,
    paired_bootstrap_ci,
    paired_cohens_d,
    probability_of_superiority,
    seed_summary,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_contains_true_mean(rng):
    d = rng.normal(0.5, 1.0, size=2000)
    ci = paired_bootstrap_ci(d, n_boot=2000, seed=1)
    assert ci.lo < 0.5 < ci.hi
    assert ci.p_value < 0.01


def test_bootstrap_ci_covers_zero_under_null(rng):
    d = rng.normal(0.0, 1.0, size=2000)
    ci = paired_bootstrap_ci(d, n_boot=2000, seed=1)
    assert ci.lo < 0 < ci.hi
    assert ci.p_value > 0.05


@pytest.mark.slow
def test_bootstrap_coverage_is_approximately_nominal():
    """Across 200 replications, a 95% CI should contain the truth ~95% of the
    time. This is the test that proves the interval means what you claim.
    """
    hits = 0
    reps = 200
    for r in range(reps):
        d = np.random.default_rng(r).normal(0.3, 1.0, size=400)
        ci = paired_bootstrap_ci(d, n_boot=800, seed=r)
        hits += int(ci.lo <= 0.3 <= ci.hi)
    coverage = hits / reps
    assert 0.90 <= coverage <= 0.99, f"coverage {coverage:.3f} is not ~0.95"


def test_block_bootstrap_widens_ci_for_autocorrelated_data(rng):
    """Serially correlated differences carry less independent information, so a
    correct interval must be *wider*. If your CI ignores autocorrelation you
    will report significance that isn't there -- and overlapping LOB windows are
    always autocorrelated.
    """
    e = rng.normal(0, 1, size=4000)
    ar = np.zeros_like(e)
    for i in range(1, e.size):
        ar[i] = 0.95 * ar[i - 1] + e[i]

    iid_ci = paired_bootstrap_ci(ar, n_boot=1500, block_size=None, seed=3)
    blk_ci = paired_bootstrap_ci(ar, n_boot=1500, block_size=100, seed=3)
    assert (blk_ci.hi - blk_ci.lo) > 1.5 * (iid_ci.hi - iid_ci.lo)


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #


def test_dm_detects_a_genuinely_better_model(rng):
    loss_b = rng.gamma(2.0, 1.0, size=1500)
    loss_a = loss_b - 0.25 + rng.normal(0, 0.05, size=1500)  # A strictly better
    stat, p = diebold_mariano(loss_a, loss_b, horizon=1)
    assert stat < 0, "negative statistic should indicate A has lower loss"
    assert p < 0.01


def test_dm_does_not_reject_identical_models(rng):
    loss = rng.gamma(2.0, 1.0, size=1500)
    stat, p = diebold_mariano(loss, loss.copy(), horizon=1)
    assert np.isnan(stat) or p > 0.5


@pytest.mark.slow
def test_dm_null_rejection_rate_is_near_nominal():
    """Type-I error check with overlapping (autocorrelated) losses. Without the
    HAC correction this test fails badly -- which is the point.
    """
    rejections = 0
    reps = 300
    horizon = 20
    for r in range(reps):
        g = np.random.default_rng(r)
        base = g.normal(0, 1, size=1200)
        # Overlapping windows induce MA(horizon-1) structure in both losses.
        kern = np.ones(horizon) / horizon
        la = np.convolve(base, kern, mode="valid") ** 2
        lb = np.convolve(g.normal(0, 1, size=1200), kern, mode="valid") ** 2
        _, p = diebold_mariano(la, lb, horizon=horizon)
        rejections += int(np.isfinite(p) and p < 0.05)
    rate = rejections / reps
    assert rate < 0.15, f"null rejection rate {rate:.3f} is far above nominal 0.05"


# --------------------------------------------------------------------------- #
# McNemar
# --------------------------------------------------------------------------- #


def test_mcnemar_on_constructed_table():
    a = np.array([True] * 60 + [False] * 40)
    b = np.array([True] * 40 + [False] * 20 + [True] * 35 + [False] * 5)
    res = mcnemar_test(a, b)
    assert res["n01"] == 35 and res["n10"] == 20
    assert 0.0 <= res["p_value"] <= 1.0


def test_mcnemar_identical_models_is_degenerate():
    a = np.array([True, False, True, True])
    res = mcnemar_test(a, a.copy())
    assert res["p_value"] == 1.0


def test_mcnemar_uses_exact_test_for_small_counts():
    a = np.array([True] * 10 + [False] * 5)
    b = np.array([True] * 8 + [False] * 2 + [True] * 4 + [False])
    res = mcnemar_test(a, b)
    assert res["method"] in {"exact", "degenerate"}


# --------------------------------------------------------------------------- #
# Multiple comparisons
# --------------------------------------------------------------------------- #


def test_benjamini_hochberg_is_monotone_and_conservative():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.60, 0.99])
    out = benjamini_hochberg(p, alpha=0.05)
    q = out["q_values"]
    assert np.all(q >= p - 1e-12), "q-values must never be below raw p-values"
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12), "q-values must be monotone in p"
    assert out["n_rejected"] < np.sum(p <= 0.05)


def test_benjamini_hochberg_controls_fdr_under_global_null(rng):
    p = rng.uniform(0, 1, size=500)
    out = benjamini_hochberg(p, alpha=0.05)
    assert out["n_rejected"] <= 5, "should reject almost nothing under the global null"


# --------------------------------------------------------------------------- #
# Effect size and seed aggregation
# --------------------------------------------------------------------------- #


def test_effect_size_and_win_rate(rng):
    d = rng.normal(1.0, 1.0, size=5000)
    assert 0.8 < paired_cohens_d(d) < 1.2
    assert 0.75 < probability_of_superiority(d) < 0.90


def test_seed_summary_reports_spread():
    vals = [0.512, 0.499, 0.521, 0.505, 0.488]
    s = seed_summary(vals)
    assert s.n == 5
    assert s.lo < s.point < s.hi


# --------------------------------------------------------------------------- #
# PBO / CSCV -- the backtest-overfitting guard
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_pbo_is_near_half_for_pure_noise():
    """If every configuration is equally worthless, picking the in-sample winner
    should have no out-of-sample value, i.e. PBO ~ 0.5.

    Note the averaging over 25 independent performance matrices: a PBO computed
    from a *single* dataset has a standard deviation around 0.20, because all
    the combinations share the same underlying data. That is a property worth
    internalising before you report a single PBO number in the paper.
    """
    pbos = [
        cscv_pbo(np.random.default_rng(s).normal(0, 1, size=(1200, 30)), n_splits=10, seed=0)["pbo"]
        for s in range(25)
    ]
    assert 0.40 <= float(np.mean(pbos)) <= 0.60, float(np.mean(pbos))


@pytest.mark.slow
def test_pbo_is_low_when_one_config_is_genuinely_better():
    g = np.random.default_rng(11)
    perf = g.normal(0, 1, size=(1600, 30))
    perf[:, 0] += 0.3  # a real edge, present in every sub-period
    out = cscv_pbo(perf, n_splits=10, seed=0)
    assert out["pbo"] < 0.20, out["pbo"]
    assert out["oos_perf"].mean() > 0


@pytest.mark.slow
def test_degradation_slope_is_negative_by_construction():
    """Pins the documented caveat: the slope is near -1 even when the edge is
    real, because the two halves partition a fixed dataset. Anyone tempted to
    report it as an overfitting diagnostic will trip this test.
    """
    g = np.random.default_rng(5)
    perf = g.normal(0, 1, size=(1200, 20))
    perf[:, 0] += 0.4
    out = cscv_pbo(perf, n_splits=8, seed=0)
    assert out["degradation_slope"] < 0


def test_pbo_rejects_odd_splits():
    with pytest.raises(ValueError):
        cscv_pbo(np.random.default_rng(0).normal(size=(400, 5)), n_splits=7)


# --------------------------------------------------------------------------- #
# The one-call comparison used in the results tables
# --------------------------------------------------------------------------- #


def test_compare_paired_reports_the_right_winner(rng):
    twap = rng.normal(8.0, 2.0, size=400)  # 8 bps shortfall
    agent = twap - rng.normal(1.0, 0.5, size=400)  # ~1 bp better, paired
    res = compare_paired(agent, twap, "PPO", "TWAP", lower_is_better=True, seed=2)
    assert res["favours"] == "PPO"
    assert res["mean_diff"] < 0
    assert res["ci_hi"] < 0, "CI should exclude zero for a real 1 bp improvement"
    assert res["win_rate_a"] > 0.9
