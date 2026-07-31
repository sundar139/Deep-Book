"""Statistical validation toolkit for DeepBook.

Everything here answers one question: *is the difference I measured larger than
the noise in my measurement process?*  Nothing in this module knows about LOBs,
transformers, or RL -- it operates on arrays of per-sample losses, per-episode
costs, or per-configuration performance, which keeps it unit-testable.

Conventions
-----------
* "loss" arrays are lower-is-better.
* "perf" arrays are higher-is-better.
* Every function that uses randomness takes an explicit ``rng`` or ``seed``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every RNG that can affect a run. Call once, at the top of main()."""
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Slower, but makes GPU reductions reproducible run-to-run.
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    p_value: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}] (p={self.p_value:.4f}, n={self.n})"


def _moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one moving-block bootstrap resample of length ``n``."""
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()
    return idx[:n]


def paired_bootstrap_ci(
    diff: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    block_size: int | None = None,
    seed: int = 0,
) -> Interval:
    """Bootstrap CI for the mean of a *paired* difference (e.g. IS_agent - IS_twap,
    episode by episode, or loss_A - loss_B sample by sample).

    Parameters
    ----------
    block_size
        ``None`` for the i.i.d. bootstrap. Set it to a positive integer to use a
        moving-block bootstrap -- required whenever the paired differences are
        serially correlated, which they *always* are for per-timestep forecast
        losses on overlapping windows. A reasonable default is
        ``block_size = window_len + horizon``.

    The two-sided p-value is the standard bootstrap inversion:
    ``2 * min(P(boot <= 0), P(boot >= 0))``.

    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return Interval(float("nan"), float("nan"), float("nan"), float("nan"), n)

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    if block_size is None or block_size <= 1:
        for b in range(n_boot):
            means[b] = d[rng.integers(0, n, size=n)].mean()
    else:
        block = min(int(block_size), n)
        for b in range(n_boot):
            means[b] = d[_moving_block_indices(n, block, rng)].mean()

    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return Interval(float(d.mean()), float(lo), float(hi), float(min(p, 1.0)), n)


# --------------------------------------------------------------------------- #
# Diebold-Mariano (forecast accuracy comparison)
# --------------------------------------------------------------------------- #


def diebold_mariano(
    loss_a: Sequence[float],
    loss_b: Sequence[float],
    horizon: int = 1,
    hac_lags: int | None = None,
) -> tuple[float, float]:
    """Harvey-Leybourne-Newbold small-sample-corrected Diebold-Mariano test.

    H0: the two forecasts have equal expected loss.
    Negative statistic => model A has lower loss (A is better).

    Uses a Bartlett-kernel HAC variance with ``horizon - 1`` lags by default,
    which is the right correction when your k-step-ahead labels create
    overlapping forecast errors. If you window your inputs with length T and
    label at horizon k, pass ``horizon = T + k`` -- the overlap, not the label
    horizon alone, drives the autocorrelation.

    Returns
    -------
    (statistic, p_value) with the p-value from a t-distribution on n-1 df.

    """
    la = np.asarray(loss_a, dtype=float)
    lb = np.asarray(loss_b, dtype=float)
    if la.shape != lb.shape:
        raise ValueError("loss arrays must have the same shape")

    d = la - lb
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return float("nan"), float("nan")

    dbar = d.mean()
    dc = d - dbar
    lags = (horizon - 1) if hac_lags is None else hac_lags
    lags = max(0, min(int(lags), n - 1))

    gamma0 = float(np.dot(dc, dc) / n)
    var = gamma0
    for k in range(1, lags + 1):
        gk = float(np.dot(dc[k:], dc[:-k]) / n)
        w = 1.0 - k / (lags + 1.0)  # Bartlett weight -> guarantees var >= 0
        var += 2.0 * w * gk

    if var <= 0 or not np.isfinite(var):
        return float("nan"), float("nan")

    dm = dbar / math.sqrt(var / n)

    h = max(1, int(horizon))
    corr = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
    if corr <= 0:
        return float("nan"), float("nan")
    dm_star = dm * math.sqrt(corr)

    p = 2.0 * stats.t.sf(abs(dm_star), df=n - 1)
    return float(dm_star), float(p)


# --------------------------------------------------------------------------- #
# McNemar (paired classifier comparison)
# --------------------------------------------------------------------------- #


def mcnemar_test(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    exact_threshold: int = 25,
) -> dict:
    """Paired test for two classifiers evaluated on the *same* samples.

    Use this rather than comparing two accuracy numbers: two models can have
    identical accuracy and completely different error sets, and comparing point
    accuracies throws away the pairing that gives you the power.

    Returns keys: n01, n10, statistic, p_value, method.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("prediction-correctness arrays must have the same shape")

    n01 = int(np.sum(~a & b))  # A wrong, B right
    n10 = int(np.sum(a & ~b))  # A right, B wrong
    total = n01 + n10

    if total == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0, "method": "degenerate"}

    if total < exact_threshold:
        p = float(stats.binomtest(n10, total, 0.5).pvalue)
        return {"n01": n01, "n10": n10, "statistic": float(n10), "p_value": p, "method": "exact"}

    stat = (abs(n01 - n10) - 1.0) ** 2 / total  # Edwards continuity correction
    p = float(stats.chi2.sf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": p, "method": "chi2"}


# --------------------------------------------------------------------------- #
# Multiple comparisons
# --------------------------------------------------------------------------- #


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR control.

    You will run one test per (model x horizon x dataset x ablation). That is
    dozens of tests. Reporting the raw p-values is how a null result gets
    mistaken for a discovery. Correct them, and say in the paper that you did.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]

    q_sorted = ranked * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return {"q_values": q, "rejected": q <= alpha, "n_rejected": int(np.sum(q <= alpha))}


# --------------------------------------------------------------------------- #
# Effect size
# --------------------------------------------------------------------------- #


def paired_cohens_d(diff: Sequence[float]) -> float:
    """Standardised effect size for a paired difference. Report it next to p."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def probability_of_superiority(diff: Sequence[float]) -> float:
    """P(difference > 0), the fraction of paired comparisons that the first method
    wins. Robust, dimensionless, and far easier to communicate than Cohen's d.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    return float(np.mean(d > 0)) if d.size else float("nan")


def seed_summary(values: Sequence[float], alpha: float = 0.05) -> Interval:
    """Across-seed summary with a t-based CI. Use for the headline table: every
    reported metric is 'mean over 5 seeds [CI]', never a single run.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return Interval(
            float(v.mean()) if n else float("nan"), float("nan"), float("nan"), float("nan"), n
        )
    mean = float(v.mean())
    sem = float(stats.sem(v))
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    p = float(stats.ttest_1samp(v, 0.0).pvalue)
    return Interval(mean, mean - tcrit * sem, mean + tcrit * sem, p, n)


# --------------------------------------------------------------------------- #
# Backtest overfitting: CSCV / PBO
# --------------------------------------------------------------------------- #


def _sharpe(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, x.mean(axis=0) / sd, 0.0)


def cscv_pbo(
    perf: np.ndarray,
    n_splits: int = 8,
    metric: Callable[[np.ndarray], np.ndarray] = _sharpe,
    max_combinations: int | None = 2000,
    seed: int = 0,
) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation (Bailey, Borwein, Lopez de Prado & Zhu).

    Parameters
    ----------
    perf
        Array of shape (T, N): per-period performance for each of N
        configurations you tried. For DeepBook this is the per-episode negative
        implementation shortfall of every executor variant, or the per-day
        metric of every hyperparameter configuration.
    n_splits
        Must be even. Rows are cut into ``n_splits`` contiguous blocks;
        every balanced half-split is used once as in-sample.

    Returns
    -------
    dict with ``pbo`` (probability that the in-sample-best configuration ranks
    below median out-of-sample), the lambda logits, and the OOS-vs-IS
    degradation slope.

    Report ``pbo`` and ``prob_oos_negative``. Do *not* read the
    ``degradation_slope`` as evidence of overfitting on its own: because the
    in-sample and out-of-sample halves are complementary partitions of a fixed
    dataset, their means are mechanically negatively related and the slope sits
    near -1 whether or not a real edge exists. It is included only for
    comparability with the published CSCV tables.

    Why you need this: you will try dozens of reward shapes, feature sets and
    hyperparameters against the same held-out period. PBO measures how much of
    your final 'winner' is selection noise. A PBO near 0.5 means your selection
    procedure has no out-of-sample predictive value at all -- and you should say
    so rather than publish the winner.

    """
    M = np.asarray(perf, dtype=float)
    if M.ndim != 2:
        raise ValueError("perf must be 2-D (T periods x N configurations)")
    T, N = M.shape
    if N < 2:
        raise ValueError("need at least 2 configurations")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    if n_splits * 2 > T:
        raise ValueError("not enough rows for the requested n_splits")

    blocks = np.array_split(np.arange(T), n_splits)
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    if max_combinations is not None and len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in pick]

    lambdas, is_perf, oos_perf = [], [], []
    for c in combos:
        is_rows = np.concatenate([blocks[i] for i in c])
        oos_rows = np.concatenate([blocks[i] for i in range(n_splits) if i not in c])

        r_is = metric(M[is_rows])
        r_oos = metric(M[oos_rows])

        best = int(np.argmax(r_is))
        rank = float(stats.rankdata(r_oos)[best])  # 1 = worst, N = best
        omega = rank / (N + 1.0)
        omega = min(max(omega, 1e-12), 1 - 1e-12)
        lambdas.append(math.log(omega / (1.0 - omega)))
        is_perf.append(float(r_is[best]))
        oos_perf.append(float(r_oos[best]))

    lam = np.asarray(lambdas)
    is_arr, oos_arr = np.asarray(is_perf), np.asarray(oos_perf)
    slope = float(np.polyfit(is_arr, oos_arr, 1)[0]) if np.ptp(is_arr) > 0 else float("nan")

    return {
        "pbo": float(np.mean(lam <= 0.0)),
        "lambdas": lam,
        "is_perf": is_arr,
        "oos_perf": oos_arr,
        "degradation_slope": slope,
        "prob_oos_negative": float(np.mean(oos_arr < 0.0)),
        "n_combinations": len(combos),
    }


# --------------------------------------------------------------------------- #
# Convenience: the comparison you will run over and over
# --------------------------------------------------------------------------- #


def compare_paired(
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    name_a: str = "A",
    name_b: str = "B",
    lower_is_better: bool = True,
    block_size: int | None = None,
    horizon: int = 1,
    seed: int = 0,
) -> dict:
    """One-call paired comparison producing everything a reviewer asks for:
    mean difference, bootstrap CI, DM p-value, effect size, win rate.

    Example:
    -------
    >>> res = compare_paired(is_agent_bps, is_twap_bps, "PPO", "TWAP")

    """
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    diff = a - b
    ci = paired_bootstrap_ci(diff, block_size=block_size, seed=seed)
    dm, dm_p = (
        diebold_mariano(a, b, horizon=horizon)
        if lower_is_better
        else diebold_mariano(-a, -b, horizon=horizon)
    )
    better = name_a if ((ci.point < 0) == lower_is_better) else name_b
    return {
        "name_a": name_a,
        "name_b": name_b,
        "mean_diff": ci.point,
        "ci_lo": ci.lo,
        "ci_hi": ci.hi,
        "bootstrap_p": ci.p_value,
        "dm_stat": dm,
        "dm_p": dm_p,
        "cohens_d": paired_cohens_d(diff),
        "win_rate_a": probability_of_superiority(-diff if lower_is_better else diff),
        "n": int(diff.size),
        "favours": better,
    }
