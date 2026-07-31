"""Univariate exponential-kernel Hawkes process: simulation, O(N) log-likelihood,
MLE fit, and a rolling intensity feature.

This exists as a *reference implementation with a recovery test* rather than a
dependency on ``tick``, which has no reliable Windows wheels. Extend to the
multivariate case (6 event types: limit/cancel/market x bid/ask) by making
mu a vector and alpha, beta matrices; the recursion generalises directly.

Intensity:  lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
Stationary iff the branching ratio n = alpha / beta < 1.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


def simulate_hawkes(mu: float, alpha: float, beta: float, T: float, seed: int = 0) -> np.ndarray:
    """Ogata's thinning algorithm. Returns sorted event times on [0, T).

    You need a simulator you trust before you can trust a fitter: the recovery
    test (simulate at known parameters -> refit -> compare) is the only way to
    know your likelihood is right. A large fraction of published Hawkes code is
    silently wrong because nobody wrote this test.
    """
    if beta <= 0 or alpha < 0 or mu <= 0:
        raise ValueError("require mu > 0, alpha >= 0, beta > 0")
    if alpha >= beta:
        raise ValueError("non-stationary: alpha/beta must be < 1")

    rng = np.random.default_rng(seed)
    events: list[float] = []
    t = 0.0
    while t < T:
        # Upper bound on the intensity from t onwards (it only decays until the
        # next event), so thinning with lam_bar is valid.
        lam_bar = mu + sum(alpha * math.exp(-beta * (t - ti)) for ti in events[-500:])
        lam_bar = max(lam_bar, mu)
        t += rng.exponential(1.0 / lam_bar)
        if t >= T:
            break
        lam_t = mu + sum(alpha * math.exp(-beta * (t - ti)) for ti in events[-500:])
        if rng.random() <= lam_t / lam_bar:
            events.append(t)
    return np.asarray(events, dtype=float)


def hawkes_nll(params: np.ndarray, t: np.ndarray, T: float) -> float:
    """Negative log-likelihood in O(N) using the exponential-kernel recursion:

        R_i = exp(-beta * (t_i - t_{i-1})) * (1 + R_{i-1}),   R_0 = 0
        ll  = sum_i log(mu + alpha * R_i) - mu*T - (alpha/beta) * sum_i (1 - exp(-beta*(T - t_i)))

    Never write the naive double sum: it is O(N^2) and a day of L2 crypto data
    has millions of events.
    """
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return 1e12

    n = t.size
    if n == 0:
        return mu * T

    R = 0.0
    ll = 0.0
    for i in range(n):
        if i > 0:
            R = math.exp(-beta * (t[i] - t[i - 1])) * (1.0 + R)
        intensity = mu + alpha * R
        if intensity <= 0:
            return 1e12
        ll += math.log(intensity)

    ll -= mu * T
    ll -= (alpha / beta) * float(np.sum(1.0 - np.exp(-beta * (T - t))))
    return -ll


def fit_hawkes_mle(t: np.ndarray, T: float, x0: tuple[float, float, float] | None = None) -> dict:
    """MLE fit. Returns mu, alpha, beta, branching ratio and convergence info.

    Always inspect ``branching_ratio``: values at or above 1 mean the fit has run
    to the non-stationary boundary, which usually indicates event-type
    contamination (e.g. cancels mixed with limit orders) rather than a genuinely
    explosive market.
    """
    t = np.asarray(t, dtype=float)
    if x0 is None:
        rate = max(t.size / max(T, 1e-9), 1e-6)
        x0 = (rate * 0.5, 0.5, 1.0)

    res = minimize(
        hawkes_nll,
        x0=np.asarray(x0, dtype=float),
        args=(t, T),
        method="L-BFGS-B",
        bounds=[(1e-8, None), (0.0, None), (1e-6, None)],
    )
    mu, alpha, beta = res.x
    return {
        "mu": float(mu),
        "alpha": float(alpha),
        "beta": float(beta),
        "branching_ratio": float(alpha / beta) if beta > 0 else float("inf"),
        "nll": float(res.fun),
        "success": bool(res.success),
        "n_events": int(t.size),
    }


def intensity_at(
    t_eval: np.ndarray, t_events: np.ndarray, mu: float, alpha: float, beta: float
) -> np.ndarray:
    """Evaluate lambda(t) at arbitrary times, causally (only events strictly before
    each evaluation point contribute). This is the feature you append to the
    model input -- and the causality here is exactly what the leakage test
    protects.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    t_events = np.asarray(t_events, dtype=float)
    out = np.empty_like(t_eval)
    # Single pass with the same recursion, advancing an event pointer.
    R, last_t, j = 0.0, None, 0
    for i, te in enumerate(t_eval):
        while j < t_events.size and t_events[j] < te:
            tj = t_events[j]
            R = (math.exp(-beta * (tj - last_t)) * R + 1.0) if last_t is not None else 1.0
            last_t = tj
            j += 1
        decay = math.exp(-beta * (te - last_t)) if last_t is not None else 0.0
        out[i] = mu + alpha * R * decay
    return out
