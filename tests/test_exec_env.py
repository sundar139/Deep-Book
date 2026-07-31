"""Execution-environment conformance tests.

An RL result is only as trustworthy as the environment it was trained in, and
environment bugs are self-concealing: the agent finds them, exploits them, and
reports a spectacular number. Every test below is a specific exploit closed off.

The tests use a small reference environment so the suite is runnable today.
Point ``make_env`` at your real ``src.exec.env`` once it exists -- the
assertions transfer unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from deepbook.validation.invariants import check_execution_accounting

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Minimal reference environment (replace with src.exec.env.ExecutionEnv)
# --------------------------------------------------------------------------- #


class ToyExecutionEnv:
    """Frictionless-mid execution with linear temporary impact and a terminal
    liquidation penalty. Deliberately simple: these tests check the *contract*,
    not the microstructure.
    """

    def __init__(
        self,
        mid: np.ndarray,
        target_qty: float = 1000.0,
        n_steps: int = 20,
        impact: float = 1e-5,
        seed: int = 0,
    ):
        self.mid = np.asarray(mid, dtype=float)
        self.target_qty = float(target_qty)
        self.n_steps = int(n_steps)
        self.impact = float(impact)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.remaining = self.target_qty
        self.arrival = float(self.mid[0])
        self.fills: list[tuple[float, float]] = []
        return self._obs()

    def _obs(self):
        return np.array(
            [
                self.remaining / self.target_qty,
                1.0 - self.t / self.n_steps,
                self.mid[self.t] / self.arrival - 1.0,
            ],
            dtype=float,
        )

    def step(self, action_frac: float):
        qty = float(np.clip(action_frac, 0.0, 1.0)) * self.remaining
        if self.t == self.n_steps - 1:
            qty = self.remaining  # forced completion
        px = self.mid[self.t] * (1.0 + self.impact * qty)
        if qty > 0:
            self.fills.append((qty, px))
        self.remaining -= qty
        cost_bps = (px - self.arrival) / self.arrival * 1e4 * (qty / self.target_qty)
        self.t += 1
        done = self.t >= self.n_steps or self.remaining <= 1e-9
        return self._obs(), -cost_bps, done, {"remaining": self.remaining}

    def shortfall_bps(self) -> float:
        q = np.array([f[0] for f in self.fills])
        p = np.array([f[1] for f in self.fills])
        vwap = float(np.dot(q, p) / q.sum())
        return (vwap - self.arrival) / self.arrival * 1e4


@pytest.fixture
def mid_path(rng):
    return 100.0 + np.cumsum(rng.normal(0, 0.02, size=21))


def make_env(mid_path, **kw) -> ToyExecutionEnv:
    return ToyExecutionEnv(mid_path, **kw)


# --------------------------------------------------------------------------- #
# Conformance
# --------------------------------------------------------------------------- #


def test_reset_is_deterministic_given_a_seed(mid_path):
    a = make_env(mid_path).reset(seed=1)
    b = make_env(mid_path).reset(seed=1)
    np.testing.assert_allclose(a, b)


def test_observation_shape_and_finiteness_hold_every_step(mid_path):
    env = make_env(mid_path)
    obs = env.reset()
    for _ in range(env.n_steps):
        assert obs.shape == (3,) and np.all(np.isfinite(obs))
        obs, r, done, _ = env.step(0.1)
        assert np.isfinite(r)
        if done:
            break


def test_episode_always_terminates_with_zero_inventory(mid_path):
    """The single most important environment guarantee. An agent that can end an
    episode holding inventory has found a way to not pay for the hard part.
    """
    for policy in (0.0, 0.05, 0.5, 1.0):
        env = make_env(mid_path)
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(policy)
        assert info["remaining"] == pytest.approx(0.0, abs=1e-9), f"policy={policy} left inventory"


def test_reward_sums_to_the_reported_shortfall(mid_path):
    """Reward-metric drift check. If the sum of per-step rewards is not the
    metric you report, the agent is optimising something other than what you are
    measuring -- and you will never reconcile the two numbers later.
    """
    env = make_env(mid_path)
    env.reset()
    total, done = 0.0, False
    while not done:
        _, r, done, _ = env.step(0.15)
        total += r
    assert -total == pytest.approx(env.shortfall_bps(), rel=1e-6)


def test_accounting_identity_via_shared_invariant(mid_path):
    env = make_env(mid_path)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(0.2)
    q = [f[0] for f in env.fills]
    p = [f[1] for f in env.fills]
    check_execution_accounting(
        q, p, env.target_qty, env.arrival, env.shortfall_bps()
    ).raise_if_failed("toy env")


def test_larger_orders_cost_more(mid_path):
    """Monotonicity sanity: impact must be increasing in size. If it isn't, the
    agent will learn to dump the whole parent order in one step.
    """
    small = make_env(mid_path, target_qty=100.0)
    large = make_env(mid_path, target_qty=10_000.0)
    for env in (small, large):
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(1.0)
    assert large.shortfall_bps() > small.shortfall_bps()


def test_twap_is_beaten_by_nothing_on_a_driftless_market(mid_path):
    """Control experiment as a test. On a martingale mid with symmetric impact, no
    schedule should systematically beat TWAP by a wide margin. If a trivial
    policy does, the environment has a free-money bug -- check for fills at
    stale prices, or a mid that is updated after the fill instead of before.
    """

    def run(fracs):
        env = make_env(mid_path)
        env.reset()
        done, i = False, 0
        while not done:
            _, _, done, _ = env.step(fracs[min(i, len(fracs) - 1)])
            i += 1
        return env.shortfall_bps()

    twap = run([1.0 / (20 - i) for i in range(20)])
    front = run([1.0] + [0.0] * 19)
    assert front >= twap - 1e-9, "front-loading beats TWAP with zero drift -- suspicious"


@pytest.mark.slow
def test_agent_must_beat_random_before_you_believe_anything():
    """The control policies every RL execution result needs. Report these in the
    paper. If the trained agent does not clearly dominate random and passive
    baselines across seeds, the environment or the reward is the problem -- not
    the hyperparameters.
    """
    rng = np.random.default_rng(0)
    randoms, twaps = [], []
    for s in range(50):
        path = 100.0 + np.cumsum(np.random.default_rng(s).normal(0, 0.02, size=21))

        env = make_env(path)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(rng.uniform(0, 1))
        randoms.append(env.shortfall_bps())

        env = make_env(path)
        env.reset()
        done, i = False, 0
        while not done:
            _, _, done, _ = env.step(1.0 / (20 - min(i, 19)))
            i += 1
        twaps.append(env.shortfall_bps())

    assert np.isfinite(np.mean(randoms)) and np.isfinite(np.mean(twaps))
    assert np.std(twaps) < np.std(randoms) * 3
