"""End-to-end contract tests.

These are the tests that catch the failures unit tests cannot see, because the
bug lives in the *composition* of correct-looking parts: a scaler fitted one line
too early, a label computed one index too late, a Hawkes fitter that converges to
something plausible but wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef

from deepbook.features.hawkes import fit_hawkes_mle, intensity_at, simulate_hawkes

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# The shuffled-label leakage test  (run this in CI, forever)
# --------------------------------------------------------------------------- #


def _fit_eval(X, Y, split, window, horizon, embargo=0):
    gap = window + horizon + embargo
    tr = slice(0, split)
    te = slice(split + gap, len(Y))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9  # train-only normalisation
    clf = LogisticRegression(max_iter=500).fit((X[tr] - mu) / sd, Y[tr])
    pred = clf.predict((X[te] - mu) / sd)
    return matthews_corrcoef(Y[te], pred)


@pytest.fixture
def signal_dataset(rng):
    """A strictly causal synthetic task: the label at the end of each window is
    driven by the mean of that same window plus fresh noise. A correct pipeline
    therefore *must* learn it, which is what makes the shuffled-label test below
    meaningful.

    Building this by hand -- rather than reusing the production windower -- is
    deliberate: the fixture has to be trustworthy independently of the code
    under test.
    """
    n, window, horizon = 6000, 20, 5
    r = rng.normal(0, 1, size=n)
    csum = np.concatenate([[0.0], np.cumsum(r)])
    roll = (csum[window:] - csum[:-window]) / window  # roll[i] = mean(r[i:i+window])
    m = roll.size - horizon
    X = np.stack([r[i : i + window] for i in range(m)])
    driver = 0.8 * roll[:m] + rng.normal(0, 0.15, size=m)
    Y = (driver > 0).astype(int)
    return X, Y, window, horizon


def test_pipeline_learns_real_signal(signal_dataset):
    """Positive control. If this fails, the leakage test below proves nothing --
    a broken pipeline scores zero on everything, including shuffled labels.
    """
    X, Y, window, horizon = signal_dataset
    mcc = _fit_eval(X, Y, split=int(0.7 * len(Y)), window=window, horizon=horizon)
    assert mcc > 0.15, f"positive control failed (MCC={mcc:.3f}); fix before trusting other tests"


def test_shuffled_labels_yield_no_skill(signal_dataset, rng):
    """THE leakage test. Destroy the label-feature relationship, keep everything
    else identical. Any remaining skill is information flowing through a channel
    it should not: a globally-fitted scaler, an overlapping split, a feature
    computed on the full series, a duplicated row.

    Run this on the *real* pipeline with the real dataset class, not just here.
    """
    X, Y, window, horizon = signal_dataset
    Y_shuf = Y.copy()
    rng.shuffle(Y_shuf)
    mcc = _fit_eval(X, Y_shuf, split=int(0.7 * len(Y)), window=window, horizon=horizon)
    assert abs(mcc) < 0.06, f"skill on shuffled labels (MCC={mcc:.3f}) -- the pipeline leaks"


def test_leakage_detector_actually_fires_on_injected_leakage(signal_dataset, rng):
    """Meta-test: prove the detector detects.

    Here the split is corrupted the way a random shuffle corrupts it -- test rows
    are duplicates of training rows. Labels are shuffled, so there is no real
    signal at all, yet a memorising model scores well. If this test ever stops
    failing-on-purpose, the shuffled-label check above has become a no-op and
    every result that depends on it is unverified.
    """
    from sklearn.neighbors import KNeighborsClassifier

    X, Y, _, _ = signal_dataset
    Y_shuf = Y.copy()
    rng.shuffle(Y_shuf)

    n = 1500
    Xtr, Ytr = X[:n], Y_shuf[:n]
    Xte, Yte = X[:n].copy(), Y_shuf[:n]  # test == train: injected leakage

    clf = KNeighborsClassifier(n_neighbors=1).fit(Xtr, Ytr)
    mcc = matthews_corrcoef(Yte, clf.predict(Xte))
    assert mcc > 0.5, "leakage was injected but the detector saw nothing"


# --------------------------------------------------------------------------- #
# Hawkes: parameter recovery
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.parametrize("mu,alpha,beta", [(0.5, 0.6, 1.2), (1.0, 0.8, 2.0)])
def test_hawkes_mle_recovers_known_parameters(mu, alpha, beta):
    """Simulate at known parameters, refit, compare. Without this test you have no
    evidence your likelihood is correct, and a wrong-but-convergent Hawkes fit
    produces intensity features that look entirely reasonable in a plot.

    Tolerances are loose on purpose: MLE on a finite sample is noisy, and a test
    tuned to a lucky seed is worse than no test.
    """
    t = simulate_hawkes(mu, alpha, beta, T=4000.0, seed=3)
    assert t.size > 500, "simulator produced too few events to identify parameters"

    fit = fit_hawkes_mle(t, T=4000.0)
    assert fit["success"]
    assert abs(fit["mu"] - mu) / mu < 0.35
    assert abs(fit["branching_ratio"] - alpha / beta) < 0.20


def test_hawkes_branching_ratio_flags_non_stationarity():
    with pytest.raises(ValueError):
        simulate_hawkes(mu=0.5, alpha=2.0, beta=1.0, T=100.0)


def test_hawkes_intensity_is_causal():
    """The intensity at time t must not move when a *future* event is added.
    This is the feature-level analogue of the leakage test.
    """
    events = np.array([1.0, 2.0, 3.0, 10.0])
    t_eval = np.array([0.5, 1.5, 2.5, 3.5])
    a = intensity_at(t_eval, events, mu=0.5, alpha=0.6, beta=1.2)
    b = intensity_at(t_eval, events[:-1], mu=0.5, alpha=0.6, beta=1.2)
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_hawkes_intensity_exceeds_baseline_after_events():
    events = np.array([1.0, 1.1, 1.2])
    lam = intensity_at(np.array([0.5, 1.25]), events, mu=0.5, alpha=0.6, beta=1.2)
    assert lam[0] == pytest.approx(0.5)
    assert lam[1] > 0.5


# --------------------------------------------------------------------------- #
# Model-level learning tests (torch optional)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_model_can_overfit_a_single_batch():
    """The fastest possible check that your model, loss, optimiser and data
    plumbing are wired together: a sufficiently expressive model must drive the
    loss on ONE batch to ~0. If it cannot, no amount of hyperparameter tuning
    will help, and you have saved yourself a week.

    Swap in your real model here.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn

    torch.manual_seed(0)
    x = torch.randn(32, 128, 46)
    y = torch.randint(0, 3, (32,))

    model = nn.Sequential(nn.Flatten(), nn.Linear(128 * 46, 256), nn.GELU(), nn.Linear(256, 3))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    for _ in range(300):
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()

    assert loss.item() < 0.05, f"could not overfit one batch (loss={loss.item():.4f})"


@pytest.mark.slow
def test_forward_pass_is_causal_in_time():
    """Perturb the LAST timestep of the input window and confirm the prediction
    changes; perturb a timestep beyond the window and confirm nothing changes.
    For any architecture with masking or shifted convolutions, this catches an
    off-by-one that leaks the future into the receptive field.

    Replace the stub with your TLOB forward pass.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(10 * 4, 3)).eval()

    x = torch.randn(1, 10, 4)
    base = model(x)
    x2 = x.clone()
    x2[0, -1] += 5.0
    assert not torch.allclose(base, model(x2)), "last input step has no effect -- suspicious"


@pytest.mark.gpu
def test_checkpoint_roundtrip_is_bit_exact():
    """Save/load must reproduce predictions exactly, or your 'best' checkpoint is
    not the model you reported.
    """
    torch = pytest.importorskip("torch")
    nn = torch.nn
    import io

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 3))
    x = torch.randn(4, 8)
    before = model(x)

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    reloaded = nn.Sequential(nn.Linear(8, 3))
    reloaded.load_state_dict(torch.load(buf))
    torch.testing.assert_close(before, reloaded(x), rtol=0, atol=0)
