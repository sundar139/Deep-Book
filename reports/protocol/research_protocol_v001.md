# DeepBook Research Protocol v001

**Status:** Planned — no experiments have been executed.
**Last updated:** 2026-07-30

---

## 1. Research Question

Can explicit order-arrival dynamics improve both the robustness of LOB
mid-price-movement prediction and the quality of execution decisions under
realistic market frictions?

The null hypothesis is that book-state alone captures all actionable information.
We test whether Hawkes-modeled event intensities add orthogonal signal that
survives negative controls and repeated temporal evaluation.

---

## 2. Research Tracks

### Track A — FI-2010 Published-Protocol Reproduction

Benchmark reproduction using the original FI-2010 dataset, feature set, train/
validation/test protocol, and label definitions.  Models: majority class,
previous movement, logistic/ridge, SVM, tree model, XGBoost/LightGBM, MLPLOB,
DeepLOB, TransLOB, TLOB.

### Track B — FI-2010 Leakage-Controlled Reevaluation

Identical FI-2010 dataset, but evaluated under chronological blocks with
purge-and-embargo splits, train-only normalization fitting, and horizon-
unbiased labels.  Same model set.  Track A results are reported alongside
Track B results; Track B is the primary FI-2010 claim.

### Track C — Modern Free Crypto L2 Research

Prospectively captured Binance and/or Coinbase public L2 order-book and trade
data.  Observable L2 event categories (bid/ask liquidity addition/removal,
buyer/seller-initiated trades).  Hawkes modeling with temporal, instrument,
venue, and regime generalization.  Realistic execution replay with queue-
position assumptions.

### Track D — Optional Paid MBO Validation

A later, explicitly authorized, quote-gated Databento MBO pilot.  Requires
human sign-off, a metadata-only quote before purchase, and a hard cost cap.
Default authorization is false.  No purchase has been made.

---

## 3. Prediction Metrics

### Primary
- macro-F1
- Matthews correlation coefficient (MCC)
- macro recall
- classwise precision and recall

### Secondary
- balanced accuracy
- negative log-likelihood
- Brier score
- expected calibration error (ECE)
- confusion matrices
- precision-recall curves
- inference latency (ms)
- parameter count
- GPU memory (GB)

---

## 4. Execution Metrics

### Primary
- implementation shortfall (bps)
- median shortfall (bps)
- 95% CVaR (bps)
- completion rate
- terminal inventory

### Secondary
- shortfall variance
- market impact
- fill rate
- passive-fill ratio
- spread paid or captured
- depth consumed
- action turnover
- inventory deviation
- inference latency (ms)

---

## 5. Evaluation Rules

### Split construction
- Chronological blocks only; never random-row shuffling.
- Train-only normalization fitting (mean, std computed from train slice).
- Purge and embargo around split boundaries:
  purge = window_length + horizon, embargo = 0 or greater.
- No windows or labels crossing sessions, gaps, or invalid reconstruction
  intervals.

### Test-set discipline
- Separate benchmark and blind-modern test sets.
- Blind test not used for: feature selection, model selection, threshold
  selection, normalization fitting, early stopping, Hawkes tuning, RL reward
  tuning, or simulator calibration.
- Regime thresholds derived from training data only.

### Statistical procedures
- Session, day, or block as the statistical unit — not overlapping windows.
- At least five seeds for normal predictor comparisons.
- Identical held-out execution episodes for paired comparisons.
- Block bootstrap confidence intervals.
- Paired permutation tests.
- Holm correction for multiple comparisons.
- Effect sizes reported alongside adjusted p-values.

---

## 6. Forecasting Baselines

1. majority class
2. previous movement
3. logistic regression / ridge
4. decision tree / random forest
5. SVM
6. XGBoost or LightGBM
7. MLPLOB
8. DeepLOB
9. TransLOB
10. TLOB
11. Hawkes-TLOB (proposed)

---

## 7. Execution Baselines

1. immediate execution
2. TWAP
3. online VWAP
4. oracle VWAP (unattainable, labeled as upper bound)
5. percentage of volume (POV)
6. Almgren–Chriss
7. passive schedule
8. random policy

---

## 8. Hawkes Negative Controls

1. true fitted intensities
2. constant-rate / Poisson intensity
3. raw recent event counts
4. within-day shuffled intensity
5. temporally shifted intensity
6. random synthetic channels

---

## 9. Execution-Signal Controls

1. book only
2. book + handcrafted flow features
3. book + Hawkes state
4. book + predictor logits
5. book + calibrated probabilities
6. probabilities + uncertainty
7. full state
8. shuffled predictor signal
9. oracle future signal (upper bound)

---

## 10. Queue Assumptions

1. optimistic (always at front)
2. proportional / central (queue position proportional to visible depth)
3. conservative (always at back)
4. **Conservative result used as the headline result.**

---

## 11. Stopping and Gate Rules

- No proposed model before credible baseline reproduction.
- No RL before simulator and static-baseline validation.
- No test-set tuning.
- Failed or negative results must be reported.
- Inability to reproduce a baseline must be documented, not hidden.
- No claim that Hawkes helps unless improvements survive negative controls
  and repeated temporal evaluation.

---

## 12. Success Definitions

| Outcome | Condition |
|---------|-----------|
| Minimum successful project | FI-2010 reproduction within published error margins, leakage-controlled reevaluation documented, at least one novel finding from Track C. |
| Strong positive outcome | Hawkes features improve prediction on both tracks after Holm correction; execution improvement survives conservative queue assumptions. |
| Exceptional outcome | Hawkes features provide complementary signal beyond what larger models capture alone; generalisation across instruments/venues demonstrated. |
| Inconclusive result | Improvements exist but do not survive all negative controls or are not statistically significant after correction. |
| Negative result | No reliable improvement; Hawkes does not add information beyond book state. Reported in full. |

---

## 13. Reproducibility Record

Every experiment records:
- Git commit (full SHA)
- dataset hash (SHA-256)
- configuration paths and hashes
- all random seeds
- exact command
- Python version and dependency snapshot
- hardware summary (CPU, RAM, GPU if used)
- output artifact paths and hashes

---

## 14. Decision Records

### Separate dataset/research tracks
The FI-2010 benchmark track, modern free L2 track, and optional MBO track are
kept as separate configurations to prevent contamination between published-
protocol reproduction and exploratory modern work.

### Observable L2 event taxonomy limitation
We use observable L2 event categories (liquidity addition/removal,
buyer/seller-initiated trades) rather than claiming exact MBO-level order
add/cancel/modify classification. This is a deliberate and documented
limitation of public L2 data.

### Blind-test and leakage policy
The blind test set is never used for any tuning decision. Chronological
splits with purge and embargo are mandatory. These rules are enforced by
invariant checks at runtime and in CI.

### Paid-data authorization policy
All paid data access requires explicit human authorization, a reviewed
quote, and a hard cost cap. The default is no authorization. No automatic
retry or un-reviewed purchase is permitted.
