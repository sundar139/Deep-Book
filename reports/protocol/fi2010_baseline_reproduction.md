# FI-2010 Baseline Reproduction Contract

**Status:** Frozen after engineering smoke tests and before any full confirmatory matrix
**Frozen at:** 2026-07-31T12:51:49Z
**Framework-freeze commit:** Resolved at runtime as the latest Git commit touching the frozen protocol, reference, and experiment-configuration files. Every new run manifest records that resolved `protocol_commit` together with `protocol_sha256`, a canonical digest over the complete frozen contract file set. A run is confirmatory only when its code commit descends from the recorded `protocol_commit` and the recorded digest still matches the working contract.
**Repeatability scope:** The architecture, hyperparameters, seeds, data contract, evaluation protocol, and acceptance bands recorded below are frozen and apply to all future confirmatory runs. This protocol was frozen after engineering smoke tests validated that models compile, datasets load, gradients flow, and checkpoints round-trip. No smoke-test metrics, parameter choices, or observed behavior influenced these frozen values.

**All prior one-epoch or partial runs are nonconfirmatory.** Their metrics are excluded from official aggregation. Smoke metrics must not influence architecture, seeds, acceptance thresholds, or full-run hyperparameters. The complete result matrix has not yet been evaluated. Test results may not be used for model selection.
**Dataset:** FI-2010, audited local extraction

## Sources

| Source | Revision or access identity | Use |
|---|---|---|
| DeepLOB paper | arXiv:1808.03668v6, `https://arxiv.org/abs/1808.03668`, accessed 2026-07-31 | Published architecture, FI-2010 setups, published comparison tables |
| Author repository | `https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books`, `master` commit `ff14d7c2fd38bdfc143389786993d0f0236d4eb8`, accessed 2026-07-31 | Primary PyTorch tensor and training reference; notebook blob `0fd2f77c7656f83580a3b159408d41d5f9e43d32` |
| FI-2010 source | Fairdata archive SHA-256 `cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664` | Audited source matrices and supplied labels |
| Repository data contract | `configs/data/fi2010.yaml`, parser version `1.1` | Selected variant, normalization, rows, horizons, provenance |

The archive digest is locally computed from the authoritative Fairdata archive. The inspected source metadata did not publish a per-file checksum. Labels are supplied by FI-2010 and are not regenerated.

## Fixed data contract

- Selected source: `no_auction`, z-score matrices.
- Internal orientation: variables by observation.
- Logical rows: 149.
- Input rows: first 40 LOB rows only; engineered rows 41--144 and label rows 145--149 are excluded from neural inputs.
- Input feature order: `{ask_price_i, ask_volume_i, bid_price_i, bid_volume_i}` for levels 1--10, as described by the paper and used by the author notebook.
- Labels: supplied rows for horizons 10, 20, 30, 50, and 100 events.
- Source labels 1, 2, 3 mean up, stationary, down. Model indices are 0, 1, 2 only at the training boundary, with the exact inverse mapping in reports.
- Sequence length: 100 historical observations.
- A sample ending at event `t` contains events `t-99` through `t`; its target is the selected supplied label at event `t`.
- Each independent day/file produces `N - 100 + 1` samples. Windows never cross a file, day, role, fold, or validation boundary.
- No supplied label row, future feature, engineered row, or test statistic enters an input or fitted object.

Daily identity is reconstructed only from audited published files: day 1 is training CF_1; days 2--10 are testing CF_1--CF_9. Cumulative training files are never concatenated. Their audited column-count differences equal the corresponding daily testing counts; this relationship is checked before use.

## FI-2010 evaluation setups

### Anchored forward folds

For fold `i` in 1--9:

- fit data: days 1 through `i`;
- validation data: training-only chronological tail, using the final complete day when at least two fit days exist, otherwise a blocked tail of the sole training day;
- test data: day `i+1` only.

The nine test days are reported separately. Fold means are unweighted across folds; pooled predictions and observation-weighted summaries are separate quantities.

### First seven / final three days

Recorded as `setup: first_seven_final_three`, `fold: null`, `day_group: days_8_9_10`.

- fit data: the single audited cumulative days 1--7 training matrix, used exactly once;
- validation data: the chronological training-only tail of that matrix, with the same purge and embargo;
- test data: the three distinct audited daily files for days 8, 9, and 10, reported separately and combined without duplicate observations.

The three test days remain separate source boundaries during window generation. Each day is windowed independently, so no window spans two days and no test observation is produced twice. Only completed per-day prediction arrays are concatenated, after each day has been windowed on its own. Every persisted sample keeps an integer `source_file_id` (the audited source fold of its file) and an integer `day_boundary_id` (8, 9, or 10); the manifest carries a `day_index_map` binding each day identifier to its audited source file digest and observation count.

### Validation and embargo

Validation is chronological and training-only. The fit/validation boundary has a purge and embargo of `sequence_length + maximum_horizon = 200` events where a complete boundary permits it. For a single available training day, a tail split is used with the same purge. Validation never selects from test data. Early stopping, learning-rate behavior, class weighting, architecture, epoch count, threshold, seed, and checkpoint selection are all fixed or selected from training data only.

The official notebook uses a contiguous 80/20 split of cumulative CF_7 and concatenates test CF_7--CF_9. This implementation deliberately keeps daily boundaries and uses a training-only chronological validation split; that is a leakage-control deviation, not a hidden substitution for published results.

## DeepLOB architecture contract

The author PyTorch notebook controls tensor details where it differs from the paper schematic. Input is `[B, 1, 100, 40]`; PyTorch dimensions are batch, channel, time, feature.

| Operation | Kernel / stride / padding | Channels | Output |
|---|---|---:|---|
| Input | -- | 1 | `[B,1,100,40]` |
| Conv1a + LeakyReLU(0.01) + BN | `1x2`, `1x2`, valid | 32 | `[B,32,100,20]` |
| Conv1b + LeakyReLU + BN | `4x1`, `1x1`, valid | 32 | `[B,32,97,20]` |
| Conv1c + LeakyReLU + BN | `4x1`, `1x1`, valid | 32 | `[B,32,94,20]` |
| Conv2a + Tanh + BN | `1x2`, `1x2`, valid | 32 | `[B,32,94,10]` |
| Conv2b + Tanh + BN | `4x1`, `1x1`, valid | 32 | `[B,32,91,10]` |
| Conv2c + Tanh + BN | `4x1`, `1x1`, valid | 32 | `[B,32,88,10]` |
| Conv3a + LeakyReLU + BN | `1x10`, `1x1`, valid | 32 | `[B,32,88,1]` |
| Conv3b + LeakyReLU + BN | `4x1`, `1x1`, valid | 32 | `[B,32,85,1]` |
| Conv3c + LeakyReLU + BN | `4x1`, `1x1`, valid | 32 | `[B,32,82,1]` |
| Inception branch 1 | `1x1` then `3x1`, same | 64 | `[B,64,82,1]` |
| Inception branch 2 | `1x1` then `5x1`, same | 64 | `[B,64,82,1]` |
| Inception branch 3 | max-pool `3x1`, stride 1, pad `(1,0)`, then `1x1` same | 64 | `[B,64,82,1]` |
| Concatenate branches | channel axis | 192 | `[B,192,82,1]` |
| Permute and flatten feature width | `[B,82,192,1]` then reshape | -- | `[B,82,192]` |
| LSTM | input 192, hidden 64, one layer, batch first | 64 | `[B,82,64]` |
| Last recurrent output | time index 81 | 64 | `[B,64]` |
| Linear classifier | 64 to 3 | 3 logits | `[B,3]` |

The official notebook has no dropout and initializes zero hidden/cell states. The implementation creates those states on the input device rather than reading a module-global device. The model returns logits; probabilities are computed with softmax by evaluation code. This differs from the notebook's terminal softmax because `CrossEntropyLoss` requires logits and the contract requires exactly three logits for correct NLL/Brier evaluation. The published paper schematic shows 16 convolution channels and 32-channel Inception branches, while the official PyTorch notebook uses 32 and 64; the official PyTorch code controls this implementation. The expected trainable count for the notebook architecture, including BatchNorm affine parameters, is 143,907. The paper's Table III reports approximately 60k parameters for its Keras implementation; both values are retained as a documented source discrepancy.

## Training contract

- Loss: `CrossEntropyLoss` on logits; no class weights unless a configuration explicitly records training-only weights. Default is no weights.
- Optimizer: Adam, learning rate `1e-4` for the reproducible PyTorch implementation. The paper reports `0.01` and the notebook uses `0.0001`; the notebook value controls this implementation.
- Batch size: 64 for the PyTorch baseline, fixed before held-out evaluation.
- Epoch limit: 50 for the local PyTorch reproduction; early stopping uses validation macro-F1 with patience 10. This is a declared deviation from the paper's approximately 100 epochs and patience 20.
- No scheduler, mixed precision, gradient clipping, or dropout.
- Seeds: 1337, 2027, 31415, 424242, 8675309 for neural experiments.
- Python, NumPy, and PyTorch seeds are fixed; DataLoader workers are seeded; deterministic algorithms and cuDNN settings are recorded. Hardware-independent bitwise identity is not claimed.
- Checkpoints are selected only by validation macro-F1, saved atomically, and contain model, optimizer, epoch, seed, configuration hash, data fingerprint, and validation metric.

## Classical and MLP-LOB contracts

- Majority: training-label frequencies with additive smoothing `1e-6`; hard prediction is the training argmax.
- Causal movement persistence: for horizon `h`, the prior supplied label at lag `h` is used only when its future-information interval has ended, and only within one independent source segment. The lag never reaches across a file, day, or role boundary, so the first `h` observations of every test day yield no prediction and are excluded from metrics rather than guessed.
- Logistic: multinomial `LogisticRegression` on the current event's 144 supplied non-label features, `lbfgs`, `C=1.0`, `max_iter=200`, fixed `random_state=1337`; the raw-LOB 40-feature form is available as a declared alternative.
- Tree: `RandomForestClassifier`, 200 trees, `max_depth=18`, `min_samples_leaf=2`, `max_features=sqrt`, training-only deterministic chronological cap of 100,000 rows when needed, `random_state` from the declared seed list.
- MLP-LOB: flattened `[100,40]` LOB input, hidden widths 128 and 64, ReLU activations, dropout 0.0, three logits, Adam and the same training/checkpoint contract. It is a fixed local comparator and is not named as another architecture.

## Metrics and aggregation

Class order is always `[up, stationary, down]`. Confusion matrices use rows for true classes and columns for predicted classes. Precision, recall, and F1 use zero division equal to zero. Primary metrics are macro-F1, MCC, classwise precision, and classwise recall. Secondary metrics are balanced accuracy, clipped-probability multiclass NLL, one-hot multiclass Brier score, fixed 10-bin ECE, and accuracy. Latency is measured after warm-up with synchronized CUDA timing when applicable. Parameter count, training duration, and peak GPU memory are recorded. Fold aggregation is an unweighted mean and standard deviation; day and pooled observation-weighted values are labeled separately.

## Acceptance and prohibited uses

For an exactly matched paper metric, the initial acceptance band is five absolute macro-F1 percentage points when no source uncertainty is supplied. No class may collapse without an explicit explanation. Qualitative horizon behavior must be consistent with the source. A result outside the band is reported, not retuned. It can only be accepted with independently verified alignment, metric, architecture, and protocol checks plus a reproducible source discrepancy explanation.

**Engineering smoke runs.** Runs classified as `smoke` (one epoch, dirty tree, mismatched config, pre-freeze code state, or explicitly marked smoke) are excluded from confirmatory reports. Their metrics are logged for engineering reference only and must never be mixed with confirmatory aggregate statistics. The report must clearly separate smoke outputs with per-run exclusion reasons.

**Run eligibility.** Eligibility is classified only after every artifact exists, and a run is confirmatory when all of the following hold: (a) the resolved protocol commit is an ancestor of the run's code commit; (b) the frozen contract digest is unchanged across the run; (c) the configuration hash still matches its configuration file; (d) the recorded data fingerprint equals the frozen FI-2010 data identity for that setup cell, and the archive digest equals the authoritative archive digest; (e) the tree was clean; (f) the status is `completed`; (g) a prediction artifact exists, loads, and has the recorded sample count; (h) for neural runs a best-model checkpoint exists with a matching digest, the termination reason is `early_stopping` or `max_epochs`, and `1 <= actual_epochs_completed <= configured_max_epochs` with `1 <= best_epoch <= actual_epochs_completed`; (i) for classical runs the epoch fields are null and the termination reason is `not_applicable`; and (j) no exclusion reason was recorded. Early stopping is a valid confirmatory outcome. Two manifests claiming the same logical identity (model, setup, fold or day group, horizon, seed, configuration hash, data fingerprint, run kind) are both excluded from aggregation.

**Frozen data identity.** `configs/references/fi2010_frozen_data_identity.yaml` records the authoritative archive digest and the audited per-file digests and observation counts for every fold and for the days 8--10 group. Data identity is never compared only against itself: each run's recorded fingerprint is recomputed from that frozen file and must match.

**Checkpoints.** Two checkpoints are kept per neural run and both are ignored by version control. `<run-id>.best.pt` holds the validation-selected weights and is the only checkpoint used for final evaluation and test prediction. `<run-id>.last.pt` is rewritten atomically at the end of every completed epoch and is the only checkpoint an interrupted run may resume from; resuming from the best checkpoint is rejected because it holds stale weights. Shuffle order for epoch `e` is a pure function of the base seed and `e`, so a resumed epoch replays exactly the order the uninterrupted run would have used.

**Reference comparison.** A published reference value is compared only when the matching model, setup, and horizon cell has complete coverage: all nine folds for the anchored setup, or the complete days 8--10 group for the first-seven/final-three setup, at every required seed. Incomplete cells are reported as `INCOMPLETE — no reference conclusion`. A single fold is never compared with a nine-fold published mean.

Test labels and predictions are never used for fitting, normalization, class weighting, learning-rate choice, architecture choice, epoch choice, threshold choice, seed choice, calibration, or checkpoint selection. No horizon-unbiased or cost-aware labels are introduced. No execution simulation or reinforcement learning is part of this contract.

Expected hardware is Windows 10, Python 3.11, NVIDIA RTX 4070 with CUDA-capable PyTorch, and the project-local `.venv`. Software versions, driver, GPU, cuDNN, configuration hash, source hashes, commit, dirty status, checkpoint hash, prediction hash, and run timestamps are recorded in every manifest.
