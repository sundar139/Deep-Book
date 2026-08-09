# FI-2010 Transformer Architecture Freeze Amendment

Status: frozen before any confirmatory TransLOB or TLOB execution.
Audit date: 2026-08-08

## Purpose and hypotheses

TransLOB tests whether replacing recurrent temporal modeling with the source-backed causal-convolution plus masked-attention architecture improves FI-2010 predictive performance and/or robustness relative to DeepLOB under identical DeepBook labels and splits.

TLOB tests whether source-backed dual spatial/temporal attention improves predictive performance and robustness relative to DeepLOB and TransLOB under identical DeepBook labels and splits.

Neither hypothesis assumes that the new model wins. Negative results are valid and will be reported.

## Source hierarchy

The TransLOB authority is Wallbridge, *Transformers for Limit Order Books*, arXiv:2003.00130v1, together with the author's repository `jwallbridge/translob` at commit `54d73260560cae4282b5effbff7c4f2158e62c08`. The TLOB authority is Berti and Kasneci, *TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data*, arXiv:2502.15757v3, together with `LeonardoBerti00/TLOB` at commit `f1c0af4d81067978914361766db0457a7d8b6a46`. Detailed file digests and classifications are in `configs/references/translob_fi2010.yaml` and `configs/references/tlob_fi2010.yaml`.

Paper statements and official source code were reconciled explicitly. The following source conflicts and adaptations are documented:

- **TransLOB QKV bias**: The official repository uses a bare weight matrix with no bias. DeepBook initially used `nn.Linear(..., bias=True)`. Corrected to `bias=False` to align with the authoritative repository.
- **TransLOB W^O output projection**: The paper equation MultiHead(X) = concat(heads)W^O includes a learned output projection. The official repository attention module omits it. DeepBook retains W^O following the paper equation. Classified as AMBIGUOUS_SOURCE_CONFLICT.
- **TransLOB L2 regularization**: The paper specifies L2 regularization on the dense-64 classifier layer. The exact coefficient is not recoverable from the paper or repository. DeepBook intentionally omits L2 regularization rather than guessing a coefficient. Classified as an intentional source-ambiguity adaptation.
- **TLOB block order**: Each TransformerLayer contains its own internal FFN/MLP. There is no separate "MLPLOB block" appended per layer. The previous reference metadata wording has been corrected.
- **TLOB preflight learning rate**: A diagnostic tiny-batch overfit learning rate of 1e-3 is used exclusively for preflight diagnostics. The frozen production learning rate is 1e-4 (Adam, batch 32, epochs 10). The diagnostic value must never be inherited by confirmatory matrix training.

## Controlled comparison contract

The controlled comparison uses the already accepted DeepBook FI-2010 representation: 40 raw LOB rows, 100-event windows, supplied official labels, horizons 10, 20, 30, 50, and 100, the existing two setups, five frozen seeds, train-only normalization, chronological validation, purge, embargo, checkpoint selection, and existing metric implementation.

Labels are not regenerated. The TLOB horizon-unbiased labeling method is explicitly excluded from this comparison and is reserved for a future separately preregistered ablation. No result using that method may enter the official-label benchmark.

The source TransLOB and TLOB horizon set of 10, 20, 50, and 100 is contextual only. It does not change the controlled five-horizon matrix.

A literature value is a direct reproduction only when labels, horizon, split, preprocessing, metric, and aggregation all match. Otherwise the value is contextual and marked `partial` or `unmatched`; no silent benchmark modification or post-hoc tolerance is permitted.

## Frozen model contracts

TransLOB uses five causal Conv1D layers with 14 channels, kernel width 2, dilations 1/2/4/8/16, ReLU, LayerNorm, the official linear -1-to-1 position coordinate, model dimension 15, three masked-attention heads with QKV bias=False (aligned to official repository), two shared transformer iterations, 4x ReLU feed-forward width, no internal transformer dropout, a learned attention output projection W^O retained per the paper equation (AMBIGUOUS_SOURCE_CONFLICT with repository), a 64-unit ReLU classifier with no L2 regularization (coefficient not recoverable from authoritative source), 0.1 classifier dropout, and three output logits. Corrected trainable parameter count: 101,895 (down from 101,940 after QKV bias removal).

TLOB uses the official temporal-then-spatial block order (each TransformerLayer contains its own internal FFN/MLP; there is no separate appended MLPLOB block), four temporal and four spatial attention layers, Bilinear Normalization, sinusoidal positional encoding, hidden dimension 40, one head, and three logits. Its DeepBook adaptation uses 40 raw features and 100 events rather than the source FI-2010 all-feature/128-event path. Attention is non-causal because the authoritative TLOB source does not require a causal mask. The LOBSTER order-type embedding is not applicable to DeepBook FI-2010.

All exact values are frozen in `configs/experiments/fi2010/translob.yaml` and `configs/experiments/fi2010/tlob.yaml`. The shared runner supplies Adam, source learning rates, source batch sizes, source epoch limits, validation-only early stopping, deterministic seeds, best/last checkpoints, and CrossEntropyLoss over logits. Probabilities are produced only at evaluation.

## Validation and leakage controls

Training windows use training partitions only. Validation is derived from training history only. Existing purge and embargo rules remain active. No fold or final-three test-day information enters fitting, normalization, checkpoint selection, calibration, or model construction. Windows are independent at day/file boundaries. Test metrics are read only after the best validation checkpoint is selected.

## Metrics and interpretation

Primary metrics are macro-F1, MCC, classwise precision, and classwise recall. Secondary metrics are balanced accuracy, accuracy, NLL, Brier, ECE, classwise F1, confusion matrix, sample count, inference latency, training duration, peak GPU memory, and parameter count.

A complete matrix supports a controlled comparison. A result is successful only in the descriptive sense that the preregistered comparison is complete and interpretable; no numeric performance threshold is introduced. A result is inconclusive when coverage or validation integrity is incomplete. A negative result is a valid finding and is reported without model removal or relabeling.

## Planned cells

The runner plans exactly 225 Setup 1 cells and 25 Setup 2 cells per transformer model: 250 TransLOB cells and 250 TLOB cells. Together with the accepted 900 cells, the future complete architecture-comparison matrix contains 1,400 cells. This amendment does not execute any of the 500 new cells.

Smoke, tiny-overfit, CUDA, memory, throughput, and resume checks are engineering preflights only. They use temporary artifacts, are marked `run_kind: smoke`, and are never eligible for confirmatory reporting.

## Freeze and execution boundary

This amendment, both frozen model configurations, source references, and the implementation must be committed before production execution. No test-derived tuning is permitted after this freeze. No alternative labels, Hawkes features, modern-data collection, execution simulation, or reinforcement learning are part of this amendment.
