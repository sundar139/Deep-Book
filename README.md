# DeepBook

Leakage-controlled limit-order-book forecasting and execution research.

**Central question:** Can explicit order-arrival dynamics improve both the
robustness of LOB mid-price-movement prediction and the quality of execution
decisions under realistic market frictions?

---

## Research Tracks

| Track | Data | Scope |
|-------|------|-------|
| FI-2010 reproduction | FI-2010 benchmark | Published-protocol baseline + leakage-controlled reevaluation |
| Modern crypto L2 | Binance / Coinbase public L2 | Hawkes modeling, temporal/regime generalization, execution replay |
| Optional MBO | Databento MBO (paid) | Quote-gated validation pilot (not yet authorized) |

### L2 Data Limitations

Public L2 order-book feeds report observable aggregate events (liquidity
addition/removal at price levels, trades). They do not reveal exact
order-level add/cancel/modify messages. This limitation is documented
throughout the methodology. The optional MBO track exists to quantify
what, if anything, is lost by working with L2 data.

---

## Repository Layout

```
.github/workflows/    CI (lint, type-check, test, secret scan)
configs/              Experiment configuration and registries
data_contracts/       JSON Schema contracts for events, books, manifests
experiments/          Example manifests
reports/protocol/     Research protocol and decision records
src/deepbook/         Package source
tests/                Unit, property, smoke, data tests
```

---

## Setup (Windows, PowerShell)

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Editable install with dev dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[dev,data,model]"

# Verify
python -m deepbook.cli.doctor
```

---

## Quality Commands

```powershell
# Lint
python -m ruff check .

# Format check
python -m ruff format --check .

# Type check
python -m mypy src

# Tests
python -m pytest tests/unit tests/property tests/smoke

# Pre-commit (all files)
python -m pre_commit run --all-files

# Local aggregate check
python scripts/check.py
```

---

## Environment Doctor

```powershell
python -m deepbook.cli.doctor
```

Reports package version, Python version, venv status, Git commit, paid-data
authorization state, and configured data/artifact roots. Exits nonzero if the
interpreter is outside the project `.venv` or an unsafe paid-data
configuration is detected. Never prints secret values.

---

## FI-2010 Acquisition and Audit

```powershell
# Authoritative Fairdata download, integrity checks, and safe extraction
python -m deepbook.data.fi2010.cli acquire

# Published-matrix and supplied-label audit
python -m deepbook.data.fi2010.cli audit

# Explicit real-data acceptance test (normally skipped without local data)
python -m pytest -m data -vv
```

The archive is pinned to the Fairdata-published byte size and a locally computed SHA-256. Raw data,
extracted files, acquisition records, manifests, and generated audit reports are
Git-ignored. See `reports/protocol/fi2010_data_provenance.md` for source identity,
license, layout, safeguards, output paths, and known metadata limitations.

---

## Data and Secret Policy

- `.env` is **git-ignored**.  Never commit credentials.
- Raw data lives under `data/raw/` (git-ignored, future DVC tracking).
- No paid data has been requested or downloaded.
- The `configs/spending_policy.yaml` encodes financial guardrails.
- All paid access requires explicit human authorization and a reviewed
  quote with a hard cost cap.

---

## Reproducibility

- Chronological splits with purge and embargo.
- Train-only normalization fitting.
- Blind test set never used for tuning.
- All experiments record commit SHA, dataset hash, configuration, seeds,
  dependency snapshot, hardware summary, and output artifact hashes.
- See `reports/protocol/research_protocol_v001.md` for the full protocol.

---

## Current Status

Repository foundation, the authoritative FI-2010 acquisition/audit pipeline, and the
ordered FI-2010 baseline runner are in place. The runner preserves supplied labels,
uses chronological training-only selection with purge/embargo, records auditable
manifests, and implements majority, causal persistence, multinomial logistic,
RandomForest, MLP-LOB, and the reference-controlled DeepLOB comparator. Results are
local reproductions, not publisher-verified benchmark values. No prohibited market
data collection, Hawkes feature extraction, execution simulation, reinforcement
learning, TransLOB, or TLOB implementation is included.
