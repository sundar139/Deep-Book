# FI-2010 Data Provenance and Audit

## Authoritative records

The acquisition target is the Fairdata-hosted archive associated with the paper
“Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine
Learning Methods” by Ntakaris et al.

- Paper: https://arxiv.org/abs/1705.03233
- DOI: https://doi.org/10.1002/for.2543
- Fairdata landing page: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
- Persistent identifier: `urn:nbn:fi:csc-kata20170601153214969115`
- Fairdata record identifier: `73eb48d7-4dbc-4a10-a52a-da745b47a649`
- Authoritative archive path: `/published/BenchmarkDatasets/BenchmarkDatasets.zip`
- Published archive bytes: `1864361899`
- Locally computed SHA-256 of the authoritative Fairdata archive: `cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664`
- Inspected Fairdata Metax metadata exposed the archive size but no per-file SHA-256.
- License: Creative Commons Attribution 4.0
- Rights holder: BigDataFinance

Fairdata requires a short-lived download authorization. The client POSTs the record
identifier and authoritative file path to the configured Fairdata authorization endpoint.
The returned URL is used only in memory, is redacted in console output, and is never
written to a tracked file or generated manifest.

The paper and Fairdata record are authoritative. The official DeepLOB, TransLOB, and TLOB
repositories are interpretation cross-checks only; they are not acquisition sources.

## Local acquisition

Use the repository-local environment:

```powershell
python -m deepbook.data.fi2010.cli acquire
```

The command:

1. requests a temporary URL from Fairdata;
2. streams into an atomic partial file with hard byte limits;
3. requires the archive byte size exposed by Fairdata metadata and the locally pinned SHA-256;
4. requires ZIP content matching the `.zip` extension;
5. rejects traversal, absolute paths, drive-qualified paths, links, encrypted members,
   duplicate normalized targets, conflicting file/directory targets, excessive member
   counts, excessive sizes, and suspicious compression ratios;
6. extracts into a temporary directory and atomically promotes it;
7. records every extracted file size and SHA-256 in an ignored extraction manifest.

Raw and extracted files remain under `data/raw/fi2010/`, which Git ignores. Generated
manifests remain under `data/interim/fi2010/`, also ignored. Interrupted downloads use a
`.part` suffix and are deleted after validation failures.

Re-running acquisition validates and preserves a matching archive and extraction tree.
Use `--offline` to forbid authorization and network access. `--source-url` and
`--expected-sha256` are runtime-only recovery controls; never put temporary URLs in
configuration, documentation, logs, or commits.

## Published matrix interpretation

The selected configuration is `NoAuction` with `ZScore` normalization. A processed matrix
has 149 logical rows:

- rows 1–40: ten LOB levels, with ask price/volume and bid price/volume fields;
- rows 41–144: 104 supplied engineered features;
- rows 145–149: supplied labels for horizons 10, 20, 30, 50, and 100 events.

The audit preserves labels exactly as supplied. Raw classes are `1=up`, `2=stationary`,
and `3=down`. The paper defines them from smoothed future mid-price movement using
threshold `0.002`; the repository does not recompute or replace those labels.

The benchmark supplies nine anchored forward-validation fold pairs. Training file `k`
represents published day indices 1 through `k`; testing file `k` represents published day
index `k+1`. Aggregate matrices concatenate five instruments: `KESBV`, `OUT1V`, `SAMPO`,
`RTRKS`, and `WRT1V`.

The processed aggregate files do not retain trustworthy per-observation instrument, day,
timestamp, order identity, exact queue position, or complete message-stream boundaries.
The audit records file and published fold boundaries and does not guess finer boundaries.

## Full audit

```powershell
python -m deepbook.data.fi2010.cli audit
python -m pytest -m data -vv
```

The audit revalidates archive and extracted-file checksums, parses matrices with bounded
memory, validates orientation and the 149-row layout, enforces label domains, and reports:

- archive hierarchy and full member inventory;
- observation counts by file, fold, and role;
- label counts and proportions by fold, role, and horizon;
- per-row minimum, maximum, mean, and standard deviation;
- missing, nonfinite, constant-row, all-zero-row, and duplicate-file findings;
- deterministic source and split manifests;
- an audit fingerprint over deterministic results.

Generated outputs are ignored and written atomically:

- `data/interim/fi2010/fi2010_source_manifest.json`
- `data/interim/fi2010/fi2010_split_manifest.json`
- `reports/results/fi2010/fi2010_data_audit.json`
- `reports/results/fi2010/fi2010_data_audit.md`

The manifest contracts are tracked at
`data_contracts/fi2010_source_manifest.schema.json` and
`data_contracts/fi2010_split_manifest.schema.json`.
