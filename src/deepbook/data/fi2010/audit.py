"""Deterministic FI-2010 provenance, split-manifest, and label audit."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import deepbook
from deepbook.data.fi2010 import PARSER_VERSION
from deepbook.data.fi2010.acquisition import (
    AcquisitionError,
    ArchiveInventory,
    ArchiveLimits,
    archive_inventory_json,
    config_sha256,
    inspect_zip,
    sha256_file,
    verify_extraction_manifest,
)
from deepbook.data.fi2010.dataset import (
    CLASS_SEMANTICS,
    ENGINEERED_FEATURE_ROWS,
    HORIZONS,
    LOB_ROWS,
    MatrixError,
    discover_matrices,
    fold_day_indices,
    select_matrices,
    stream_matrix_audit,
)

_INSTRUMENTS = ("KESBV", "OUT1V", "SAMPO", "RTRKS", "WRT1V")
_LIMITATIONS = (
    "The processed aggregate matrices contain all five instruments consecutively but do not "
    "retain trustworthy per-observation day, instrument, timestamp, order identity, exact queue "
    "position, or complete message-stream boundaries. File and published fold boundaries are "
    "preserved; no finer boundaries are guessed."
)


class AuditError(RuntimeError):
    """Raised when the full-data audit cannot establish a passing result."""


def _git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0 or status.returncode != 0:
        return "unknown", True
    return commit.stdout.strip(), bool(status.stdout.strip())


def _limits(config: dict[str, Any]) -> ArchiveLimits:
    raw = config["archive_limits"]
    return ArchiveLimits(
        maximum_archive_bytes=int(raw["maximum_archive_bytes"]),
        maximum_extracted_bytes=int(raw["maximum_extracted_bytes"]),
        maximum_member_bytes=int(raw["maximum_member_bytes"]),
        maximum_member_count=int(raw["maximum_member_count"]),
        maximum_compression_ratio=float(raw["maximum_compression_ratio"]),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _hierarchy(inventory: ArchiveInventory) -> dict[str, Any]:
    top_level: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    for member in inventory.members:
        top_level[member.path.split("/", 1)[0]] += 1
        if not member.is_directory:
            suffixes[Path(member.path).suffix.lower() or "<none>"] += 1
    return {
        "top_level_member_counts": dict(sorted(top_level.items())),
        "file_extension_counts": dict(sorted(suffixes.items())),
    }


def _aggregate_label_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    def empty_horizons() -> dict[str, Counter[str]]:
        return {
            str(horizon): Counter({str(label): 0 for label in CLASS_SEMANTICS})
            for horizon in HORIZONS
        }

    overall = empty_horizons()
    by_role: dict[str, dict[str, Counter[str]]] = defaultdict(empty_horizons)
    by_fold_role: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(empty_horizons)
    )
    for record in records:
        role = str(record["role"])
        fold = str(record["fold"])
        for horizon in map(str, HORIZONS):
            counts = record["label_distributions"][horizon]["counts"]
            overall[horizon].update(counts)
            by_role[role][horizon].update(counts)
            by_fold_role[fold][role][horizon].update(counts)

    def finalize(source: dict[str, Counter[str]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for horizon in map(str, HORIZONS):
            counts = {label: int(source[horizon][label]) for label in map(str, CLASS_SEMANTICS)}
            total = sum(counts.values())
            result[horizon] = {
                "counts": counts,
                "proportions": {
                    label: count / total if total else 0.0 for label, count in counts.items()
                },
            }
        return result

    return {
        "overall_file_rows": finalize(overall),
        "by_role": {role: finalize(values) for role, values in sorted(by_role.items())},
        "by_fold_and_role": {
            fold: {role: finalize(values) for role, values in sorted(roles.items())}
            for fold, roles in sorted(by_fold_role.items(), key=lambda item: int(item[0]))
        },
    }


def _observation_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_fold: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_role: Counter[str] = Counter()
    for record in records:
        count = int(record["observation_count"])
        role = str(record["role"])
        by_fold[str(record["fold"])][role] += count
        by_role[role] += count
    first_training = sum(
        int(record["observation_count"])
        for record in records
        if record["fold"] == 1 and record["role"] == "training"
    )
    protocol_unique = first_training + by_role["testing"]
    return {
        "by_file": {
            str(record["member_path"]): int(record["observation_count"]) for record in records
        },
        "by_fold_and_role": {
            fold: dict(sorted(roles.items()))
            for fold, roles in sorted(by_fold.items(), key=lambda item: int(item[0]))
        },
        "by_role_including_anchored_repetition": dict(sorted(by_role.items())),
        "all_file_columns_including_anchored_repetition": sum(by_role.values()),
        "protocol_unique_representations": protocol_unique,
        "protocol_unique_definition": (
            "training fold 1 (published day index 1) plus all testing files "
            "(published day indices 2 through 10)"
        ),
    }


def _split_manifests(
    records: list[dict[str, Any]], archive_sha256: str, variant: str, normalization: str
) -> dict[str, Any]:
    splits = []
    for record in records:
        fold = int(record["fold"])
        role = str(record["role"])
        splits.append(
            {
                "protocol_identifier": "fi2010_anchored_forward_cross_validation",
                "benchmark_variant": variant,
                "normalization": normalization,
                "source_archive_sha256": archive_sha256,
                "fold_identifier": fold,
                "role": role,
                "source_member_path": record["member_path"],
                "source_orientation": record["source_orientation"],
                "internal_orientation": record["internal_orientation"],
                "row_count": record["row_count"],
                "observation_count": record["observation_count"],
                "feature_row_count": LOB_ROWS + ENGINEERED_FEATURE_ROWS,
                "lob_row_count": record["lob_row_count"],
                "engineered_feature_row_count": record["engineered_feature_row_count"],
                "label_row_count": record["label_row_count"],
                "horizons_events": record["horizons_events"],
                "class_encodings": record["class_encodings"],
                "label_counts_by_horizon": {
                    horizon: distribution["counts"]
                    for horizon, distribution in record["label_distributions"].items()
                },
                "observed_missing_tokens": record["observed_missing_tokens"],
                "observed_nonfinite_tokens": record["observed_nonfinite_tokens"],
                "rejected_rows": record["rejected_rows"],
                "validated_nonfinite_values": record["validated_nonfinite_values"],
                "file_sha256": record["file_sha256"],
                "parser_version": PARSER_VERSION,
                "known_day_indices": fold_day_indices(fold, role),
                "known_instruments": list(_INSTRUMENTS),
                "unresolved_boundary_limitations": _LIMITATIONS,
            }
        )
    return {"schema_version": 1, "splits": splits}


def _source_manifest(
    config: dict[str, Any],
    inventory: ArchiveInventory,
    extraction_manifest: dict[str, Any],
    generated_utc: str,
    git_commit: str,
    variant: str,
    normalization: str,
) -> dict[str, Any]:
    dataset = config["dataset"]
    source = config["source"]
    return {
        "schema_version": 1,
        "dataset_name": dataset["name"],
        "dataset_title": dataset["title"],
        "persistent_identifier": dataset["persistent_identifier"],
        "source_landing_page_identifier": dataset["landing_page_identifier"],
        "source_type": "authoritative_fairdata_ida_archive",
        "accessed_utc": generated_utc,
        "archive_filename": source["archive_filename"],
        "archive_size_bytes": source["archive_size_bytes"],
        "archive_sha256": source["archive_sha256"],
        "checksum_provenance": source["checksum_provenance"],
        "access_type": dataset["access_type"],
        "license": dataset["license"],
        "rights_holder": dataset["rights_holder"],
        "member_count": inventory.member_count,
        "extracted_byte_size": extraction_manifest["extracted_bytes"],
        "selected_benchmark_variant": variant,
        "selected_normalization": normalization,
        "acquisition_command": "python -m deepbook.data.fi2010.cli acquire",
        "acquisition_git_commit": git_commit,
        "status": "audited",
        "deviations": [],
        "notes": [
            (
                "The SHA-256 was computed locally from the archive downloaded from the "
                "authoritative Fairdata source. The inspected Metax metadata did not publish "
                "a per-file checksum. The locally computed digest is pinned for subsequent "
                "integrity and reproducibility checks. This locally computed digest does not "
                "independently establish that the authoritative server was uncompromised at "
                "acquisition time."
            ),
            (
                "Fairdata authorization returns a temporary URL valid for 72 hours; "
                "that URL is not recorded."
            ),
            _LIMITATIONS,
        ],
    }


def _duplicates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[str(record["file_sha256"])].append(str(record["member_path"]))
    return [
        {"sha256": digest, "member_paths": paths}
        for digest, paths in sorted(grouped.items())
        if len(paths) > 1
    ]


def _fingerprint(value: dict[str, Any]) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def build_audit(
    *,
    repository_root: Path,
    config_path: Path,
    archive_path: Path,
    extraction_root: Path,
    generated_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the real archive and return audit, source, and split documents."""
    from deepbook.data.fi2010.acquisition import load_config

    config = load_config(config_path)
    source = config["source"]
    expected_size = int(source["archive_size_bytes"])
    expected_sha256 = str(source["archive_sha256"])
    if not archive_path.is_file() or archive_path.stat().st_size != expected_size:
        raise AuditError("authoritative archive is absent or has the wrong byte size")
    if sha256_file(archive_path) != expected_sha256:
        raise AuditError("authoritative archive SHA-256 does not match source metadata")

    try:
        inventory = inspect_zip(archive_path, _limits(config))
        extraction_manifest, _ = verify_extraction_manifest(extraction_root, expected_sha256)
    except AcquisitionError as error:
        raise AuditError(str(error)) from error
    member_paths = [member.path for member in inventory.members if not member.is_directory]
    matrices = discover_matrices(extraction_root, member_paths)
    selection = config["selection"]
    variant = str(selection["benchmark_variant"])
    normalization = str(selection["normalization"])
    try:
        selected = select_matrices(matrices, variant, normalization)
    except MatrixError as error:
        raise AuditError(str(error)) from error

    records: list[dict[str, Any]] = []
    for source_file in selected:
        try:
            records.append(stream_matrix_audit(source_file, extraction_root))
        except MatrixError as error:
            raise AuditError(str(error)) from error

    discovered_configurations = Counter(
        (matrix.benchmark_variant, matrix.normalization) for matrix in matrices
    )
    fold_count = len({int(record["fold"]) for record in records})
    expectations = config["published_expectations"]
    expected_fold_count = int(expectations["fold_count"])
    if fold_count != expected_fold_count:
        raise AuditError(f"observed {fold_count} folds, expected {expected_fold_count}")
    expected_matrix_count = int(expectations["selected_matrix_count"])
    if len(records) != expected_matrix_count:
        raise AuditError(
            f"observed {len(records)} selected matrices, expected {expected_matrix_count}"
        )

    timestamp = generated_utc or datetime.now(UTC).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    git_commit, git_dirty = _git_state(repository_root)
    source_document = _source_manifest(
        config,
        inventory,
        extraction_manifest,
        timestamp,
        git_commit,
        variant,
        normalization,
    )
    split_document = _split_manifests(records, expected_sha256, variant, normalization)
    deterministic_results = {
        "source_archive_sha256": expected_sha256,
        "source_archive_size_bytes": expected_size,
        "archive_inventory": archive_inventory_json(inventory),
        "archive_hierarchy_summary": _hierarchy(inventory),
        "extracted_tree": extraction_manifest,
        "discovered_matrix_configurations": [
            {
                "benchmark_variant": key[0],
                "normalization": key[1],
                "file_count": count,
            }
            for key, count in sorted(discovered_configurations.items())
        ],
        "selected_files": records,
        "observation_counts": _observation_counts(records),
        "label_distributions": _aggregate_label_counts(records),
        "duplicate_file_hashes": _duplicates(records),
        "observed_missing_tokens": sum(
            int(record["observed_missing_tokens"]) for record in records
        ),
        "observed_nonfinite_tokens": sum(
            int(record["observed_nonfinite_tokens"]) for record in records
        ),
        "rejected_rows": sum(int(record["rejected_rows"]) for record in records),
        "validated_nonfinite_values": sum(
            int(record["validated_nonfinite_values"]) for record in records
        ),
        "constant_row_findings": {
            str(record["member_path"]): record["constant_rows"] for record in records
        },
        "all_zero_row_findings": {
            str(record["member_path"]): record["all_zero_rows"] for record in records
        },
        "split_manifests": split_document,
        "unresolved_metadata_limitations": [_LIMITATIONS],
    }
    checks = {
        "archive_valid": True,
        "extraction_valid": True,
        "matrices_valid": all(record["row_count"] == 149 for record in records),
        "labels_valid": all(
            set(record["class_encodings"]) == {"1", "2", "3"} for record in records
        ),
        "counts_complete": all(
            sum(distribution["counts"].values()) == record["observation_count"]
            for record in records
            for distribution in record["label_distributions"].values()
        ),
        "no_unresolved_corruption": True,
    }
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "command": "python -m deepbook.data.fi2010.cli audit",
        "generated_utc": timestamp,
        "code_commit": git_commit,
        "git_tree_dirty": git_dirty,
        "package_version": deepbook.__version__,
        "python_version": platform.python_version(),
        "configuration_path": config_path.relative_to(repository_root).as_posix(),
        "configuration_sha256": config_sha256(config_path),
        "parser_version": PARSER_VERSION,
        "checks": checks,
        "source_manifest": source_document,
        "results": deterministic_results,
    }
    audit["data_fingerprint"] = _fingerprint(deterministic_results)
    return audit, source_document, split_document


def _markdown(audit: dict[str, Any]) -> str:
    results = audit["results"]
    observations = results["observation_counts"]
    lines = [
        "# FI-2010 Data Audit",
        "",
        f"- Status: **{audit['status']}**",
        f"- Generated UTC: `{audit['generated_utc']}`",
        f"- Code commit: `{audit['code_commit']}` (dirty: `{audit['git_tree_dirty']}`)",
        f"- Source archive SHA-256: `{results['source_archive_sha256']}`",
        f"- Source archive bytes: `{results['source_archive_size_bytes']}`",
        f"- Data fingerprint: `{audit['data_fingerprint']}`",
        f"- Parser version: `{audit['parser_version']}`",
        "",
        "## Selected benchmark data",
        "",
        f"- Variant: `{audit['source_manifest']['selected_benchmark_variant']}`",
        f"- Normalization: `{audit['source_manifest']['selected_normalization']}`",
        f"- Processed matrices: `{len(results['selected_files'])}`",
        f"- Unique processed representations: `{observations['protocol_unique_representations']}`",
        (
            "- Columns across anchored files (includes repeated earlier days): "
            f"`{observations['all_file_columns_including_anchored_repetition']}`"
        ),
        "- Layout: 149 rows = 40 LOB + 104 supplied engineered features + 5 labels",
        "- Label horizons: 10, 20, 30, 50, 100 events",
        "- Raw classes: 1=up, 2=stationary, 3=down",
        "",
        "## Observations by fold and role",
        "",
        "| Fold | Training | Testing |",
        "|---:|---:|---:|",
    ]
    for fold, counts in observations["by_fold_and_role"].items():
        lines.append(f"| {fold} | {counts.get('training', 0)} | {counts.get('testing', 0)} |")
    lines.extend(["", "## Label distributions by role", ""])
    for role, horizons in results["label_distributions"]["by_role"].items():
        lines.extend(
            [
                f"### {role.title()}",
                "",
                "| Horizon | Up | Stationary | Down |",
                "|---:|---:|---:|---:|",
            ]
        )
        for horizon, distribution in horizons.items():
            counts = distribution["counts"]
            proportions = distribution["proportions"]
            cells = [f"{counts[label]} ({proportions[label]:.6%})" for label in ("1", "2", "3")]
            lines.append(f"| {horizon} | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")
    lines.extend(
        [
            "## Quality findings",
            "",
            f"- Observed missing tokens: `{results['observed_missing_tokens']}`",
            f"- Observed nonfinite tokens: `{results['observed_nonfinite_tokens']}`",
            f"- Rejected rows: `{results['rejected_rows']}`",
            f"- Validated nonfinite values: `{results['validated_nonfinite_values']}`",
            f"- Duplicate selected-file hashes: `{len(results['duplicate_file_hashes'])}`",
            "- Constant and all-zero row indices are recorded per file in the JSON report.",
            "- Per-row minimum, maximum, mean, and standard deviation are recorded in JSON.",
            "",
            "## Metadata limitation",
            "",
            results["unresolved_metadata_limitations"][0],
            "",
            "The complete archive/member inventory, per-file statistics, fold-role-horizon label "
            "counts, and split manifests are in the JSON report.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_outputs(
    audit: dict[str, Any],
    source_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    *,
    interim_root: Path,
    report_root: Path,
) -> dict[str, Path]:
    """Atomically write machine-readable manifests and audit reports."""
    paths = {
        "source_manifest": interim_root / "fi2010_source_manifest.json",
        "split_manifest": interim_root / "fi2010_split_manifest.json",
        "audit_json": report_root / "fi2010_data_audit.json",
        "audit_markdown": report_root / "fi2010_data_audit.md",
    }
    _write_json(paths["source_manifest"], source_manifest)
    _write_json(paths["split_manifest"], split_manifest)
    _write_json(paths["audit_json"], audit)
    _write_text(paths["audit_markdown"], _markdown(audit))
    return paths
