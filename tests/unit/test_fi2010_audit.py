"""Synthetic end-to-end tests for FI-2010 manifests and audits."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import jsonschema
import pytest
import yaml

from deepbook.data.fi2010.acquisition import ArchiveLimits, safe_extract_zip
from deepbook.data.fi2010.audit import AuditError, build_audit, write_audit_outputs


def _matrix_text(observations: int = 3) -> str:
    rows = [
        [float((row + 1) * 100 + column) for column in range(observations)] for row in range(144)
    ]
    rows.extend([[float((column % 3) + 1) for column in range(observations)] for _ in range(5)])
    return "\n".join(" ".join(map(str, row)) for row in rows) + "\n"


def _archive(path: Path) -> None:
    matrix = _matrix_text().encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for fold in range(1, 10):
            directory = "BenchmarkDatasets/NoAuction/ZScore"
            archive.writestr(f"{directory}/Train_Dst_NoAuction_ZScore_CF_{fold}.txt", matrix)
            archive.writestr(f"{directory}/Test_Dst_NoAuction_ZScore_CF_{fold}.txt", matrix)
        archive.writestr("BenchmarkDatasets/README.txt", b"synthetic test fixture")


def _config(path: Path, archive: Path) -> None:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    config = {
        "schema_version": 1,
        "dataset": {
            "name": "FI-2010",
            "title": "Synthetic test fixture",
            "persistent_identifier": "urn:test:fi2010",
            "landing_page_identifier": "test-record",
            "access_type": "synthetic test only",
            "license": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
            "rights_holder": "Test",
        },
        "source": {
            "archive_filename": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": digest,
            "checksum_provenance": "locally-computed from authoritative Fairdata archive",
        },
        "selection": {"benchmark_variant": "no_auction", "normalization": "zscore"},
        "published_expectations": {"fold_count": 9, "selected_matrix_count": 18},
        "archive_limits": {
            "maximum_archive_bytes": 2_000_000,
            "maximum_extracted_bytes": 2_000_000,
            "maximum_member_bytes": 100_000,
            "maximum_member_count": 100,
            "maximum_compression_ratio": 100,
        },
    }
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _schema(repository_root: Path, name: str) -> dict:
    return json.loads((repository_root / "data_contracts" / name).read_text(encoding="utf-8"))


def test_audit_is_deterministic_and_manifests_match_schemas(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    archive = workspace / "fixture.zip"
    config = workspace / "configs" / "fi2010.yaml"
    extraction = workspace / "extracted"
    _archive(archive)
    _config(config, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    limits = ArchiveLimits(2_000_000, 2_000_000, 100_000, 100, 100)
    safe_extract_zip(archive, extraction, limits, archive_sha256=digest)
    generated = "2026-07-31T00:00:00Z"

    first = build_audit(
        repository_root=workspace,
        config_path=config,
        archive_path=archive,
        extraction_root=extraction,
        generated_utc=generated,
    )
    second = build_audit(
        repository_root=workspace,
        config_path=config,
        archive_path=archive,
        extraction_root=extraction,
        generated_utc=generated,
    )

    assert first == second
    audit, source, splits = first
    assert audit["status"] == "PASS"
    assert audit["data_fingerprint"] == second[0]["data_fingerprint"]
    assert audit["results"]["observation_counts"]["protocol_unique_representations"] == 30
    assert audit["results"]["label_distributions"]["overall_file_rows"]["10"]["counts"] == {
        "1": 18,
        "2": 18,
        "3": 18,
    }
    assert len(splits["splits"]) == 18
    generated_provenance = json.dumps(source).lower()
    tracked_provenance = (
        (repository_root / "reports" / "protocol" / "fi2010_data_provenance.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    unsupported_phrases = (
        "published " + "sha-256",
        "checksum is " + "published by fairdata",
        "checksum is " + "published by the fairdata metax",
        "source-" + "published checksum",
    )
    assert all(phrase not in generated_provenance for phrase in unsupported_phrases)
    assert all(phrase not in tracked_provenance for phrase in unsupported_phrases)
    assert source["checksum_provenance"] == "locally-computed from authoritative Fairdata archive"
    assert "locally computed sha-256 of the authoritative fairdata archive" in tracked_provenance
    assert "no per-file sha-256" in tracked_provenance
    assert any(
        "computed locally from the archive downloaded from the authoritative fairdata source"
        in note.lower()
        for note in source["notes"]
    )

    jsonschema.Draft202012Validator(
        _schema(repository_root, "fi2010_source_manifest.schema.json"),
        format_checker=jsonschema.FormatChecker(),
    ).validate(source)
    jsonschema.Draft202012Validator(
        _schema(repository_root, "fi2010_split_manifest.schema.json")
    ).validate(splits)

    outputs = write_audit_outputs(
        audit,
        source,
        splits,
        interim_root=workspace / "interim",
        report_root=workspace / "reports",
    )
    initial_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in outputs.items()
    }
    write_audit_outputs(
        *second,
        interim_root=workspace / "interim",
        report_root=workspace / "reports",
    )
    assert initial_hashes == {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in outputs.items()
    }


def test_audit_rejects_tampered_extraction(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    archive = workspace / "fixture.zip"
    config = workspace / "configs" / "fi2010.yaml"
    extraction = workspace / "extracted"
    _archive(archive)
    _config(config, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    limits = ArchiveLimits(2_000_000, 2_000_000, 100_000, 100, 100)
    safe_extract_zip(archive, extraction, limits, archive_sha256=digest)
    matrix = next(extraction.rglob("Train_*.txt"))
    matrix.write_bytes(b"tampered")

    with pytest.raises(AuditError, match="failed validation"):
        build_audit(
            repository_root=workspace,
            config_path=config,
            archive_path=archive,
            extraction_root=extraction,
            generated_utc="2026-07-31T00:00:00Z",
        )
