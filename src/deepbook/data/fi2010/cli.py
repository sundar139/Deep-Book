"""Command-line interface for authoritative FI-2010 acquisition and audit."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepbook.data.fi2010.acquisition import (
    AcquisitionError,
    AcquisitionResult,
    ArchiveLimits,
    authorize_download,
    download_archive,
    inspect_zip,
    load_config,
    redact_url,
    safe_extract_zip,
    sha256_file,
)
from deepbook.data.fi2010.audit import AuditError, build_audit, write_audit_outputs

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _inside_repository(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise AcquisitionError(f"destination must remain inside the repository: {resolved}")
    return resolved


def _configured_path(root: Path, value: str) -> Path:
    path = Path(value)
    return _inside_repository(path if path.is_absolute() else root / path, root)


def _archive_limits(config: dict[str, Any]) -> ArchiveLimits:
    values = config["archive_limits"]
    return ArchiveLimits(
        maximum_archive_bytes=int(values["maximum_archive_bytes"]),
        maximum_extracted_bytes=int(values["maximum_extracted_bytes"]),
        maximum_member_bytes=int(values["maximum_member_bytes"]),
        maximum_member_count=int(values["maximum_member_count"]),
        maximum_compression_ratio=float(values["maximum_compression_ratio"]),
    )


def _paths(
    root: Path,
    config: dict[str, Any],
    expected_sha256: str,
    destination_root: Path | None,
) -> tuple[Path, Path, Path, Path]:
    local = config["local_paths"]
    raw_root = (
        _inside_repository(destination_root, root)
        if destination_root is not None
        else _configured_path(root, str(local["raw_root"]))
    )
    interim_root = _configured_path(root, str(local["interim_root"]))
    report_root = _configured_path(root, str(local["report_root"]))
    archive_name = str(config["source"]["archive_filename"])
    archive_path = raw_root / "archive" / expected_sha256 / archive_name
    extraction_root = raw_root / "extracted" / expected_sha256
    return archive_path, extraction_root, interim_root, report_root


def _record_acquisition(
    path: Path,
    config: dict[str, Any],
    result: AcquisitionResult,
    extraction_bytes: int,
    member_count: int,
    checksum_provenance: str,
) -> None:
    value = {
        "schema_version": 1,
        "dataset_name": config["dataset"]["name"],
        "persistent_identifier": config["dataset"]["persistent_identifier"],
        "accessed_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "archive_filename": result.archive_path.name,
        "archive_size_bytes": result.size_bytes,
        "archive_sha256": result.sha256,
        "checksum_provenance": checksum_provenance,
        "archive_member_count": member_count,
        "extracted_byte_size": extraction_bytes,
        "status": "validated",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AcquisitionError(f"existing acquisition record is invalid: {path}") from error
        identity = ("archive_size_bytes", "archive_sha256", "archive_member_count")
        if any(existing.get(key) != value[key] for key in identity):
            raise AcquisitionError(
                f"existing acquisition record conflicts with the archive: {path}"
            )
        return
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _acquire(args: argparse.Namespace) -> int:
    root = _repository_root()
    config_path = _inside_repository(args.config, root)
    config = load_config(config_path)
    source = config["source"]
    published_sha256 = str(source["archive_sha256"]).lower()
    expected_sha256 = (args.expected_sha256 or published_sha256).lower()
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise AcquisitionError("expected SHA-256 must be exactly 64 hexadecimal characters")
    checksum_provenance = (
        str(source["checksum_provenance"])
        if expected_sha256 == published_sha256
        else "runtime-supplied expected checksum"
    )
    archive_path, extraction_root, interim_root, _ = _paths(
        root, config, expected_sha256, args.destination_root
    )
    expected_size = int(source["archive_size_bytes"])
    timeout = int(source["timeout_seconds"])
    limits = _archive_limits(config)

    if args.offline:
        if not archive_path.is_file():
            raise AcquisitionError(f"offline archive is unavailable: {archive_path}")
        size = archive_path.stat().st_size
        digest = sha256_file(archive_path)
        if size != expected_size or digest != expected_sha256:
            raise AcquisitionError("offline archive does not match the expected size and SHA-256")
        result = AcquisitionResult(archive_path, size, digest, True)
        print(f"Archive: preserved validated local file {archive_path}")
    else:
        if args.source_url:
            url = args.source_url
            print(f"Source: runtime override {redact_url(url)}")
        else:
            url = authorize_download(
                str(source["authorization_url"]),
                str(config["dataset"]["landing_page_identifier"]),
                str(source["authoritative_file_path"]),
                timeout,
            )
            print("Source: authoritative Fairdata temporary download authorization")
        result = download_archive(
            url,
            archive_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            maximum_archive_bytes=limits.maximum_archive_bytes,
            timeout=timeout,
            force=args.force,
        )
        behavior = "preserved validated local file" if result.preserved_existing else "downloaded"
        print(f"Archive: {behavior} {archive_path}")

    inventory = inspect_zip(archive_path, limits)
    extraction = safe_extract_zip(
        archive_path,
        extraction_root,
        limits,
        archive_sha256=expected_sha256,
        force=args.force,
    )
    extraction_behavior = (
        "preserved validated tree" if extraction.preserved_existing else "extracted"
    )
    print(f"Extraction: {extraction_behavior} {extraction_root}")
    print(f"Archive bytes: {result.size_bytes}")
    print(f"Archive SHA-256 ({checksum_provenance}): {result.sha256}")
    print(
        f"Archive inventory: {inventory.member_count} members, {inventory.file_count} files, "
        f"{inventory.uncompressed_bytes} uncompressed bytes"
    )
    record_path = interim_root / "acquisitions" / f"{expected_sha256}.json"
    _record_acquisition(
        record_path,
        config,
        result,
        extraction.extracted_bytes,
        inventory.member_count,
        checksum_provenance,
    )
    print(f"Acquisition record: {record_path}")
    return 0


def _acquisition_timestamp(interim_root: Path, expected_sha256: str) -> str:
    record_path = interim_root / "acquisitions" / f"{expected_sha256}.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"validated acquisition record is unavailable: {record_path}") from error
    if record.get("archive_sha256") != expected_sha256:
        raise AuditError("acquisition record does not match the configured archive")
    timestamp = record.get("accessed_utc")
    if not isinstance(timestamp, str) or not timestamp:
        raise AuditError("acquisition record has no valid access timestamp")
    return timestamp


def _audit(args: argparse.Namespace) -> int:
    root = _repository_root()
    config_path = _inside_repository(args.config, root)
    config = load_config(config_path)
    expected_sha256 = str(config["source"]["archive_sha256"]).lower()
    archive_path, extraction_root, interim_root, report_root = _paths(
        root, config, expected_sha256, args.destination_root
    )
    audit, source_manifest, split_manifest = build_audit(
        repository_root=root,
        config_path=config_path,
        archive_path=archive_path,
        extraction_root=extraction_root,
        generated_utc=_acquisition_timestamp(interim_root, expected_sha256),
    )
    paths = write_audit_outputs(
        audit,
        source_manifest,
        split_manifest,
        interim_root=interim_root,
        report_root=report_root,
    )
    print(f"Audit status: {audit['status']}")
    print(f"Data fingerprint: {audit['data_fingerprint']}")
    print(
        "Unique processed representations: "
        f"{audit['results']['observation_counts']['protocol_unique_representations']}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    root = _repository_root()
    parser = argparse.ArgumentParser(description="Acquire and audit the authoritative FI-2010 data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire = subparsers.add_parser("acquire", help="download, validate, and safely extract data")
    acquire.add_argument("--config", type=Path, default=root / "configs" / "data" / "fi2010.yaml")
    acquire.add_argument("--destination-root", type=Path)
    acquire.add_argument("--source-url", help="runtime-only source URL override")
    acquire.add_argument("--expected-sha256", help="runtime-only expected archive checksum")
    acquire.add_argument("--offline", action="store_true", help="forbid network access")
    acquire.add_argument("--force", action="store_true", help="replace invalid local content")
    acquire.set_defaults(handler=_acquire)

    audit = subparsers.add_parser("audit", help="audit published files and supplied labels")
    audit.add_argument("--config", type=Path, default=root / "configs" / "data" / "fi2010.yaml")
    audit.add_argument("--destination-root", type=Path)
    audit.set_defaults(handler=_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the FI-2010 command-line interface."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (AcquisitionError, AuditError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
