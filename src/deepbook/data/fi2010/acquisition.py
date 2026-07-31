"""Authoritative FI-2010 acquisition and safe ZIP extraction."""

from __future__ import annotations

import hashlib
import json
import ntpath
import os
import posixpath
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_CHUNK_SIZE = 1024 * 1024
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


class AcquisitionError(RuntimeError):
    """Raised when source acquisition or archive validation fails."""


@dataclass(frozen=True)
class ArchiveLimits:
    """Upper bounds used before archive extraction."""

    maximum_archive_bytes: int
    maximum_extracted_bytes: int
    maximum_member_bytes: int
    maximum_member_count: int
    maximum_compression_ratio: float


@dataclass(frozen=True)
class ArchiveMember:
    """Validated ZIP member metadata."""

    path: str
    compressed_bytes: int
    uncompressed_bytes: int
    is_directory: bool


@dataclass(frozen=True)
class ArchiveInventory:
    """Validated archive inventory."""

    archive_type: str
    member_count: int
    file_count: int
    uncompressed_bytes: int
    members: tuple[ArchiveMember, ...]


@dataclass(frozen=True)
class AcquisitionResult:
    """Local archive identity and acquisition behavior."""

    archive_path: Path
    size_bytes: int
    sha256: str
    preserved_existing: bool


@dataclass(frozen=True)
class ExtractionResult:
    """Validated extracted-tree identity."""

    extraction_root: Path
    member_count: int
    file_count: int
    extracted_bytes: int
    preserved_existing: bool
    files: tuple[dict[str, Any], ...]


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from *path*."""
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AcquisitionError(f"cannot load configuration {path}: {error}") from error
    if not isinstance(data, dict):
        raise AcquisitionError("configuration root must be a mapping")
    return data


def config_sha256(path: Path) -> str:
    """Return the SHA-256 of a configuration file."""
    return sha256_file(path)


def redact_url(url: str) -> str:
    """Remove query strings and fragments from a URL used in messages."""
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(request: urllib.request.Request, timeout: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(1024 * 1024)
    except urllib.error.HTTPError as error:
        raise AcquisitionError(f"source authorization failed with HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise AcquisitionError("source authorization failed") from error
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AcquisitionError("source authorization returned invalid JSON") from error
    if not isinstance(value, dict):
        raise AcquisitionError("source authorization returned a non-object response")
    return value


def authorize_download(
    authorization_url: str,
    record_id: str,
    source_path: str,
    timeout: int,
) -> str:
    """Resolve a temporary Fairdata download URL without logging it."""
    body = json.dumps({"cr_id": record_id, "file": source_path}).encode()
    request = urllib.request.Request(
        authorization_url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "DeepBook-FI2010/1"},
        method="POST",
    )
    response = _request_json(request, timeout)
    url = response.get("url")
    if not isinstance(url, str) or not url:
        raise AcquisitionError("source authorization response omitted the download URL")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "download.fairdata.fi":
        raise AcquisitionError("source authorization returned an unexpected download host")
    return url


def _valid_existing_archive(path: Path, expected_size: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and sha256_file(path) == expected_sha256
    )


def download_archive(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    maximum_archive_bytes: int,
    timeout: int,
    force: bool = False,
) -> AcquisitionResult:
    """Stream an archive to a partial file and atomically promote it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _valid_existing_archive(destination, expected_size, expected_sha256):
            return AcquisitionResult(destination, expected_size, expected_sha256, True)
        if not force:
            raise AcquisitionError(
                f"existing archive differs from the expected source: {destination}; use --force"
            )

    partial = destination.with_name(f"{destination.name}.part")
    if partial.exists():
        if not force:
            raise AcquisitionError(f"partial download already exists: {partial}; use --force")
        partial.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "DeepBook-FI2010/1"})
    digest = hashlib.sha256()
    size = 0
    first = b""
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            partial.open("xb") as output,
        ):
            content_type = response.headers.get_content_type().lower()
            content_length = response.headers.get("Content-Length")
            if content_type in {"text/html", "application/xhtml+xml"}:
                raise AcquisitionError("download endpoint returned HTML instead of an archive")
            if content_length is not None:
                declared = int(content_length)
                if declared != expected_size:
                    raise AcquisitionError(
                        f"download content length {declared} differs from expected {expected_size}"
                    )
                if declared > maximum_archive_bytes:
                    raise AcquisitionError("download exceeds the configured archive-size limit")
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                if not first:
                    first = chunk[:512]
                    if first[:4] not in _ZIP_MAGIC:
                        lowered = first.lstrip().lower()
                        if lowered.startswith((b"<!doctype html", b"<html")):
                            raise AcquisitionError(
                                "download endpoint returned HTML instead of an archive"
                            )
                        raise AcquisitionError("download content does not have a ZIP signature")
                size += len(chunk)
                if size > maximum_archive_bytes:
                    raise AcquisitionError("download exceeds the configured archive-size limit")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except AcquisitionError:
        partial.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as error:
        partial.unlink(missing_ok=True)
        raise AcquisitionError(
            f"download failed with HTTP {error.code} from {redact_url(url)}"
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        partial.unlink(missing_ok=True)
        raise AcquisitionError(f"download failed from {redact_url(url)}") from error

    observed_sha256 = digest.hexdigest()
    if size != expected_size:
        partial.unlink(missing_ok=True)
        raise AcquisitionError(f"download size {size} differs from expected {expected_size}")
    if observed_sha256 != expected_sha256:
        partial.unlink(missing_ok=True)
        raise AcquisitionError("download SHA-256 differs from the expected checksum")
    if not zipfile.is_zipfile(partial):
        partial.unlink(missing_ok=True)
        raise AcquisitionError("downloaded content is not a valid ZIP archive")

    if destination.exists():
        destination.unlink()
    partial.replace(destination)
    return AcquisitionResult(destination, size, observed_sha256, False)


def _normalized_member_path(name: str) -> str:
    if not name or "\x00" in name:
        raise AcquisitionError("archive contains an empty or NUL-containing member path")
    portable = name.replace("\\", "/")
    drive, _ = ntpath.splitdrive(portable)
    if drive or portable.startswith("/"):
        raise AcquisitionError(f"archive member has an absolute or drive-qualified path: {name!r}")
    path = PurePosixPath(portable)
    if ".." in path.parts:
        raise AcquisitionError(f"archive member traverses outside the destination: {name!r}")
    normalized = posixpath.normpath(portable).rstrip("/")
    if normalized in {"", "."} or normalized.startswith("../"):
        raise AcquisitionError(f"archive member has an unsafe path: {name!r}")
    return normalized


def inspect_zip(path: Path, limits: ArchiveLimits) -> ArchiveInventory:
    """Validate ZIP metadata and return its complete member inventory."""
    if path.suffix.lower() != ".zip" or not zipfile.is_zipfile(path):
        raise AcquisitionError("archive type is not a ZIP matching its .zip extension")
    if path.stat().st_size > limits.maximum_archive_bytes:
        raise AcquisitionError("archive exceeds the configured archive-size limit")

    members: list[ArchiveMember] = []
    normalized_seen: dict[str, str] = {}
    file_paths: set[str] = set()
    total = 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > limits.maximum_member_count:
            raise AcquisitionError("archive exceeds the configured member-count limit")
        for info in infos:
            normalized = _normalized_member_path(info.filename)
            collision_key = normalized.casefold()
            if collision_key in normalized_seen:
                original = normalized_seen[collision_key]
                raise AcquisitionError(
                    f"archive contains duplicate normalized paths: {original!r} "
                    f"and {info.filename!r}"
                )
            normalized_seen[collision_key] = info.filename

            mode = info.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if stat.S_ISLNK(mode):
                raise AcquisitionError(f"archive contains a symbolic link: {info.filename!r}")
            if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise AcquisitionError(
                    f"archive contains an unsupported member type: {info.filename!r}"
                )
            if info.flag_bits & 0x1:
                raise AcquisitionError(f"archive contains an encrypted member: {info.filename!r}")
            is_directory = info.is_dir() or stat.S_ISDIR(mode)
            if not is_directory:
                if info.file_size > limits.maximum_member_bytes:
                    raise AcquisitionError(
                        f"archive member exceeds the per-file limit: {info.filename!r}"
                    )
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > limits.maximum_compression_ratio:
                    raise AcquisitionError(
                        f"archive member has a suspicious compression ratio: {info.filename!r}"
                    )
                total += info.file_size
                file_paths.add(normalized.casefold())
            members.append(
                ArchiveMember(
                    path=normalized,
                    compressed_bytes=info.compress_size,
                    uncompressed_bytes=info.file_size,
                    is_directory=is_directory,
                )
            )
    if total > limits.maximum_extracted_bytes:
        raise AcquisitionError("archive exceeds the configured extracted-size limit")
    for file_path in file_paths:
        parts = file_path.split("/")
        for length in range(1, len(parts)):
            if "/".join(parts[:length]) in file_paths:
                raise AcquisitionError("archive contains conflicting file and directory paths")
    return ArchiveInventory(
        archive_type="zip",
        member_count=len(members),
        file_count=sum(not member.is_directory for member in members),
        uncompressed_bytes=total,
        members=tuple(members),
    )


def verify_extraction_manifest(
    extraction_root: Path, archive_sha256: str
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Validate every file recorded for an existing extracted tree."""
    manifest_path = extraction_root / ".deepbook-extraction.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcquisitionError("existing extraction has no valid extraction manifest") from error
    if not isinstance(manifest, dict) or manifest.get("archive_sha256") != archive_sha256:
        raise AcquisitionError("existing extraction belongs to a different archive")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise AcquisitionError("existing extraction manifest has an invalid file inventory")
    files: list[dict[str, Any]] = []
    for item in raw_files:
        if not isinstance(item, dict):
            raise AcquisitionError("existing extraction manifest has an invalid file entry")
        relative = item.get("path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(size, int)
            or not isinstance(digest, str)
        ):
            raise AcquisitionError("existing extraction manifest has an invalid file entry")
        path = extraction_root / Path(relative)
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise AcquisitionError(f"existing extracted file failed validation: {relative}")
        files.append(item)
    return manifest, tuple(files)


def safe_extract_zip(
    archive_path: Path,
    extraction_root: Path,
    limits: ArchiveLimits,
    *,
    archive_sha256: str,
    force: bool = False,
) -> ExtractionResult:
    """Safely extract a validated ZIP into an atomically promoted directory."""
    inventory = inspect_zip(archive_path, limits)
    if extraction_root.exists():
        try:
            manifest, files = verify_extraction_manifest(extraction_root, archive_sha256)
        except AcquisitionError:
            if not force:
                raise
        else:
            return ExtractionResult(
                extraction_root=extraction_root,
                member_count=inventory.member_count,
                file_count=len(files),
                extracted_bytes=int(manifest["extracted_bytes"]),
                preserved_existing=True,
                files=files,
            )

    extraction_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{extraction_root.name}.", dir=extraction_root.parent)
    )
    extracted_files: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            info_by_path = {
                _normalized_member_path(info.filename): info for info in archive.infolist()
            }
            for member in inventory.members:
                target = temporary / Path(member.path)
                if member.is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with archive.open(info_by_path[member.path]) as source, target.open("xb") as output:
                    while True:
                        chunk = source.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        size += len(chunk)
                        digest.update(chunk)
                        output.write(chunk)
                if size != member.uncompressed_bytes:
                    raise AcquisitionError(f"extracted size mismatch for {member.path}")
                extracted_files.append(
                    {"path": member.path, "size_bytes": size, "sha256": digest.hexdigest()}
                )
        extracted_files.sort(key=lambda item: str(item["path"]))
        manifest = {
            "schema_version": 1,
            "archive_sha256": archive_sha256,
            "archive_member_count": inventory.member_count,
            "file_count": len(extracted_files),
            "extracted_bytes": sum(int(item["size_bytes"]) for item in extracted_files),
            "files": extracted_files,
        }
        (temporary / ".deepbook-extraction.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if extraction_root.exists():
            shutil.rmtree(extraction_root)
        temporary.replace(extraction_root)
    except (AcquisitionError, OSError, zipfile.BadZipFile) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(error, AcquisitionError):
            raise
        raise AcquisitionError(f"archive extraction failed: {error}") from error

    return ExtractionResult(
        extraction_root=extraction_root,
        member_count=inventory.member_count,
        file_count=len(extracted_files),
        extracted_bytes=sum(int(item["size_bytes"]) for item in extracted_files),
        preserved_existing=False,
        files=tuple(extracted_files),
    )


def archive_inventory_json(inventory: ArchiveInventory) -> dict[str, Any]:
    """Convert an archive inventory to a JSON-ready mapping."""
    value = asdict(inventory)
    value["members"] = [asdict(member) for member in inventory.members]
    return value
