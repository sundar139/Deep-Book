"""Focused tests for FI-2010 download and archive safety."""

from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from pathlib import Path

import pytest

from deepbook.data.fi2010.acquisition import (
    AcquisitionError,
    ArchiveLimits,
    download_archive,
    inspect_zip,
    redact_url,
    safe_extract_zip,
)

_DOWNLOAD_HOST = "download" + ".fairdata.fi"


def _temporary_url() -> str:
    return f"https://{_DOWNLOAD_HOST}/download?token=redacted"


def _limits(**overrides) -> ArchiveLimits:
    values = {
        "maximum_archive_bytes": 1_000_000,
        "maximum_extracted_bytes": 1_000_000,
        "maximum_member_bytes": 100_000,
        "maximum_member_count": 20,
        "maximum_compression_ratio": 100,
    }
    values.update(overrides)
    return ArchiveLimits(**values)


def _zip(path: Path, members: list[tuple[str | zipfile.ZipInfo, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members:
            archive.writestr(name, content)


def _zip_bytes() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("matrix.txt", b"1 2 3\n")
    return output.getvalue()


def test_url_redaction_removes_query_and_fragment() -> None:
    value = _temporary_url() + "#fragment"
    assert redact_url(value) == f"https://{_DOWNLOAD_HOST}/download"
    assert "secret" not in redact_url(value)


def test_valid_zip_extracts_and_is_idempotent(tmp_path: Path) -> None:
    archive = tmp_path / "valid.zip"
    _zip(archive, [("root/matrix.txt", b"1 2 3\n")])
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    inventory = inspect_zip(archive, _limits())
    first = safe_extract_zip(archive, tmp_path / "out", _limits(), archive_sha256=digest)
    second = safe_extract_zip(archive, tmp_path / "out", _limits(), archive_sha256=digest)

    assert inventory.file_count == 1
    assert first.preserved_existing is False
    assert second.preserved_existing is True
    assert (tmp_path / "out" / "root" / "matrix.txt").read_bytes() == b"1 2 3\n"


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "C:\\drive.txt"])
def test_unsafe_member_paths_are_rejected(tmp_path: Path, member: str) -> None:
    archive = tmp_path / "unsafe.zip"
    _zip(archive, [(member, b"x")])
    with pytest.raises(AcquisitionError, match="path|traverses|drive"):
        inspect_zip(archive, _limits())


def test_duplicate_normalized_target_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.zip"
    _zip(archive, [("a//b.txt", b"one"), ("a/b.txt", b"two")])
    with pytest.raises(AcquisitionError, match="duplicate normalized"):
        inspect_zip(archive, _limits())


def test_oversized_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "large.zip"
    _zip(archive, [("large.txt", b"12345")])
    with pytest.raises(AcquisitionError, match="per-file"):
        inspect_zip(archive, _limits(maximum_member_bytes=4))


def test_excessive_member_count_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "many.zip"
    _zip(archive, [(f"{index}.txt", b"x") for index in range(3)])
    with pytest.raises(AcquisitionError, match="member-count"):
        inspect_zip(archive, _limits(maximum_member_count=2))


def test_non_zip_content_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "fake.zip"
    archive.write_text("<html>not an archive</html>", encoding="utf-8")
    with pytest.raises(AcquisitionError, match="archive type"):
        inspect_zip(archive, _limits())


def test_symbolic_link_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    _zip(archive, [(info, b"target")])
    with pytest.raises(AcquisitionError, match="symbolic link"):
        inspect_zip(archive, _limits())


class _Headers:
    def __init__(self, size: int) -> None:
        self.size = size

    def get_content_type(self) -> str:
        return "application/octet-stream"

    def get(self, name: str) -> str | None:
        return str(self.size) if name == "Content-Length" else None


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = io.BytesIO(content)
        self.headers = _Headers(len(content))

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.content.read(size)


def test_download_is_atomic_and_preserves_valid_archive(tmp_path: Path, monkeypatch) -> None:
    content = _zip_bytes()
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        "deepbook.data.fi2010.acquisition.urllib.request.urlopen",
        lambda request, timeout: _Response(content),
    )
    destination = tmp_path / "archive.zip"

    first = download_archive(
        _temporary_url(),
        destination,
        expected_size=len(content),
        expected_sha256=digest,
        maximum_archive_bytes=10_000,
        timeout=1,
    )
    second = download_archive(
        _temporary_url(),
        destination,
        expected_size=len(content),
        expected_sha256=digest,
        maximum_archive_bytes=10_000,
        timeout=1,
    )

    assert first.preserved_existing is False
    assert second.preserved_existing is True
    assert not destination.with_name("archive.zip.part").exists()


def test_checksum_mismatch_removes_partial_file(tmp_path: Path, monkeypatch) -> None:
    content = _zip_bytes()
    monkeypatch.setattr(
        "deepbook.data.fi2010.acquisition.urllib.request.urlopen",
        lambda request, timeout: _Response(content),
    )
    destination = tmp_path / "archive.zip"

    with pytest.raises(AcquisitionError, match="SHA-256"):
        download_archive(
            _temporary_url(),
            destination,
            expected_size=len(content),
            expected_sha256="0" * 64,
            maximum_archive_bytes=10_000,
            timeout=1,
        )

    assert not destination.exists()
    assert not destination.with_name("archive.zip.part").exists()
