"""Real authoritative FI-2010 audit acceptance test."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from deepbook.data.fi2010.acquisition import load_config
from deepbook.data.fi2010.audit import build_audit

pytestmark = pytest.mark.data


def test_authoritative_archive_passes_complete_audit() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "data" / "fi2010.yaml"
    config = load_config(config_path)
    digest = str(config["source"]["archive_sha256"])
    raw_root = root / str(config["local_paths"]["raw_root"])
    archive = raw_root / "archive" / digest / str(config["source"]["archive_filename"])
    extraction = raw_root / "extracted" / digest
    record_path = (
        root / str(config["local_paths"]["interim_root"]) / "acquisitions" / f"{digest}.json"
    )
    if not archive.is_file() or not record_path.is_file():
        pytest.skip("authoritative FI-2010 archive is not locally acquired")
    record = json.loads(record_path.read_text(encoding="utf-8"))

    audit, source, splits = build_audit(
        repository_root=root,
        config_path=config_path,
        archive_path=archive,
        extraction_root=extraction,
        generated_utc=record["accessed_utc"],
    )

    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert source["archive_sha256"] == digest
    assert source["archive_size_bytes"] == int(config["source"]["archive_size_bytes"])
    assert len(splits["splits"]) == int(config["published_expectations"]["selected_matrix_count"])
    assert audit["results"]["observed_missing_tokens"] == 0
    assert audit["results"]["observed_nonfinite_tokens"] == 0
    assert audit["results"]["validated_nonfinite_values"] == 0
    for split in splits["splits"]:
        for counts in split["label_counts_by_horizon"].values():
            assert sum(counts.values()) == split["observation_count"]

    source_schema = json.loads(
        (root / "data_contracts" / "fi2010_source_manifest.schema.json").read_text(encoding="utf-8")
    )
    split_schema = json.loads(
        (root / "data_contracts" / "fi2010_split_manifest.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(
        source_schema, format_checker=jsonschema.FormatChecker()
    ).validate(source)
    jsonschema.Draft202012Validator(split_schema).validate(splits)
