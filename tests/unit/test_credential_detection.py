"""Tests for tracked-content policy detection."""

from __future__ import annotations

import json

import pytest

from deepbook.repository_policy import (
    TEMPORARY_DOWNLOAD_URL_PATTERN,
    find_credential_violations,
    find_scientific_import_violations,
    find_stage_terms,
)


@pytest.mark.parametrize(
    ("label", "separator", "number"),
    [
        ("phase", "", "0"),
        ("phase", "_", "0"),
        ("phase", "-", "0"),
        ("phase", " ", "0"),
        ("phase", "", "1"),
        ("phase", "_", "1"),
        ("phase", "-", "1"),
        ("phase", " ", "1"),
        ("step", "", "0"),
        ("step", "_", "0"),
        ("step", "-", "0"),
        ("step", " ", "0"),
        ("milestone", "", "0"),
        ("milestone", "_", "0"),
        ("milestone", "-", "0"),
        ("milestone", " ", "0"),
    ],
)
def test_stage_label_variants_detected(label: str, separator: str, number: str) -> None:
    text = f"{label}{separator}{number}"

    assert find_stage_terms(text) == [text]


@pytest.mark.parametrize(
    "text",
    [
        "phased rollout",
        "stepwise",
        "milestones",
        "phase transition",
        "the phase of the moon",
    ],
)
def test_professional_text_does_not_trigger_stage_policy(text: str) -> None:
    assert find_stage_terms(text) == []


def test_empty_env_assignment_passes() -> None:
    assert find_credential_violations("DATABENTO_API_KEY=") == []


@pytest.mark.parametrize(
    "value",
    ["${DATABENTO_API_KEY}", "<YOUR_KEY>", "changeme"],
)
def test_template_placeholders_pass(value: str) -> None:
    assert find_credential_violations(f'DATABENTO_API_KEY="{value}"') == []


def test_nonempty_databento_key_fails() -> None:
    text = f"DATABENTO_API_KEY={'real-looking-value'}"

    assert find_credential_violations(text) == ["DATABENTO_API_KEY"]


def test_generic_api_key_fails() -> None:
    text = f"API_KEY={'generic-real-looking-value'}"

    assert find_credential_violations(text) == ["API_KEY"]


def test_json_credential_fails() -> None:
    text = json.dumps({"API_KEY": "json-real-looking-value"})

    assert find_credential_violations(text) == ["API_KEY"]


def test_pem_private_key_header_fails() -> None:
    text = "-----BEGIN " + "RSA PRIVATE KEY-----"

    assert find_credential_violations(text) == ["PRIVATE_KEY"]


def test_unrelated_word_key_passes() -> None:
    assert find_credential_violations("This keyboard key is unrelated.") == []


def test_findings_are_redacted() -> None:
    value = "never-print-this-value"
    text = f"COINBASE_API_SECRET={value}"

    findings = [f"{name}=<redacted>" for name in find_credential_violations(text)]

    assert findings == ["COINBASE_API_SECRET=<redacted>"]
    assert value not in " ".join(findings)


def test_temporary_fairdata_url_is_detected_without_storing_a_token() -> None:
    host = "download" + ".fairdata.fi"
    runtime_url = f"https://{host}/download?token=" + "runtime-secret"
    assert TEMPORARY_DOWNLOAD_URL_PATTERN.search(runtime_url)


def test_numpy_is_allowed_in_modeling_paths() -> None:
    statement = "import " + "numpy as np"
    assert find_scientific_import_violations("src/deepbook/data/fi2010/dataset.py", statement) == []
    assert find_scientific_import_violations("src/deepbook/models/classical.py", statement) == []
    assert find_scientific_import_violations("src/deepbook/model.py", statement) == [
        "NumPy import outside FI-2010 data code"
    ]


def test_modeling_scientific_import_is_allowed_only_in_approved_paths() -> None:
    assert (
        find_scientific_import_violations("src/deepbook/models/classical.py", "import " + "sklearn")
        == []
    )
    assert find_scientific_import_violations(
        "src/deepbook/data/fi2010/dataset.py", "import " + "sklearn"
    ) == ["prohibited scientific import"]
