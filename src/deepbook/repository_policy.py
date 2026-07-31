"""Repository policy checks for DeepBook."""

from __future__ import annotations

import json
import re
import subprocess
import urllib.parse
from pathlib import Path

from deepbook.registry import RegistryError, validate_ablations, validate_hypotheses

PROHIBITED_PATHS = (
    "src/deepbook/features/hawkes.py",
    "src/deepbook/eval/stats.py",
    "src/deepbook/validation/invariants.py",
)
PROHIBITED_IMPORT_PATTERN = (
    r"import (scipy|sklearn|torch|pandas|gymnasium|stable_baselines3)"
    r"|from (scipy|sklearn|torch|pandas|gymnasium|stable_baselines3)"
)
NUMPY_IMPORT_PATTERN = r"(?:import numpy|from numpy)"
_ALLOWED_NUMPY_PREFIX = "src/deepbook/data/fi2010/"
TEMPORARY_DOWNLOAD_URL_PATTERN = re.compile(
    r"https://download\.fairdata\.fi(?::[0-9]+)?/[^\s\"']*\?[^\s\"']+",
    re.IGNORECASE,
)
STAGE_PATTERN = re.compile(r"\b(?:phase|step|milestone)[\s_-]*[0-9]+\b", re.IGNORECASE)
CREDENTIAL_NAMES = (
    "DATABENTO_API_KEY",
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "COINBASE_API_KEY",
    "COINBASE_API_SECRET",
    "API_KEY",
    "API_SECRET",
    "ACCESS_TOKEN",
    "PRIVATE_KEY",
)
_CREDENTIAL_PATTERN = re.compile(
    rf"^\s*[\"']?(?P<name>{'|'.join(CREDENTIAL_NAMES)})[\"']?\s*[:=]\s*(?P<value>.*?)\s*,?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
PEM_HEADER_PATTERN = re.compile(r"-----BEGIN(?: [A-Z]+)* PRIVATE KEY-----", re.IGNORECASE)
_ALLOWED_PLACEHOLDERS = re.compile(r"^(?:\$\{[^}}]+\}|<[^>]+>|changeme)$", re.IGNORECASE)
_POLICY_SOURCE = "src/deepbook/repository_policy.py"
_BINARY_SUFFIXES = {".ckpt", ".jpg", ".lock", ".png", ".pt", ".pth"}


def find_scientific_import_violations(relative_path: str, text: str) -> list[str]:
    """Return import-policy findings for one tracked Python source file."""
    violations: list[str] = []
    if re.search(PROHIBITED_IMPORT_PATTERN, text):
        violations.append("prohibited scientific import")
    if re.search(NUMPY_IMPORT_PATTERN, text) and not relative_path.startswith(
        _ALLOWED_NUMPY_PREFIX
    ):
        violations.append("NumPy import outside FI-2010 data code")
    return violations


def find_stage_terms(text: str) -> list[str]:
    """Return prohibited implementation-stage labels found in text."""
    return [match.group(0) for match in STAGE_PATTERN.finditer(text)]


def _json_credential_names(value: object) -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                isinstance(key, str)
                and key.upper() in CREDENTIAL_NAMES
                and isinstance(child, str)
                and child
                and not _ALLOWED_PLACEHOLDERS.fullmatch(child)
            ):
                findings.append(key.upper())
            findings.extend(_json_credential_names(child))
    elif isinstance(value, list):
        for child in value:
            findings.extend(_json_credential_names(child))
    return findings


def find_credential_violations(text: str) -> list[str]:
    """Return redacted credential findings without exposing assigned values."""
    findings: list[str] = []
    for match in _CREDENTIAL_PATTERN.finditer(text):
        value = match.group("value").strip().strip("\"'").strip()
        if value and not _ALLOWED_PLACEHOLDERS.fullmatch(value):
            findings.append(match.group("name").upper())
    try:
        findings.extend(_json_credential_names(json.loads(text)))
    except (json.JSONDecodeError, TypeError):
        pass
    if PEM_HEADER_PATTERN.search(text):
        findings.append("PRIVATE_KEY")
    return list(dict.fromkeys(findings))


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )


def _tracked_files(root: Path) -> list[str]:
    result = _git(root, "ls-files", "-z")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return [path for path in result.stdout.split("\0") if path]


def _read_tracked_text(root: Path, relative_path: str) -> str | None:
    path = root / relative_path
    if not path.is_file() or path.suffix.lower() in _BINARY_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def check_repository(root: Path) -> list[str]:
    """Return redacted repository-policy violations for a Git checkout."""
    violations: list[str] = []

    if _git(root, "ls-files", "--error-unmatch", ".env").returncode == 0:
        violations.append(".env is tracked by git")
    if _git(root, "ls-files", "--error-unmatch", ".env.example").returncode != 0:
        violations.append(".env.example is not tracked by git")
    if _git(root, "check-ignore", ".env").returncode != 0:
        violations.append(".env is not ignored")
    for relative_path in ("data/raw/example.jsonl", "artifacts/test.txt"):
        if _git(root, "check-ignore", relative_path).returncode != 0:
            violations.append(f"{relative_path} is not ignored")

    for relative_path in PROHIBITED_PATHS:
        if (root / relative_path).exists():
            violations.append(f"prohibited implementation file exists: {relative_path}")

    tracked = _tracked_files(root)
    for relative_path in tracked:
        if relative_path.startswith(("data/raw/fi2010/", "data/interim/fi2010/")):
            violations.append(f"raw or generated FI-2010 data is tracked: {relative_path}")
        text = _read_tracked_text(root, relative_path)
        if text is None:
            continue
        if relative_path.endswith(".py") and relative_path != _POLICY_SOURCE:
            for finding in find_scientific_import_violations(relative_path, text):
                violations.append(f"{finding} in {relative_path}")
        if relative_path != _POLICY_SOURCE:
            for term in find_stage_terms(text):
                violations.append(f"stage terminology in {relative_path}: {term!r}")
        if TEMPORARY_DOWNLOAD_URL_PATTERN.search(text):
            violations.append(f"temporary Fairdata download URL in {relative_path}")
        for name in find_credential_violations(text):
            violations.append(f"credential in {relative_path}: {name}=<redacted>")

    import yaml

    policy = yaml.safe_load((root / "configs" / "spending_policy.yaml").read_text())
    if policy["paid_data"]["default_authorization"] is not False:
        violations.append("paid-data authorization does not default to false")
    if policy["core"]["research_budget_usd"] != 0:
        violations.append("research budget is not zero")

    fi2010_config_path = root / "configs" / "data" / "fi2010.yaml"
    if fi2010_config_path.exists():
        fi2010_config = yaml.safe_load(fi2010_config_path.read_text())
        allowed_source_hosts = {"etsin.fairdata.fi", "metax.fairdata.fi"}
        for key in ("metadata_url", "landing_page_url", "authorization_url"):
            url = fi2010_config["source"][key]
            if urllib.parse.urlsplit(url).hostname not in allowed_source_hosts:
                violations.append(f"unofficial FI-2010 source configured in {key}")

    try:
        validate_hypotheses(yaml.safe_load((root / "configs" / "hypotheses.yaml").read_text()))
        validate_ablations(yaml.safe_load((root / "configs" / "ablations.yaml").read_text()))
    except RegistryError as error:
        violations.append(f"registry validation failed: {error}")

    return violations


def main(root: Path | None = None) -> int:
    """Run repository checks and return a process exit code."""
    checkout = root or Path(__file__).resolve().parents[2]
    try:
        violations = check_repository(checkout)
    except (OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"FAIL: repository policy could not run: {error}")
        return 1

    if violations:
        for violation in violations:
            print(f"FAIL: {violation}")
        print(f"\n{len(violations)} policy check(s) FAILED")
        return 1

    print("All repository policy checks PASSED")
    return 0
