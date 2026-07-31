"""Registry validation for hypothesis and ablation YAML files.

Accepts already-parsed YAML data (dict) and validates the current repository
contract. Raises ValueError on malformed data.
"""

from __future__ import annotations

from typing import Any


class RegistryError(ValueError):
    """Raised when a registry document violates the contract."""


HYPOTHESIS_REQUIRED_FIELDS = [
    "id",
    "description",
    "outcome_metrics",
    "comparison_groups",
    "datasets",
    "required_controls",
    "falsification",
    "status",
]

HYPOTHESIS_ALLOWED_STATUSES = {
    "planned",
    "running",
    "confirmed",
    "rejected",
    "inconclusive",
}

ABLATION_REQUIRED_FIELDS = ["id", "description", "status"]

ABLATION_ALLOWED_STATUSES = {"planned", "running", "completed", "aborted"}


def validate_registry(
    data: dict[str, Any],
    collection_key: str,
    allowed_statuses: set[str],
    required_fields: list[str],
) -> None:
    """Validate a parsed registry document.

    Args:
        data: The parsed YAML document (must be a dict).
        collection_key: The top-level key containing the list of entries.
        allowed_statuses: Set of valid status strings.
        required_fields: List of fields required on every entry.

    Raises:
        RegistryError: On any contract violation.

    """
    if not isinstance(data, dict):
        raise RegistryError(f"document root must be a dict, got {type(data).__name__}")

    if collection_key not in data:
        raise RegistryError(f"missing required collection key: {collection_key!r}")

    entries = data[collection_key]
    if not isinstance(entries, list):
        raise RegistryError(f"{collection_key!r} must be a list, got {type(entries).__name__}")

    ids_seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RegistryError(f"{collection_key}[{i}] must be a dict, got {type(entry).__name__}")

        # ID validation
        if "id" not in entry:
            raise RegistryError(f"{collection_key}[{i}] missing 'id'")
        entry_id = entry["id"]
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise RegistryError(f"{collection_key}[{i}].id must be a nonempty string")
        if entry_id in ids_seen:
            raise RegistryError(f"{collection_key}[{i}].id {entry_id!r} is a duplicate")
        ids_seen.add(entry_id)

        # Status validation
        if "status" not in entry:
            raise RegistryError(f"{collection_key}[{i}] missing 'status'")
        status = entry["status"]
        if not isinstance(status, str) or status not in allowed_statuses:
            raise RegistryError(
                f"{collection_key}[{i}].status {status!r} not in {sorted(allowed_statuses)}"
            )

        # Required fields
        for field in required_fields:
            if field not in entry:
                raise RegistryError(f"{collection_key}[{i}] missing required field {field!r}")

        description = entry.get("description")
        if not isinstance(description, str) or not description.strip():
            raise RegistryError(f"{collection_key}[{i}].description must be a nonempty string")

        # Field shape checks for collections
        for list_field in ["outcome_metrics", "comparison_groups", "datasets", "required_controls"]:
            if list_field in entry:
                val = entry[list_field]
                if not isinstance(val, list):
                    raise RegistryError(f"{collection_key}[{i}].{list_field} must be a list")
                if len(val) == 0:
                    raise RegistryError(f"{collection_key}[{i}].{list_field} must not be empty")

        # Falsification must be a non-trivial string
        if "falsification" in entry:
            fals = entry["falsification"]
            if not isinstance(fals, str) or len(fals.strip()) < 10:
                raise RegistryError(
                    f"{collection_key}[{i}].falsification must be a string >= 10 chars"
                )


def validate_hypotheses(data: dict[str, Any]) -> None:
    """Validate a parsed hypotheses.yaml document."""
    validate_registry(
        data,
        collection_key="hypotheses",
        allowed_statuses=HYPOTHESIS_ALLOWED_STATUSES,
        required_fields=HYPOTHESIS_REQUIRED_FIELDS,
    )


def validate_ablations(data: dict[str, Any]) -> None:
    """Validate a parsed ablations.yaml document."""
    validate_registry(
        data,
        collection_key="ablations",
        allowed_statuses=ABLATION_ALLOWED_STATUSES,
        required_fields=ABLATION_REQUIRED_FIELDS,
    )
