"""FI-2010 matrix discovery, layout validation, and statistics."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepbook.data.fi2010 import PARSER_VERSION
from deepbook.data.fi2010.acquisition import AcquisitionError, sha256_file

LOGICAL_ROWS = 149
LOB_ROWS = 40
ENGINEERED_FEATURE_ROWS = 104
LABEL_ROWS = 5
HORIZONS = (10, 20, 30, 50, 100)
CLASS_SEMANTICS = {1: "up", 2: "stationary", 3: "down"}
INTERNAL_ORIENTATION = "variables_by_observation"

_FILE_PATTERN = re.compile(
    r"^(?P<role>Train|Test)_Dst_(?P<variant>NoAuction|Auction)_"
    r"(?P<normalization>ZScore|MinMax|DecPre)_CF_(?P<fold>[0-9]+)\.txt$",
    re.IGNORECASE,
)
_VARIANTS = {"noauction": "no_auction", "auction": "auction"}
_NORMALIZATIONS = {"zscore": "zscore", "minmax": "minmax", "decpre": "decimal_precision"}
_ROLES = {"train": "training", "test": "testing"}


class MatrixError(ValueError):
    """Raised when an FI-2010 matrix violates the published layout."""


@dataclass(frozen=True)
class MatrixFile:
    """A processed benchmark file identified from its authoritative name."""

    member_path: str
    path: Path
    role: str
    benchmark_variant: str
    normalization: str
    fold: int


@dataclass(frozen=True)
class ParsedMatrix:
    """Validated matrix and zero-copy row-family views."""

    source: MatrixFile
    values: Any
    lob: Any
    engineered_features: Any
    labels: Any
    source_orientation: str

    @property
    def observation_count(self) -> int:
        """Return the number of event representations in the matrix."""
        return int(self.values.shape[1])


def _matrix_file(member_path: str, extraction_root: Path) -> MatrixFile | None:
    match = _FILE_PATTERN.fullmatch(Path(member_path).name)
    if match is None:
        return None
    return MatrixFile(
        member_path=member_path,
        path=extraction_root / Path(member_path),
        role=_ROLES[match.group("role").lower()],
        benchmark_variant=_VARIANTS[match.group("variant").lower()],
        normalization=_NORMALIZATIONS[match.group("normalization").lower()],
        fold=int(match.group("fold")),
    )


def discover_matrices(
    extraction_root: Path,
    member_paths: list[str] | tuple[str, ...],
) -> list[MatrixFile]:
    """Discover named benchmark matrices while preserving archive order."""
    matrices = []
    for member_path in member_paths:
        matrix = _matrix_file(member_path, extraction_root)
        if matrix is not None:
            matrices.append(matrix)
    return matrices


def select_matrices(
    matrices: list[MatrixFile], benchmark_variant: str, normalization: str
) -> list[MatrixFile]:
    """Select one published benchmark configuration without inventing files."""
    selected = [
        matrix
        for matrix in matrices
        if matrix.benchmark_variant == benchmark_variant and matrix.normalization == normalization
    ]
    if not selected:
        raise MatrixError(
            f"no matrices found for variant={benchmark_variant!r}, normalization={normalization!r}"
        )
    folds: dict[int, set[str]] = {}
    for matrix in selected:
        roles = folds.setdefault(matrix.fold, set())
        if matrix.role in roles:
            raise MatrixError(f"duplicate {matrix.role} matrix for fold {matrix.fold}")
        roles.add(matrix.role)
    incomplete = {fold: roles for fold, roles in folds.items() if roles != {"training", "testing"}}
    if incomplete:
        raise MatrixError(f"incomplete training/testing fold pairs: {incomplete}")
    return selected


def _validate_source_file(path: Path, extraction_root: Path) -> None:
    try:
        resolved = path.resolve(strict=True)
        root = extraction_root.resolve(strict=True)
    except OSError as error:
        raise MatrixError(f"matrix path cannot be resolved: {path}") from error
    if not resolved.is_relative_to(root):
        raise MatrixError(f"matrix is outside the validated extraction root: {path}")
    if path.is_symlink() or not resolved.is_file():
        raise MatrixError(f"matrix is not a regular file: {path}")


def parse_matrix(source: MatrixFile, extraction_root: Path) -> ParsedMatrix:
    """Load one numeric matrix and normalize it to variables-by-observation."""
    _validate_source_file(source.path, extraction_root)
    try:
        import numpy as np
    except ImportError as error:
        raise MatrixError("NumPy is required; install the data dependency group") from error

    try:
        raw = np.loadtxt(source.path, dtype=np.float64, comments=None, ndmin=2)
    except (OSError, ValueError) as error:
        raise MatrixError(f"cannot parse numeric matrix {source.member_path}: {error}") from error
    if raw.ndim != 2:
        raise MatrixError(f"matrix must be two-dimensional: {source.member_path}")
    rows, columns = (int(raw.shape[0]), int(raw.shape[1]))
    if rows == LOGICAL_ROWS and columns == LOGICAL_ROWS:
        raise MatrixError(f"matrix orientation is ambiguous at 149 by 149: {source.member_path}")
    if rows == LOGICAL_ROWS:
        values = raw
        source_orientation = INTERNAL_ORIENTATION
    elif columns == LOGICAL_ROWS:
        values = raw.T
        source_orientation = "observation_by_variable"
    else:
        raise MatrixError(
            f"matrix must have exactly {LOGICAL_ROWS} logical variables, got {rows} by {columns}: "
            f"{source.member_path}"
        )
    if values.shape[1] == 0:
        raise MatrixError(f"matrix has no observations: {source.member_path}")
    if not np.isfinite(values).all():
        raise MatrixError(f"matrix contains NaN or infinity: {source.member_path}")

    lob = values[:LOB_ROWS]
    engineered = values[LOB_ROWS : LOB_ROWS + ENGINEERED_FEATURE_ROWS]
    labels = values[-LABEL_ROWS:]
    if lob.shape[0] != LOB_ROWS or engineered.shape[0] != ENGINEERED_FEATURE_ROWS:
        raise MatrixError(f"matrix row-family layout is invalid: {source.member_path}")
    rounded = np.rint(labels)
    if not np.equal(labels, rounded).all():
        raise MatrixError(f"label rows are not integer-valued: {source.member_path}")
    observed_domain = {int(value) for value in np.unique(rounded)}
    if not observed_domain.issubset(CLASS_SEMANTICS):
        raise MatrixError(
            f"label domain {sorted(observed_domain)} is outside {sorted(CLASS_SEMANTICS)}: "
            f"{source.member_path}"
        )
    return ParsedMatrix(
        source=source,
        values=values,
        lob=lob,
        engineered_features=engineered,
        labels=labels,
        source_orientation=source_orientation,
    )


def _numeric_line(line: bytes, source: MatrixFile, line_number: int) -> Any:
    import numpy as np

    try:
        text = line.decode("ascii")
    except UnicodeDecodeError as error:
        raise MatrixError(
            f"matrix contains non-ASCII data at line {line_number}: {source.member_path}"
        ) from error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            values = np.fromstring(text, dtype=np.float64, sep=" ")
    except (DeprecationWarning, ValueError) as error:
        raise MatrixError(
            f"matrix has a malformed numeric token at line {line_number}: {source.member_path}"
        ) from error
    if values.size == 0:
        raise MatrixError(f"matrix has an empty row at line {line_number}: {source.member_path}")
    if not np.isfinite(values).all():
        raise MatrixError(f"matrix contains NaN or infinity: {source.member_path}")
    return values


def _label_counts(values: Any, source: MatrixFile) -> dict[str, int]:
    import numpy as np

    rounded = np.rint(values)
    if not np.equal(values, rounded).all():
        raise MatrixError(f"label rows are not integer-valued: {source.member_path}")
    observed = {int(value) for value in np.unique(rounded)}
    if not observed.issubset(CLASS_SEMANTICS):
        raise MatrixError(
            f"label domain {sorted(observed)} is outside {sorted(CLASS_SEMANTICS)}: "
            f"{source.member_path}"
        )
    return {str(label): int(np.count_nonzero(rounded == label)) for label in CLASS_SEMANTICS}


def _row_statistics(row_number: int, values: Any) -> dict[str, float | int]:
    import numpy as np

    return {
        "row": row_number,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "standard_deviation": float(np.std(values, dtype=np.float64)),
    }


def stream_matrix_audit(source: MatrixFile, extraction_root: Path) -> dict[str, Any]:
    """Validate and audit one matrix without loading the complete matrix into memory."""
    import hashlib

    import numpy as np

    _validate_source_file(source.path, extraction_root)
    digest = hashlib.sha256()
    row_statistics: list[dict[str, float | int]] = []
    label_counts: dict[str, dict[str, int]] = {}
    try:
        with source.path.open("rb") as stream:
            first_line = stream.readline()
            digest.update(first_line)
            first = _numeric_line(first_line, source, 1)
            if int(first.size) != LOGICAL_ROWS:
                observation_count = int(first.size)
                row_statistics.append(_row_statistics(1, first))
                rows_seen = 1
                for line_number, line in enumerate(stream, start=2):
                    digest.update(line)
                    values = _numeric_line(line, source, line_number)
                    if int(values.size) != observation_count:
                        raise MatrixError(
                            f"matrix row {line_number} has {int(values.size)} values, "
                            f"expected {observation_count}: {source.member_path}"
                        )
                    row_statistics.append(_row_statistics(line_number, values))
                    if line_number > LOB_ROWS + ENGINEERED_FEATURE_ROWS:
                        horizon = str(HORIZONS[line_number - LOGICAL_ROWS + LABEL_ROWS - 1])
                        label_counts[horizon] = _label_counts(values, source)
                    rows_seen = line_number
                if rows_seen != LOGICAL_ROWS:
                    raise MatrixError(
                        f"matrix must have exactly {LOGICAL_ROWS} logical variables, "
                        f"got {rows_seen} rows: {source.member_path}"
                    )
                source_orientation = INTERNAL_ORIENTATION
            else:
                minima = first.copy()
                maxima = first.copy()
                sums = first.copy()
                sums_of_squares = np.square(first, dtype=np.float64)
                label_counters = {
                    str(horizon): {str(label): 0 for label in CLASS_SEMANTICS}
                    for horizon in HORIZONS
                }
                first_labels = first[-LABEL_ROWS:]
                for index, horizon in enumerate(map(str, HORIZONS)):
                    counts = _label_counts(first_labels[index : index + 1], source)
                    for label, count in counts.items():
                        label_counters[horizon][label] += count
                observation_count = 1
                for line_number, line in enumerate(stream, start=2):
                    digest.update(line)
                    values = _numeric_line(line, source, line_number)
                    if int(values.size) != LOGICAL_ROWS:
                        raise MatrixError(
                            f"matrix observation {line_number} has {int(values.size)} values, "
                            f"expected {LOGICAL_ROWS}: {source.member_path}"
                        )
                    minima = np.minimum(minima, values)
                    maxima = np.maximum(maxima, values)
                    sums += values
                    sums_of_squares += np.square(values, dtype=np.float64)
                    labels = values[-LABEL_ROWS:]
                    for index, horizon in enumerate(map(str, HORIZONS)):
                        counts = _label_counts(labels[index : index + 1], source)
                        for label, count in counts.items():
                            label_counters[horizon][label] += count
                    observation_count += 1
                if observation_count == LOGICAL_ROWS:
                    raise MatrixError(
                        f"matrix orientation is ambiguous at 149 by 149: {source.member_path}"
                    )
                means = sums / observation_count
                variances = np.maximum(sums_of_squares / observation_count - np.square(means), 0.0)
                standard_deviations = np.sqrt(variances)
                row_statistics = [
                    {
                        "row": index + 1,
                        "minimum": float(minima[index]),
                        "maximum": float(maxima[index]),
                        "mean": float(means[index]),
                        "standard_deviation": float(standard_deviations[index]),
                    }
                    for index in range(LOGICAL_ROWS)
                ]
                label_counts = label_counters
                source_orientation = "observation_by_variable"
    except OSError as error:
        raise MatrixError(f"cannot read numeric matrix {source.member_path}: {error}") from error

    constant_rows = [int(row["row"]) for row in row_statistics if row["minimum"] == row["maximum"]]
    all_zero_rows = [
        int(row["row"]) for row in row_statistics if row["minimum"] == 0.0 and row["maximum"] == 0.0
    ]
    distributions = {
        horizon: {
            "counts": counts,
            "proportions": {label: count / observation_count for label, count in counts.items()},
        }
        for horizon, counts in label_counts.items()
    }
    return {
        "member_path": source.member_path,
        "file_sha256": digest.hexdigest(),
        "role": source.role,
        "fold": source.fold,
        "benchmark_variant": source.benchmark_variant,
        "normalization": source.normalization,
        "source_orientation": source_orientation,
        "internal_orientation": INTERNAL_ORIENTATION,
        "row_count": LOGICAL_ROWS,
        "observation_count": observation_count,
        "lob_row_count": LOB_ROWS,
        "engineered_feature_row_count": ENGINEERED_FEATURE_ROWS,
        "label_row_count": LABEL_ROWS,
        "horizons_events": list(HORIZONS),
        "class_encodings": {str(key): value for key, value in CLASS_SEMANTICS.items()},
        "label_distributions": distributions,
        "observed_missing_tokens": 0,
        "observed_nonfinite_tokens": 0,
        "rejected_rows": 0,
        "validated_nonfinite_values": 0,
        "all_zero_rows": all_zero_rows,
        "constant_rows": constant_rows,
        "row_statistics": row_statistics,
        "parser_version": PARSER_VERSION,
    }


def _counts_and_proportions(labels: Any) -> dict[str, dict[str, Any]]:
    import numpy as np

    result: dict[str, dict[str, Any]] = {}
    for index, horizon in enumerate(HORIZONS):
        values, counts = np.unique(labels[index].astype(np.int8, copy=False), return_counts=True)
        by_class = {str(label): 0 for label in CLASS_SEMANTICS}
        for label, count in zip(values.tolist(), counts.tolist(), strict=True):
            by_class[str(int(label))] = int(count)
        total = sum(by_class.values())
        result[str(horizon)] = {
            "counts": by_class,
            "proportions": {
                label: count / total if total else 0.0 for label, count in by_class.items()
            },
        }
    return result


def matrix_audit(parsed: ParsedMatrix) -> dict[str, Any]:
    """Return deterministic statistics for one validated matrix."""
    import numpy as np

    values = parsed.values
    row_minimum = np.asarray(np.min(values, axis=1))
    row_maximum = np.asarray(np.max(values, axis=1))
    row_mean = np.asarray(np.mean(values, axis=1, dtype=np.float64))
    row_standard_deviation = np.asarray(np.std(values, axis=1, dtype=np.float64))
    constant = np.equal(row_minimum, row_maximum)
    all_zero = np.logical_and(constant, np.equal(row_minimum, 0.0))
    rows = [
        {
            "row": index + 1,
            "minimum": float(row_minimum[index]),
            "maximum": float(row_maximum[index]),
            "mean": float(row_mean[index]),
            "standard_deviation": float(row_standard_deviation[index]),
        }
        for index in range(LOGICAL_ROWS)
    ]
    return {
        "member_path": parsed.source.member_path,
        "file_sha256": sha256_file(parsed.source.path),
        "role": parsed.source.role,
        "fold": parsed.source.fold,
        "benchmark_variant": parsed.source.benchmark_variant,
        "normalization": parsed.source.normalization,
        "source_orientation": parsed.source_orientation,
        "internal_orientation": INTERNAL_ORIENTATION,
        "row_count": LOGICAL_ROWS,
        "observation_count": parsed.observation_count,
        "lob_row_count": LOB_ROWS,
        "engineered_feature_row_count": ENGINEERED_FEATURE_ROWS,
        "label_row_count": LABEL_ROWS,
        "horizons_events": list(HORIZONS),
        "class_encodings": {str(key): value for key, value in CLASS_SEMANTICS.items()},
        "label_distributions": _counts_and_proportions(parsed.labels),
        "observed_missing_tokens": 0,
        "observed_nonfinite_tokens": 0,
        "rejected_rows": 0,
        "validated_nonfinite_values": 0,
        "all_zero_rows": (np.flatnonzero(all_zero) + 1).tolist(),
        "constant_rows": (np.flatnonzero(constant) + 1).tolist(),
        "row_statistics": rows,
        "parser_version": PARSER_VERSION,
    }


def fold_day_indices(fold: int, role: str) -> list[int]:
    """Return source-described anchored day indices for a file role."""
    if fold < 1:
        raise MatrixError(f"fold identifier must be positive, got {fold}")
    if role == "training":
        return list(range(1, fold + 1))
    if role == "testing":
        return [fold + 1]
    raise MatrixError(f"unknown file role: {role}")


def acquisition_error_to_matrix(error: AcquisitionError) -> MatrixError:
    """Translate an acquisition failure at a parser boundary."""
    return MatrixError(str(error))
