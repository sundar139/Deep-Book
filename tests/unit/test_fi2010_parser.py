"""Synthetic FI-2010 matrix parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepbook.data.fi2010.dataset import (
    MatrixError,
    MatrixFile,
    matrix_audit,
    parse_matrix,
    stream_matrix_audit,
)


def _rows(observations: int = 3) -> list[list[float]]:
    values = [
        [float((row + 1) * 100 + column) for column in range(observations)] for row in range(144)
    ]
    labels = [[float((column % 3) + 1) for column in range(observations)] for _ in range(5)]
    return values + labels


def _write(path: Path, rows: list[list[float]], transpose: bool = False) -> None:
    values = list(map(list, zip(*rows, strict=True))) if transpose else rows
    path.write_text("\n".join(" ".join(map(str, row)) for row in values) + "\n", encoding="utf-8")


def _source(path: Path) -> MatrixFile:
    return MatrixFile(
        member_path=path.name,
        path=path,
        role="training",
        benchmark_variant="no_auction",
        normalization="zscore",
        fold=1,
    )


@pytest.mark.parametrize(
    ("transpose", "source_orientation"),
    [(False, "variables_by_observation"), (True, "observation_by_variable")],
)
def test_parser_detects_orientation_and_row_families(
    tmp_path: Path, transpose: bool, source_orientation: str
) -> None:
    path = tmp_path / "matrix.txt"
    _write(path, _rows(), transpose)

    parsed = parse_matrix(_source(path), tmp_path)

    assert parsed.values.shape == (149, 3)
    assert parsed.lob.shape == (40, 3)
    assert parsed.engineered_features.shape == (104, 3)
    assert parsed.labels.shape == (5, 3)
    assert parsed.source_orientation == source_orientation
    streamed = stream_matrix_audit(_source(path), tmp_path)
    loaded = matrix_audit(parsed)
    for key in (
        "file_sha256",
        "source_orientation",
        "observation_count",
        "label_distributions",
        "constant_rows",
        "all_zero_rows",
    ):
        assert streamed[key] == loaded[key]
    for streamed_row, loaded_row in zip(
        streamed["row_statistics"], loaded["row_statistics"], strict=True
    ):
        assert streamed_row["row"] == loaded_row["row"]
        assert streamed_row["minimum"] == loaded_row["minimum"]
        assert streamed_row["maximum"] == loaded_row["maximum"]
        assert streamed_row["mean"] == pytest.approx(loaded_row["mean"])
        assert streamed_row["standard_deviation"] == pytest.approx(loaded_row["standard_deviation"])


def test_parser_rejects_wrong_logical_row_count(tmp_path: Path) -> None:
    path = tmp_path / "matrix.txt"
    _write(path, _rows()[:-1])
    with pytest.raises(MatrixError, match="149 logical variables"):
        parse_matrix(_source(path), tmp_path)


def test_parser_rejects_inconsistent_row_length(tmp_path: Path) -> None:
    rows = _rows()
    rows[10] = rows[10][:-1]
    path = tmp_path / "matrix.txt"
    _write(path, rows)
    with pytest.raises(MatrixError, match="cannot parse numeric matrix"):
        parse_matrix(_source(path), tmp_path)


def test_parser_rejects_malformed_numeric_token(tmp_path: Path) -> None:
    path = tmp_path / "matrix.txt"
    _write(path, _rows())
    content = path.read_text(encoding="utf-8").replace("100.0", "not-a-number", 1)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(MatrixError, match="cannot parse numeric matrix"):
        parse_matrix(_source(path), tmp_path)


def test_parser_rejects_noninteger_label(tmp_path: Path) -> None:
    rows = _rows()
    rows[-1][0] = 1.5
    path = tmp_path / "matrix.txt"
    _write(path, rows)
    with pytest.raises(MatrixError, match="not integer-valued"):
        parse_matrix(_source(path), tmp_path)


def test_parser_rejects_invalid_label_domain(tmp_path: Path) -> None:
    rows = _rows()
    rows[-1][0] = 4.0
    path = tmp_path / "matrix.txt"
    _write(path, rows)
    with pytest.raises(MatrixError, match="outside"):
        parse_matrix(_source(path), tmp_path)


@pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
def test_parser_rejects_nonfinite_values(tmp_path: Path, token: str) -> None:
    path = tmp_path / "matrix.txt"
    _write(path, _rows())
    content = path.read_text(encoding="utf-8").replace("100.0", token, 1)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(MatrixError, match="NaN or infinity"):
        parse_matrix(_source(path), tmp_path)


def test_matrix_audit_reports_labels_and_constant_rows(tmp_path: Path) -> None:
    rows = _rows()
    rows[0] = [0.0, 0.0, 0.0]
    path = tmp_path / "matrix.txt"
    _write(path, rows)

    result = matrix_audit(parse_matrix(_source(path), tmp_path))

    assert result["all_zero_rows"] == [1]
    assert result["constant_rows"] == [1]
    assert result["label_distributions"]["10"]["counts"] == {"1": 1, "2": 1, "3": 1}
    assert result["observed_missing_tokens"] == 0
    assert result["observed_nonfinite_tokens"] == 0
    assert result["validated_nonfinite_values"] == 0


def test_streaming_audit_rejects_malformed_numeric_token(tmp_path: Path) -> None:
    path = tmp_path / "matrix.txt"
    _write(path, _rows())
    content = path.read_text(encoding="utf-8").replace("100.0", "not-a-number", 1)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(MatrixError, match="malformed numeric token"):
        stream_matrix_audit(_source(path), tmp_path)


def test_streaming_audit_rejects_invalid_label(tmp_path: Path) -> None:
    rows = _rows()
    rows[-1][0] = 4.0
    path = tmp_path / "matrix.txt"
    _write(path, rows)
    with pytest.raises(MatrixError, match="outside"):
        stream_matrix_audit(_source(path), tmp_path)


def test_parser_rejects_file_outside_extraction_root(tmp_path: Path) -> None:
    extraction = tmp_path / "extracted"
    extraction.mkdir()
    path = tmp_path / "outside.txt"
    _write(path, _rows())
    with pytest.raises(MatrixError, match="outside"):
        parse_matrix(_source(path), extraction)
