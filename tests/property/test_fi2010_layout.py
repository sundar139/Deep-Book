"""Property checks for FI-2010 matrix orientation normalization."""

from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from deepbook.data.fi2010.dataset import MatrixFile, parse_matrix


@given(observations=st.integers(min_value=1, max_value=20), transpose=st.booleans())
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_supported_orientation_preserves_observation_count(
    tmp_path: Path, observations: int, transpose: bool
) -> None:
    rows = [
        [float((row + 1) * 100 + column) for column in range(observations)] for row in range(144)
    ]
    rows.extend([[float((column % 3) + 1) for column in range(observations)] for _ in range(5)])
    output = list(map(list, zip(*rows, strict=True))) if transpose else rows
    path = tmp_path / "matrix.txt"
    path.write_text("\n".join(" ".join(map(str, row)) for row in output) + "\n", encoding="utf-8")
    source = MatrixFile(
        member_path=path.name,
        path=path,
        role="testing",
        benchmark_variant="no_auction",
        normalization="zscore",
        fold=1,
    )

    parsed = parse_matrix(source, tmp_path)

    assert parsed.values.shape == (149, observations)
    assert parsed.observation_count == observations
