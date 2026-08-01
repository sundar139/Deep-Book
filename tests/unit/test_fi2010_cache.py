from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepbook.training.fi2010 import (
    ModelingCacheSpec,
    load_matrix_cache,
    modeling_cache_path,
    save_matrix_cache,
)


def _spec(**overrides: object) -> ModelingCacheSpec:
    values: dict[str, object] = {
        "archive_sha256": "a" * 64,
        "source_matrix_sha256": "b" * 64,
        "benchmark_variant": "no_auction",
        "normalization": "zscore",
        "feature_rows": 144,
        "parser_version": "1.1",
        "sequence_length": 100,
        "horizon": 100,
        "configuration_hash": "c" * 64,
        "dtype": "float32",
    }
    values.update(overrides)
    return ModelingCacheSpec(**values)


def test_matrix_cache_is_atomic_and_identity_bound(tmp_path: Path) -> None:
    spec = _spec()
    path = modeling_cache_path(tmp_path, spec)
    features = np.ones((144, 4), dtype=np.float32)
    labels = np.ones((5, 4), dtype=np.int8)

    save_matrix_cache(path, features, labels, spec)
    loaded_features, loaded_labels = load_matrix_cache(path, spec)
    np.testing.assert_array_equal(loaded_features, features)
    np.testing.assert_array_equal(loaded_labels, labels)
    assert spec.key != _spec(sequence_length=99).key
    with pytest.raises(ValueError, match="metadata"):
        load_matrix_cache(path, _spec(sequence_length=99))
