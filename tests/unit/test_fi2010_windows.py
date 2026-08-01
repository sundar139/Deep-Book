from __future__ import annotations

import numpy as np

from deepbook.training.fi2010 import (
    SegmentedWindowDataset,
    label_to_index,
    make_windows,
    source_label,
    window_count,
)


def _matrix(observations: int = 5) -> tuple[np.ndarray, np.ndarray]:
    lob = np.arange(40 * observations, dtype=np.float32).reshape(40, observations)
    labels = np.vstack(
        [
            np.arange(1, observations + 1, dtype=np.int8) % 3 + 1,
            np.full(observations, 1, dtype=np.int8),
            np.full(observations, 2, dtype=np.int8),
            np.full(observations, 3, dtype=np.int8),
            np.full(observations, 1, dtype=np.int8),
        ]
    )
    return lob, labels


def test_window_count_and_first_last_target_alignment() -> None:
    lob, labels = _matrix()
    features, targets = make_windows(lob, labels, horizon_index=0, sequence_length=3)

    assert window_count(5, 3) == 3
    assert features.shape == (3, 3, 40)
    assert targets.tolist() == [int(labels[0, 2]) - 1, int(labels[0, 3]) - 1, int(labels[0, 4]) - 1]
    np.testing.assert_array_equal(features[0], lob[:, :3].T)
    np.testing.assert_array_equal(features[-1], lob[:, 2:5].T)


def test_source_label_mapping_round_trips() -> None:
    assert label_to_index(np.array([1, 2, 3], dtype=np.int8)).tolist() == [0, 1, 2]
    assert source_label(np.array([0, 1, 2], dtype=np.int8)).tolist() == [1, 2, 3]


def test_segmented_windows_do_not_cross_boundaries() -> None:
    first_lob, first_labels = _matrix(4)
    second_lob, second_labels = _matrix(4)
    second_lob += 10000
    dataset = SegmentedWindowDataset(
        [(first_lob, first_labels), (second_lob, second_labels)],
        horizon_index=0,
        sequence_length=3,
    )

    assert len(dataset) == 4
    first_x, _ = dataset[0]
    boundary_x, _ = dataset[2]
    assert float(first_x[0, -1, 0]) < 10000
    assert float(boundary_x[0, 0, 0]) >= 10000


def test_label_rows_are_not_in_window_features() -> None:
    lob, labels = _matrix()
    features, _ = make_windows(lob, labels, horizon_index=4, sequence_length=3)
    assert features.shape[-1] == 40
    assert not np.shares_memory(features, labels)
