"""Real-data acceptance for the first-seven/final-three setup.

Proves against the audited archive that Setup 2 fits the days 1-7 cumulative file
exactly once, tests days 8, 9, and 10 exactly once each, duplicates no test
observation, and never lets a window cross a day boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepbook.training.data import load_day_group
from deepbook.training.fi2010 import (
    DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
    SETUP_FIRST_SEVEN_FINAL_THREE,
    SegmentedWindowDataset,
    expected_data_fingerprint,
    frozen_data_identity,
    window_count,
)

pytestmark = pytest.mark.data

ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_LENGTH = 100


@pytest.fixture(scope="module")
def day_group():
    """Load the audited days 1-7 / days 8-10 matrices, skipping without local data."""
    split_manifest = ROOT / "data" / "interim" / "fi2010" / "fi2010_split_manifest.json"
    if not split_manifest.is_file():
        pytest.skip("audited FI-2010 split manifest is not locally available")
    return load_day_group(ROOT, DAY_GROUP_FIRST_SEVEN_FINAL_THREE)


def test_training_uses_the_cumulative_days_one_to_seven_file_exactly_once(day_group) -> None:
    frozen = frozen_data_identity(ROOT)["day_groups"][DAY_GROUP_FIRST_SEVEN_FINAL_THREE]
    assert day_group.training_file_sha256 == str(frozen["training_file_sha256"])
    assert day_group.training_labels.shape[1] == int(frozen["training_observations"])
    assert day_group.training_lob.shape == (40, int(frozen["training_observations"]))
    # The cumulative file is never also used as a test source.
    assert day_group.training_file_sha256 not in {day.file_sha256 for day in day_group.test_days}


def test_days_eight_nine_and_ten_each_appear_exactly_once(day_group) -> None:
    indices = [day.day_index for day in day_group.test_days]
    assert indices == [8, 9, 10]
    assert len(set(indices)) == 3
    assert len({day.file_sha256 for day in day_group.test_days}) == 3
    assert [day.source_fold for day in day_group.test_days] == [7, 8, 9]


def test_no_test_observation_is_duplicated_across_days(day_group) -> None:
    frozen = frozen_data_identity(ROOT)["day_groups"][DAY_GROUP_FIRST_SEVEN_FINAL_THREE]
    declared = {int(day["day_index"]): int(day["observations"]) for day in frozen["test_days"]}
    total = 0
    for day in day_group.test_days:
        assert day.observation_count == declared[day.day_index]
        total += day.observation_count
    assert total == sum(declared.values())
    # Distinct files of distinct lengths cannot be the same observations twice.
    assert len({day.observation_count for day in day_group.test_days}) == 3


def test_valid_sample_count_is_the_sum_of_independent_per_day_windows(day_group) -> None:
    per_day = [window_count(day.observation_count, SEQUENCE_LENGTH) for day in day_group.test_days]
    datasets = [
        SegmentedWindowDataset([(day.lob, day.labels)], 0, SEQUENCE_LENGTH)
        for day in day_group.test_days
    ]
    assert [len(dataset) for dataset in datasets] == per_day
    assert sum(per_day) == 55379 + 52073 + 31838

    total_observations = sum(day.observation_count for day in day_group.test_days)
    # Concatenating first would have manufactured windows spanning two days.
    assert sum(per_day) == total_observations - 3 * (SEQUENCE_LENGTH - 1)
    assert sum(per_day) < window_count(total_observations, SEQUENCE_LENGTH)


def test_no_window_spans_two_audited_days(day_group) -> None:
    for day in day_group.test_days:
        dataset = SegmentedWindowDataset([(day.lob, day.labels)], 0, SEQUENCE_LENGTH)
        first, _ = dataset[0]
        last, _ = dataset[len(dataset) - 1]
        assert np.array_equal(
            np.asarray(first)[0], np.asarray(day.lob[:, :SEQUENCE_LENGTH].T, dtype=np.float32)
        )
        assert np.array_equal(
            np.asarray(last)[0],
            np.asarray(day.lob[:, -SEQUENCE_LENGTH:].T, dtype=np.float32),
        )


def test_day_group_fingerprint_matches_the_frozen_contract(day_group) -> None:
    assert day_group.data_fingerprint == expected_data_fingerprint(
        ROOT,
        setup=SETUP_FIRST_SEVEN_FINAL_THREE,
        day_group=DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
    )
    assert sorted(day_group.testing_file_sha256_by_day) == ["10", "8", "9"]
    assert len(day_group.testing_file_sha256) == 64
