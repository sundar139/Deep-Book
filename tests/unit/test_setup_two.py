"""First-seven/final-three setup: planning, windowing, and boundary identity."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from deepbook.training.fi2010 import (
    DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
    SETUP_ANCHORED_FORWARD,
    SETUP_FIRST_SEVEN_FINAL_THREE,
    SegmentedWindowDataset,
    day_group_fingerprint,
    expected_data_fingerprint,
    frozen_data_identity,
    validate_run_manifest,
    window_count,
)
from deepbook.training.runner import (
    PredictionBundle,
    RunData,
    RunSpec,
    SourceSegment,
    _per_day_metrics,
    _seeds_for_model,
    planned_run_specs,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "data_contracts" / "fi2010_run_manifest.schema.json"
SEQUENCE_LENGTH = 100


def _segment(day_index: int, source_fold: int, observations: int) -> SourceSegment:
    rng = np.random.default_rng(day_index)
    lob = rng.standard_normal((40, observations)).astype(np.float32)
    labels = rng.integers(1, 4, (5, observations)).astype(np.int8)
    return SourceSegment(
        day_index=day_index,
        source_fold=source_fold,
        file_sha256=f"{day_index:064d}",
        lob=lob,
        features=np.concatenate([lob, rng.standard_normal((104, observations))]).astype(np.float32),
        labels=labels,
    )


# --- Planning ---------------------------------------------------------------


def test_planned_matrix_has_both_setups_with_model_specific_seeds() -> None:
    specs = planned_run_specs(ROOT)
    frozen = frozen_data_identity(ROOT)
    fold_count = len(frozen["folds"])
    day_group_count = len(frozen["day_groups"])
    horizons = 5

    by_setup = Counter(spec.setup for spec in specs)
    assert by_setup[SETUP_ANCHORED_FORWARD] > 0
    assert by_setup[SETUP_FIRST_SEVEN_FINAL_THREE] > 0

    expected_total = 0
    for model in (
        "majority",
        "causal_persistence",
        "logistic_current_event",
        "random_forest",
        "mlplob",
        "deeplob",
    ):
        seeds = len(_seeds_for_model(model))
        model_specs = [spec for spec in specs if spec.model == model]
        setup_one = [spec for spec in model_specs if spec.setup == SETUP_ANCHORED_FORWARD]
        setup_two = [spec for spec in model_specs if spec.setup == SETUP_FIRST_SEVEN_FINAL_THREE]
        assert len(setup_one) == fold_count * horizons * seeds, model
        assert len(setup_two) == day_group_count * horizons * seeds, model
        expected_total += len(setup_one) + len(setup_two)

    assert len(specs) == expected_total
    assert len(specs) == 900
    assert by_setup[SETUP_ANCHORED_FORWARD] == 810
    assert by_setup[SETUP_FIRST_SEVEN_FINAL_THREE] == 90


def test_deterministic_classical_models_plan_one_seed_only() -> None:
    for model in ("majority", "causal_persistence", "logistic_current_event"):
        assert _seeds_for_model(model) == (1337,)
    for model in ("random_forest", "mlplob", "deeplob"):
        assert len(_seeds_for_model(model)) == 5


def test_planned_run_specs_are_unique_and_deterministic() -> None:
    first = planned_run_specs(ROOT)
    second = planned_run_specs(ROOT)
    assert [spec.run_id for spec in first] == [spec.run_id for spec in second]
    assert len({spec.run_id for spec in first}) == len(first)


def test_setup_two_run_identifier_names_the_day_group() -> None:
    spec = RunSpec(
        model="deeplob",
        setup=SETUP_FIRST_SEVEN_FINAL_THREE,
        horizon=10,
        seed=1337,
        day_group=DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
    )
    assert spec.run_id == "deeplob-first-seven-final-three-days-8-9-10-h10-s1337"
    assert spec.fold is None


# --- Frozen contract --------------------------------------------------------


def test_frozen_contract_uses_days_one_to_seven_once_and_three_distinct_test_days() -> None:
    group = frozen_data_identity(ROOT)["day_groups"][DAY_GROUP_FIRST_SEVEN_FINAL_THREE]
    assert group["training_days"] == [1, 2, 3, 4, 5, 6, 7]
    assert group["training_fold"] == 7

    days = [int(day["day_index"]) for day in group["test_days"]]
    assert days == [8, 9, 10]
    assert len(set(days)) == 3

    digests = [str(day["file_sha256"]) for day in group["test_days"]]
    assert len(set(digests)) == 3
    assert str(group["training_file_sha256"]) not in digests


def test_setup_two_fingerprint_is_derived_from_every_audited_file() -> None:
    frozen = frozen_data_identity(ROOT)
    group = frozen["day_groups"][DAY_GROUP_FIRST_SEVEN_FINAL_THREE]
    testing_by_day = {str(day["day_index"]): str(day["file_sha256"]) for day in group["test_days"]}
    computed = day_group_fingerprint(
        str(frozen["archive_sha256"]),
        DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
        str(group["training_file_sha256"]),
        testing_by_day,
    )
    assert computed == expected_data_fingerprint(
        ROOT,
        setup=SETUP_FIRST_SEVEN_FINAL_THREE,
        day_group=DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
    )

    tampered = dict(testing_by_day)
    tampered["9"] = "0" * 64
    assert (
        day_group_fingerprint(
            str(frozen["archive_sha256"]),
            DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
            str(group["training_file_sha256"]),
            tampered,
        )
        != computed
    )


def test_frozen_contract_sample_count_matches_independent_day_windowing() -> None:
    group = frozen_data_identity(ROOT)["day_groups"][DAY_GROUP_FIRST_SEVEN_FINAL_THREE]
    per_day = [
        window_count(int(day["observations"]), SEQUENCE_LENGTH) for day in group["test_days"]
    ]
    total_observations = sum(int(day["observations"]) for day in group["test_days"])
    # Independent windowing loses sequence_length - 1 samples per day, never once overall.
    assert sum(per_day) == total_observations - 3 * (SEQUENCE_LENGTH - 1)
    assert sum(per_day) < window_count(total_observations, SEQUENCE_LENGTH)


# --- Window construction ----------------------------------------------------


def test_no_window_crosses_a_day_boundary() -> None:
    segments = [_segment(8, 7, 400), _segment(9, 8, 300), _segment(10, 9, 250)]
    sequence_length = 50
    per_day = [
        SegmentedWindowDataset([(seg.lob, seg.labels)], 0, sequence_length) for seg in segments
    ]
    combined = SegmentedWindowDataset(
        [(seg.lob, seg.labels) for seg in segments], 0, sequence_length
    )
    assert len(combined) == sum(len(dataset) for dataset in per_day)

    # Every window drawn from the combined dataset must be byte-identical to a
    # window of exactly one day, which is only possible if none of them span days.
    offset = 0
    for segment, dataset in zip(segments, per_day, strict=True):
        for index in range(len(dataset)):
            combined_window, combined_target = combined[offset + index]
            day_window, day_target = dataset[index]
            assert torch_equal(combined_window, day_window), (segment.day_index, index)
            assert int(combined_target) == int(day_target)
        offset += len(dataset)


def torch_equal(left: object, right: object) -> bool:
    """Return True when two tensors hold identical values."""
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def test_independent_day_windows_never_reuse_an_observation() -> None:
    segments = [_segment(8, 7, 200), _segment(9, 8, 180)]
    sequence_length = 40
    seen: set[tuple[int, float]] = set()
    for segment in segments:
        dataset = SegmentedWindowDataset([(segment.lob, segment.labels)], 0, sequence_length)
        for index in range(len(dataset)):
            window, _ = dataset[index]
            marker = (segment.day_index, float(np.asarray(window)[0, -1, 0]))
            assert marker not in seen
            seen.add(marker)


# --- Boundary identifiers ---------------------------------------------------


def test_concatenated_predictions_keep_source_and_day_identity() -> None:
    segments = [_segment(8, 7, 60), _segment(9, 8, 40), _segment(10, 9, 25)]
    bundle = PredictionBundle()
    rng = np.random.default_rng(0)
    for segment in segments:
        count = segment.observation_count
        true_values = rng.integers(0, 3, count).astype(np.int64)
        predictions = rng.integers(0, 3, count).astype(np.int64)
        probabilities = np.full((count, 3), 1.0 / 3.0)
        bundle.add(segment, true_values, predictions, probabilities)

    payload = bundle.concatenate()
    assert payload["y_true"].shape[0] == sum(seg.observation_count for seg in segments)
    assert payload["day_boundary_id"].dtype == np.int64
    assert payload["source_file_id"].dtype == np.int64

    for segment in segments:
        mask = payload["day_boundary_id"] == segment.day_index
        assert int(mask.sum()) == segment.observation_count
        assert set(payload["source_file_id"][mask].tolist()) == {segment.source_fold}
        # sample_index restarts at zero within each independently windowed day
        assert payload["sample_index"][mask].tolist() == list(range(segment.observation_count))


def test_per_day_metrics_aggregate_consistently_with_combined_metrics() -> None:
    segments = [_segment(8, 7, 90), _segment(9, 8, 60), _segment(10, 9, 30)]
    bundle = PredictionBundle()
    rng = np.random.default_rng(5)
    for segment in segments:
        count = segment.observation_count
        true_values = rng.integers(0, 3, count).astype(np.int64)
        predictions = true_values.copy()
        predictions[: count // 3] = (predictions[: count // 3] + 1) % 3
        probabilities = np.eye(3)[predictions].astype(np.float64)
        bundle.add(segment, true_values, predictions, probabilities)

    payload = bundle.concatenate()
    per_day = _per_day_metrics(payload)
    assert sorted(per_day) == ["10", "8", "9"]
    assert sum(per_day[key]["sample_count"] for key in per_day) == payload["y_true"].shape[0]

    weighted = (
        sum(per_day[key]["accuracy"] * per_day[key]["sample_count"] for key in per_day)
        / payload["y_true"].shape[0]
    )
    combined_accuracy = float(np.mean(payload["y_pred"] == payload["y_true"]))
    assert weighted == pytest.approx(combined_accuracy)


def test_run_data_day_index_map_records_each_audited_source_file() -> None:
    segments = tuple(_segment(day, fold, 50) for day, fold in ((8, 7), (9, 8), (10, 9)))
    data = RunData(
        setup=SETUP_FIRST_SEVEN_FINAL_THREE,
        fold=None,
        day_group=DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
        training_lob=np.zeros((40, 10), dtype=np.float32),
        training_features=np.zeros((144, 10), dtype=np.float32),
        training_labels=np.ones((5, 10), dtype=np.int8),
        test_segments=segments,
        archive_sha256="a" * 64,
        training_file_sha256="b" * 64,
        testing_file_sha256="c" * 64,
    )
    mapping = data.day_index_map
    assert sorted(mapping) == ["10", "8", "9"]
    assert mapping["8"]["source_fold"] == 7
    assert mapping["10"]["observations"] == 50


# --- Schema conditions ------------------------------------------------------


def _manifest(**overrides: object) -> dict:
    manifest = {
        "schema_version": 1,
        "run_id": "setup-two-run",
        "run_kind": "smoke",
        "eligible_for_confirmatory_report": False,
        "exclusion_reasons": ["probe"],
        "protocol_commit": "0" * 40,
        "protocol_sha256": "a" * 64,
        "code_commit": "1" * 40,
        "code_commit_timestamp": "2026-01-01T00:00:00+00:00",
        "code_commit_tree": "2" * 40,
        "git_tree_dirty": False,
        "model": "majority",
        "setup": SETUP_FIRST_SEVEN_FINAL_THREE,
        "fold": None,
        "day_group": DAY_GROUP_FIRST_SEVEN_FINAL_THREE,
        "horizon": 10,
        "seed": 1337,
        "status": "completed",
        "metrics": {"test": {"macro_f1": 0.5}},
        "configuration_path": "configs/experiments/fi2010/classical.yaml",
        "configuration_hash": "b" * 64,
        "data_fingerprint": "fingerprint",
        "archive_sha256": "c" * 64,
        "training_file_sha256": "d" * 64,
        "testing_file_sha256": "e" * 64,
        "testing_file_sha256_by_day": {"8": "f" * 64, "9": "0" * 64, "10": "1" * 64},
        "day_index_map": {
            "8": {"source_fold": 7, "file_sha256": "f" * 64, "observations": 55478},
            "9": {"source_fold": 8, "file_sha256": "0" * 64, "observations": 52172},
            "10": {"source_fold": 9, "file_sha256": "1" * 64, "observations": 31937},
        },
        "termination_reason": "not_applicable",
        "configured_max_epochs": None,
        "actual_epochs_completed": None,
        "best_epoch": None,
        "parameter_count": 0,
        "started_utc": "2026-01-01T00:00:00Z",
        "completed_utc": "2026-01-01T00:01:00Z",
        "resumed": False,
        "device": "cpu",
        "environment": {"python": "3.11"},
        "prediction_path": "artifacts/fi2010/baselines/predictions/setup-two-run.npz",
        "prediction_sha256": "2" * 64,
        "sample_count": 139290,
        "class_order": ["up", "stationary", "down"],
    }
    manifest.update(overrides)
    return manifest


def test_setup_two_manifest_requires_null_fold_and_a_day_group() -> None:
    validate_run_manifest(_manifest(), SCHEMA)
    with pytest.raises(Exception, match="."):
        validate_run_manifest(_manifest(fold=3), SCHEMA)
    with pytest.raises(Exception, match="."):
        validate_run_manifest(_manifest(day_group=None), SCHEMA)


def test_setup_one_manifest_requires_a_fold_and_a_null_day_group() -> None:
    setup_one = _manifest(setup=SETUP_ANCHORED_FORWARD, fold=4, day_group=None)
    validate_run_manifest(setup_one, SCHEMA)
    with pytest.raises(Exception, match="."):
        validate_run_manifest(
            _manifest(setup=SETUP_ANCHORED_FORWARD, fold=None, day_group=None), SCHEMA
        )
    with pytest.raises(Exception, match="."):
        validate_run_manifest(
            _manifest(setup=SETUP_ANCHORED_FORWARD, fold=4, day_group="days_8_9_10"), SCHEMA
        )
