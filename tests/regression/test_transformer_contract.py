"""Regression checks for the frozen transformer experiment contracts."""

from __future__ import annotations

from pathlib import Path

import yaml

from deepbook.training.runner import (
    ALL_MODELS,
    SETUP_ANCHORED_FORWARD,
    SETUP_FIRST_SEVEN_FINAL_THREE,
    planned_run_specs,
)

ROOT = Path(__file__).resolve().parents[2]


def _config(model: str) -> dict[str, object]:
    path = ROOT / "configs" / "experiments" / "fi2010" / f"{model}.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_transformer_configs_use_official_labels_and_common_targets() -> None:
    assert {"translob", "tlob"}.issubset(ALL_MODELS)
    for model in ("translob", "tlob"):
        config = _config(model)
        assert config["input_rows"] == 40
        assert config["sequence_length"] == 100
        assert config["horizons"] == [10, 20, 30, 50, 100]
        assert config["labels"] == "official_supplied"
        assert config["alternative_labeling"] == "excluded"
        assert config["seeds"] == [1337, 2027, 31415, 424242, 8675309]
        assert (ROOT / config["source_reference"]).is_file()
        assert config["validation"] == {
            "policy": "chronological_training_only",
            "purge_events": 200,
            "embargo_events": 200,
            "normalization_fit": "training_only",
        }


def test_transformer_matrix_is_exactly_250_cells_per_model() -> None:
    specs = planned_run_specs(ROOT)
    for model in ("translob", "tlob"):
        selected = [spec for spec in specs if spec.model == model]
        assert len(selected) == 250
        assert len([spec for spec in selected if spec.setup == SETUP_ANCHORED_FORWARD]) == 225
        assert len([spec for spec in selected if spec.setup == SETUP_FIRST_SEVEN_FINAL_THREE]) == 25

    assert len(specs) == 1400
