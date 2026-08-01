"""Best-versus-last checkpoint semantics and bit-exact interruption/resume.

Uninterrupted epoch ``e`` and resumed epoch ``e`` must consume the identical
sample order, so an interrupted run that continues to the same final epoch has to
land on exactly the same parameters, optimizer state, and validation predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from deepbook.models.mlplob import MLPLOB
from deepbook.training.fi2010 import (
    SegmentedWindowDataset,
    epoch_shuffle_seed,
    load_training_state,
    save_checkpoint_atomic,
    seed_everything,
)
from deepbook.training.loop import fit_torch_classifier, predict_raw

SEQUENCE_LENGTH = 16
PARAMETER_TOLERANCE = 0.0
COMMON = {
    "seed": 1337,
    "patience": 99,
    "batch_size": 8,
    "learning_rate": 5e-4,
    "device": "cpu",
    "configuration_hash": "configuration",
    "data_fingerprint": "fingerprint",
    "protocol_hash": "protocol",
}


def _learnable(observations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a separable synthetic segment so validation improves across epochs."""
    rng = np.random.default_rng(seed)
    classes = rng.integers(0, 3, observations)
    lob = (rng.standard_normal((40, observations)) * 0.3 + classes * 2.0).astype(np.float32)
    labels = np.tile(classes + 1, (5, 1)).astype(np.int8)
    return lob, labels


def _dataset(observations: int, seed: int) -> SegmentedWindowDataset:
    lob, labels = _learnable(observations, seed)
    return SegmentedWindowDataset([(lob, labels)], 0, SEQUENCE_LENGTH)


def _model() -> MLPLOB:
    seed_everything(11)
    return MLPLOB(input_shape=(SEQUENCE_LENGTH, 40), hidden_sizes=(12, 6))


def _parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten() for parameter in model.parameters()])


def _optimizer_signature(state: dict) -> list[float]:
    values: list[float] = []
    for entry in sorted(state["state"], key=str):
        slot = state["state"][entry]
        values.append(float(slot["step"]) if "step" in slot else -1.0)
        for key in ("exp_avg", "exp_avg_sq"):
            if key in slot:
                values.append(float(torch.as_tensor(slot[key]).sum()))
    return values


def _fit(model: torch.nn.Module, tmp_path: Path, tag: str, epochs: int, **extra: object):
    return fit_torch_classifier(
        model,
        _dataset(400, 0),
        _dataset(200, 1),
        max_epochs=epochs,
        best_checkpoint_path=tmp_path / f"{tag}.best.pt",
        last_checkpoint_path=tmp_path / f"{tag}.last.pt",
        **COMMON,  # type: ignore[arg-type]
        **extra,  # type: ignore[arg-type]
    )


# --- Shuffle order ----------------------------------------------------------


def test_epoch_shuffle_seed_depends_only_on_base_seed_and_epoch() -> None:
    assert epoch_shuffle_seed(1337, 4) == epoch_shuffle_seed(1337, 4)
    assert epoch_shuffle_seed(1337, 4) != epoch_shuffle_seed(1337, 5)
    assert epoch_shuffle_seed(1337, 4) != epoch_shuffle_seed(2027, 4)
    with pytest.raises(ValueError, match="epoch must be positive"):
        epoch_shuffle_seed(1337, 0)


# --- Best versus last -------------------------------------------------------


def test_best_and_last_checkpoints_are_separate_files_with_distinct_roles(
    tmp_path: Path,
) -> None:
    result = _fit(_model(), tmp_path, "run", 6)
    best = tmp_path / "run.best.pt"
    last = tmp_path / "run.last.pt"
    assert best.is_file() and last.is_file()

    best_payload = torch.load(best, map_location="cpu", weights_only=False)
    last_payload = torch.load(last, map_location="cpu", weights_only=False)
    assert best_payload["checkpoint_kind"] == "best"
    assert last_payload["checkpoint_kind"] == "last"
    assert last_payload["completed_epoch"] == result.actual_epochs_completed
    assert last_payload["next_epoch"] == result.actual_epochs_completed + 1
    assert last_payload["best_epoch"] == result.best_epoch
    assert best_payload["epoch"] == result.best_epoch
    for key in ("python_rng", "numpy_rng", "torch_rng", "scaler_state", "scheduler_state"):
        assert key in last_payload


def test_resume_refuses_the_stale_best_checkpoint(tmp_path: Path) -> None:
    _fit(_model(), tmp_path, "run", 4)
    with pytest.raises(ValueError, match="last-state checkpoint"):
        _fit(_model(), tmp_path, "resumed", 6, resume_from=tmp_path / "run.best.pt")


def test_resume_rejects_a_start_epoch_beyond_the_maximum(tmp_path: Path) -> None:
    _fit(_model(), tmp_path, "run", 6)
    with pytest.raises(ValueError, match="exceeds max_epochs"):
        _fit(_model(), tmp_path, "short", 6, resume_from=tmp_path / "run.last.pt")


def test_resume_rejects_mismatched_provenance(tmp_path: Path) -> None:
    _fit(_model(), tmp_path, "run", 3)
    for field, message in (
        ("configuration_hash", "configuration hash mismatch"),
        ("data_fingerprint", "data fingerprint mismatch"),
        ("protocol_hash", "protocol hash mismatch"),
    ):
        options = {**COMMON, field: "tampered"}
        with pytest.raises(ValueError, match=message):
            fit_torch_classifier(
                _model(),
                _dataset(400, 0),
                _dataset(200, 1),
                max_epochs=6,
                best_checkpoint_path=tmp_path / "x.best.pt",
                last_checkpoint_path=tmp_path / "x.last.pt",
                resume_from=tmp_path / "run.last.pt",
                **options,  # type: ignore[arg-type]
            )


def test_load_training_state_rejects_a_best_checkpoint(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "best.pt"
    save_checkpoint_atomic(
        path,
        model,
        optimizer,
        epoch=3,
        seed=1337,
        configuration_hash="c",
        data_fingerprint="d",
        best_validation_metric=0.5,
    )
    with pytest.raises(ValueError, match="last-state checkpoint"):
        load_training_state(path, _model(), torch.optim.Adam(_model().parameters(), lr=1e-3))


# --- Deterministic resume ---------------------------------------------------


def _compare_uninterrupted_with_resumed(tmp_path: Path, total: int, interrupt_at: int) -> None:
    uninterrupted_model = _model()
    uninterrupted = _fit(uninterrupted_model, tmp_path, "full", total)

    interrupted_model = _model()
    partial = _fit(interrupted_model, tmp_path, "part", interrupt_at)
    assert partial.actual_epochs_completed == interrupt_at

    resumed_model = _model()
    resumed = _fit(resumed_model, tmp_path, "part", total, resume_from=tmp_path / "part.last.pt")

    assert resumed.actual_epochs_completed == uninterrupted.actual_epochs_completed
    assert resumed.best_epoch == uninterrupted.best_epoch
    assert resumed.patience_counter == uninterrupted.patience_counter
    assert resumed.best_validation_metric == pytest.approx(
        uninterrupted.best_validation_metric, abs=1e-12
    )
    assert resumed.final_training_loss == pytest.approx(uninterrupted.final_training_loss, abs=1e-9)

    left = _parameters(uninterrupted_model)
    right = _parameters(resumed_model)
    difference = float((left - right).abs().max())
    assert difference <= PARAMETER_TOLERANCE, f"max parameter difference {difference}"

    validation = _dataset(200, 1)
    device = torch.device("cpu")
    _, uninterrupted_predictions, uninterrupted_probabilities, _ = predict_raw(
        uninterrupted_model, validation, device, 8
    )
    _, resumed_predictions, resumed_probabilities, _ = predict_raw(
        resumed_model, validation, device, 8
    )
    assert np.array_equal(uninterrupted_predictions, resumed_predictions)
    assert np.allclose(uninterrupted_probabilities, resumed_probabilities, atol=0.0, rtol=0.0)

    last = torch.load(tmp_path / "part.last.pt", map_location="cpu", weights_only=False)
    reference = torch.load(tmp_path / "full.last.pt", map_location="cpu", weights_only=False)
    assert _optimizer_signature(last["optimizer"]) == pytest.approx(
        _optimizer_signature(reference["optimizer"]), abs=1e-9
    )


def test_interrupted_run_resumes_to_the_uninterrupted_result(tmp_path: Path) -> None:
    _compare_uninterrupted_with_resumed(tmp_path, total=6, interrupt_at=3)


def test_resume_is_exact_when_the_best_epoch_precedes_the_interruption(
    tmp_path: Path,
) -> None:
    """The last-state checkpoint, not the stale best one, must drive continuation."""
    model = _model()
    partial = _fit(model, tmp_path, "gap", 5)
    assert partial.best_epoch <= partial.actual_epochs_completed
    last = torch.load(tmp_path / "gap.last.pt", map_location="cpu", weights_only=False)
    best = torch.load(tmp_path / "gap.best.pt", map_location="cpu", weights_only=False)
    if last["best_epoch"] < last["completed_epoch"]:
        differing = [
            key
            for key in last["model"]
            if not torch.equal(last["model"][key].cpu(), best["model"][key].cpu())
        ]
        assert differing, "last state should differ from the stale best state"
    _compare_uninterrupted_with_resumed(tmp_path, total=8, interrupt_at=5)
