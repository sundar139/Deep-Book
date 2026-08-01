from __future__ import annotations

from deepbook.models.mlplob import MLPLOB
from deepbook.training.gates import tiny_batch_overfit_gate


def test_mlp_tiny_batch_overfit_gate() -> None:
    result = tiny_batch_overfit_gate(
        lambda: MLPLOB(input_shape=(4, 40), hidden_sizes=(32, 16)),
        input_shape=(4, 40),
        device="cpu",
        seed=1337,
        steps=80,
    )
    assert result["passed"]
    assert result["final_loss"] < result["initial_loss"]
