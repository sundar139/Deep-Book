from __future__ import annotations

import torch

from deepbook.models.deeplob import DeepLOB, parameter_count
from deepbook.models.mlplob import MLPLOB


def test_mlplob_forward_shape_and_finite_backward() -> None:
    model = MLPLOB(input_shape=(100, 40), hidden_sizes=(16, 8))
    x = torch.randn(4, 1, 100, 40)
    logits = model(x)
    assert logits.shape == (4, 3)
    loss = logits.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_deeplob_reference_shapes_and_parameter_count() -> None:
    model = DeepLOB()
    x = torch.randn(2, 1, 100, 40)
    logits, trace = model.forward_with_trace(x)
    assert logits.shape == (2, 3)
    assert trace["conv1"] == (2, 32, 94, 20)
    assert trace["conv2"] == (2, 32, 88, 10)
    assert trace["conv3"] == (2, 32, 82, 1)
    assert trace["inception"] == (2, 192, 82, 1)
    assert trace["lstm"] == (2, 82, 64)
    assert parameter_count(model) == 143907


def test_deeplob_cpu_forward_backward_is_finite() -> None:
    model = DeepLOB()
    x = torch.randn(2, 1, 100, 40)
    y = torch.tensor([0, 2])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
