"""Frozen architecture tests for the FI-2010 transformer extensions."""

from __future__ import annotations

import copy

import pytest
import torch

from deepbook.models.tlob import TLOB
from deepbook.models.tlob import parameter_count as tlob_parameter_count
from deepbook.models.translob import TransLOB
from deepbook.models.translob import parameter_count as translob_parameter_count

INPUT_SHAPE = (2, 1, 100, 40)


def _finite_gradients(model: torch.nn.Module) -> None:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert any(
        torch.linalg.vector_norm(gradient) > 0 for gradient in gradients if gradient is not None
    )


def test_translob_frozen_shape_count_and_backward() -> None:
    torch.manual_seed(7)
    model = TransLOB()
    x = torch.randn(INPUT_SHAPE)
    logits, trace = model.forward_with_trace(x)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert trace["convolution"] == (2, 100, 14)
    assert trace["encoded"] == (2, 100, 15)
    assert translob_parameter_count(model) == 101895
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
    _finite_gradients(model)


def test_translob_causal_convolution_and_attention_mask() -> None:
    torch.manual_seed(8)
    model = TransLOB().eval()
    x = torch.randn(INPUT_SHAPE)
    encoded, trace = model.encode_with_trace(x)
    future_changed = x.clone()
    future_changed[:, :, 70:, :] += 100.0
    changed_encoded = model.encode(future_changed)
    assert torch.allclose(encoded[:, :70], changed_encoded[:, :70], atol=1e-5, rtol=1e-5)
    for attention in trace["attention"]:
        upper = torch.triu(attention, diagonal=1)
        assert torch.count_nonzero(upper.abs() > 1e-7) == 0


def test_translob_seeded_initialization_and_serialization_round_trip() -> None:
    torch.manual_seed(9)
    first = TransLOB()
    torch.manual_seed(9)
    second = TransLOB()
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )
    state = copy.deepcopy(first.state_dict())
    restored = TransLOB()
    restored.load_state_dict(state)
    x = torch.randn(INPUT_SHAPE)
    first.eval()
    restored.eval()
    assert torch.equal(first(x), restored(x))


def test_tlob_frozen_shape_count_and_backward() -> None:
    torch.manual_seed(10)
    model = TLOB()
    x = torch.randn(2, 100, 40)
    logits, trace = model.forward_with_trace(x)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert trace["temporal_attention"]
    assert trace["spatial_attention"]
    assert all(item.shape[-2:] == (100, 100) for item in trace["temporal_attention"])
    assert [tuple(item.shape[-2:]) for item in trace["spatial_attention"]] == [
        (40, 40),
        (40, 40),
        (40, 40),
        (10, 10),
    ]
    assert trace["attention_order"] == ["temporal", "spatial"] * 4
    assert tlob_parameter_count(model) == 734478
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
    _finite_gradients(model)


def test_tlob_is_noncausal_and_dual_attention_axes_are_distinct() -> None:
    torch.manual_seed(11)
    model = TLOB().eval()
    x = torch.randn(2, 100, 40)
    _, trace = model.forward_with_trace(x)
    future_changed = x.clone()
    future_changed[:, 70:, :] += 100.0
    assert not torch.allclose(model(x), model(future_changed))
    spatial_changed = x.clone()
    spatial_changed[:, :, 20:] += 100.0
    assert not torch.allclose(model(x), model(spatial_changed))
    assert trace["temporal_attention"][0].shape[-1] == 100
    assert trace["spatial_attention"][0].shape[-1] == 40


def test_tlob_seeded_initialization_and_serialization_round_trip() -> None:
    torch.manual_seed(12)
    first = TLOB()
    torch.manual_seed(12)
    second = TLOB()
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )
    state = copy.deepcopy(first.state_dict())
    restored = TLOB()
    restored.load_state_dict(state)
    x = torch.randn(2, 100, 40)
    first.eval()
    restored.eval()
    assert torch.equal(first(x), restored(x))


@pytest.mark.parametrize("model_cls", [TransLOB, TLOB])
def test_models_reject_wrong_feature_width(model_cls: type[torch.nn.Module]) -> None:
    with pytest.raises(ValueError, match="40"):
        model_cls()(torch.randn(2, 1, 100, 39))


def test_frozen_models_reject_invalid_architecture_configuration() -> None:
    with pytest.raises(ValueError, match="two shared transformer blocks"):
        TransLOB(transformer_blocks=1)
    with pytest.raises(ValueError, match="positive"):
        TLOB(num_layers=0)
