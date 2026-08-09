"""CUDA preflight coverage for both new model implementations."""

from __future__ import annotations

import pytest
import torch

from deepbook.models import TLOB, TransLOB

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("model_cls", [TransLOB, TLOB])
def test_transformer_forward_backward_cuda_without_artifacts(
    model_cls: type[torch.nn.Module], tmp_path
) -> None:
    model = model_cls().to("cuda")
    x = torch.randn(2, 1, 100, 40, device="cuda")
    labels = torch.tensor([0, 2], device="cuda")
    logits = model(x)
    assert logits.device.type == "cuda"
    assert logits.shape == (2, 3)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert list(tmp_path.iterdir()) == []
