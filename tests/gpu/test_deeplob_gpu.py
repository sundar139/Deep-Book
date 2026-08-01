"""GPU tests for DeepLOB: forward, loss, backward on CUDA synthetic data."""

from __future__ import annotations

import pytest
import torch

# Mark all tests in this module as gpu; requires pytest marker registration in conftest or pyproject
pytestmark = pytest.mark.gpu


def _cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_deeplob_forward_cuda_finite_logits() -> None:
    """DeepLOB forward pass on CUDA produces finite logits."""
    from deepbook.models.deeplob import DeepLOB

    device = torch.device("cuda")
    model = DeepLOB().to(device)
    batch = torch.randn(4, 1, 100, 40, device=device)
    with torch.inference_mode():
        logits = model(batch)
    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_deeplob_loss_computation_cuda() -> None:
    """DeepLOB computes finite loss on CUDA."""
    from deepbook.models.deeplob import DeepLOB

    device = torch.device("cuda")
    model = DeepLOB().to(device)
    batch = torch.randn(8, 1, 100, 40, device=device)
    labels = torch.randint(0, 3, (8,), device=device)
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss).all()
    assert loss.item() > 0


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_deeplob_backward_cuda_finite_gradients() -> None:
    """DeepLOB backward pass on CUDA produces finite gradients."""
    from deepbook.models.deeplob import DeepLOB, parameter_count

    device = torch.device("cuda")
    model = DeepLOB().to(device)
    batch = torch.randn(4, 1, 100, 40, device=device)
    labels = torch.randint(0, 3, (4,), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()

    # Verify all gradients are finite
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"

    # No persistent checkpoint written
    assert parameter_count(model) == 143907
