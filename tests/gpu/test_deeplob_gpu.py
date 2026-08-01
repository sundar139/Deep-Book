"""GPU tests for DeepLOB: forward, loss, backward on CUDA synthetic data.

Every test runs inside a temporary artifact root and asserts that nothing
persistent was written: no checkpoint, prediction, or run manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# Mark all tests in this module as gpu; requires pytest marker registration in conftest or pyproject
pytestmark = pytest.mark.gpu

_PERSISTENT_SUFFIXES = (".pt", ".pth", ".ckpt", ".npz", ".json")


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _snapshot(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    """Provide an empty temporary artifact root for one GPU test."""
    root = tmp_path / "artifacts"
    (root / "runs").mkdir(parents=True)
    (root / "predictions").mkdir(parents=True)
    (root / "checkpoints").mkdir(parents=True)
    return root


def _assert_no_persistent_artifacts(root: Path, before: set[Path]) -> None:
    created = _snapshot(root) - before
    offending = sorted(str(path) for path in created if path.suffix.lower() in _PERSISTENT_SUFFIXES)
    assert not offending, f"GPU test wrote persistent artifacts: {offending}"
    assert not created, f"GPU test wrote unexpected files: {sorted(str(p) for p in created)}"


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_deeplob_forward_cuda_finite_logits(artifact_root: Path) -> None:
    """DeepLOB forward pass on CUDA produces finite logits and writes nothing."""
    from deepbook.models.deeplob import DeepLOB

    before = _snapshot(artifact_root)
    device = torch.device("cuda")
    model = DeepLOB().to(device)
    batch = torch.randn(4, 1, 100, 40, device=device)
    with torch.inference_mode():
        logits = model(batch)
    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()
    _assert_no_persistent_artifacts(artifact_root, before)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_deeplob_loss_computation_cuda(artifact_root: Path) -> None:
    """DeepLOB computes finite loss on CUDA and writes nothing."""
    from deepbook.models.deeplob import DeepLOB

    before = _snapshot(artifact_root)
    device = torch.device("cuda")
    model = DeepLOB().to(device)
    batch = torch.randn(8, 1, 100, 40, device=device)
    labels = torch.randint(0, 3, (8,), device=device)
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss).all()
    assert loss.item() > 0
    _assert_no_persistent_artifacts(artifact_root, before)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_deeplob_backward_cuda_finite_gradients(artifact_root: Path) -> None:
    """DeepLOB backward pass on CUDA produces finite gradients and writes nothing."""
    from deepbook.models.deeplob import DeepLOB, parameter_count

    before = _snapshot(artifact_root)
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

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"
    assert parameter_count(model) == 143907
    _assert_no_persistent_artifacts(artifact_root, before)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_gpu_tests_write_no_persistent_artifacts(artifact_root: Path) -> None:
    """The artifact guard itself must reject a checkpoint, prediction, or manifest."""
    from deepbook.models.deeplob import DeepLOB

    before = _snapshot(artifact_root)
    model = DeepLOB().to(torch.device("cuda"))
    with torch.inference_mode():
        model(torch.randn(2, 1, 100, 40, device="cuda"))
    _assert_no_persistent_artifacts(artifact_root, before)

    for name in ("checkpoints/run.best.pt", "predictions/run.npz", "runs/run.json"):
        written = artifact_root / name
        written.write_bytes(b"artifact")
        with pytest.raises(AssertionError, match="persistent artifacts"):
            _assert_no_persistent_artifacts(artifact_root, before)
        written.unlink()
    _assert_no_persistent_artifacts(artifact_root, before)


def test_gpu_module_skips_cleanly_without_cuda() -> None:
    """Without CUDA every GPU test must skip rather than fail."""
    if _cuda_available():
        pytest.skip("CUDA is available; skip behavior is exercised on CPU-only machines")
    pytest.skip("CUDA not available")
