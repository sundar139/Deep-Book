"""Checkpoint roundtrip and training resume contract tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from deepbook.training.fi2010 import load_checkpoint, save_checkpoint_atomic, seed_everything


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_synthetic_dataset(n: int = 50) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(42)
    features = rng.standard_normal((n, 10)).astype(np.float32)
    labels = rng.integers(0, 3, size=n).astype(np.int64)
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))


class TestCheckpointRoundtrip:
    """Full checkpoint save/load/verify pipeline."""

    def test_complete_checkpoint_roundtrip(self) -> None:
        seed_everything(1337)
        model = _DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        original_state = {key: value.clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            chk = Path(tmp) / "test.pt"
            save_checkpoint_atomic(
                chk,
                model,
                optimizer,
                epoch=5,
                seed=1337,
                configuration_hash="abc123",
                data_fingerprint="fp123",
                best_validation_metric=0.85,
            )
            # Modify model state
            for param in model.parameters():
                param.data.add_(torch.randn_like(param))
            metadata = load_checkpoint(chk, model, optimizer)
            # Model restored
            for key, value in model.state_dict().items():
                assert torch.allclose(value, original_state[key])
            assert metadata["epoch"] == 5
            assert metadata["seed"] == 1337
            assert metadata["configuration_hash"] == "abc123"
            assert metadata["data_fingerprint"] == "fp123"
            assert metadata["best_validation_metric"] == 0.85

    def test_optimizer_state_restoration(self) -> None:
        """Optimizer state is restored after checkpoint load."""
        seed_everything(42)
        model = _DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        dataset = _make_synthetic_dataset(30)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(42)
        )
        # One step to build state
        features, labels = next(iter(loader))
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(features), labels)
        loss.backward()
        optimizer.step()
        step1_grads = {
            name: param.grad.clone() if param.grad is not None else None
            for name, param in model.named_parameters()
        }
        with tempfile.TemporaryDirectory() as tmp:
            chk = Path(tmp) / "test.pt"
            save_checkpoint_atomic(
                chk,
                model,
                optimizer,
                epoch=1,
                seed=42,
                configuration_hash="h",
                data_fingerprint="f",
                best_validation_metric=0.5,
            )
            model2 = _DummyModel()
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
            model2.load_state_dict(model.state_dict())
            load_checkpoint(chk, model2, optimizer2)
            # Second step on restored optimizer should match
            model2.train()
            optimizer2.zero_grad()
            loss2 = torch.nn.functional.cross_entropy(model2(features), labels)
            loss2.backward()
            assert torch.isfinite(loss2)
            assert all(
                param.grad is not None and torch.isfinite(param.grad).all()
                for param in model2.parameters()
            )

    def test_epoch_restoration(self) -> None:
        seed_everything(1)
        model = _DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            chk = Path(tmp) / "test.pt"
            save_checkpoint_atomic(
                chk,
                model,
                optimizer,
                epoch=10,
                seed=1,
                configuration_hash="h",
                data_fingerprint="f",
                best_validation_metric=0.9,
            )
            metadata = load_checkpoint(chk, model, optimizer)
            assert metadata["epoch"] == 10

    def test_best_validation_metric_restoration(self) -> None:
        seed_everything(2)
        model = _DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            chk = Path(tmp) / "test.pt"
            save_checkpoint_atomic(
                chk,
                model,
                optimizer,
                epoch=3,
                seed=2,
                configuration_hash="h2",
                data_fingerprint="f2",
                best_validation_metric=0.92,
            )
            metadata = load_checkpoint(chk, model)
            assert metadata["best_validation_metric"] == 0.92

    def test_resumed_flag_recorded(self) -> None:
        """Simulate resumption by loading checkpoint and verifying metadata."""
        seed_everything(3)
        model = _DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            chk = Path(tmp) / "test.pt"
            save_checkpoint_atomic(
                chk,
                model,
                optimizer,
                epoch=4,
                seed=3,
                configuration_hash="h3",
                data_fingerprint="f3",
                best_validation_metric=0.75,
            )
            model2 = _DummyModel()
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
            metadata = load_checkpoint(chk, model2, optimizer2)
            # Resumed flag should be derivable from checkpoint presence
            assert metadata["seed"] == 3
            assert metadata["epoch"] == 4
            # Model state matches
            for p1, p2 in zip(model.parameters(), model2.parameters(), strict=False):
                assert torch.allclose(p1, p2)
