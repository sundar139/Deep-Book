"""Tiny-batch capacity and training sanity gates."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from deepbook.training.fi2010 import seed_everything


def tiny_batch_overfit_gate(
    model_factory: Callable[[], nn.Module],
    *,
    input_shape: tuple[int, int],
    device: str,
    seed: int,
    steps: int = 50,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Require a fresh model to memorize a tiny deterministic three-class batch."""
    if steps <= 0 or not 0.0 < threshold <= 1.0:
        raise ValueError("steps must be positive and threshold must be in (0,1]")
    seed_everything(seed)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn((9, 1, input_shape[0], input_shape[1]), generator=generator)
    labels = torch.arange(9, dtype=torch.long) % 3
    model = model_factory().to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    started = time.perf_counter()
    model.train()
    initial_loss = float("nan")
    for iteration in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features.to(target_device))
        loss = torch.nn.functional.cross_entropy(logits, labels.to(target_device))
        if iteration == 0:
            initial_loss = float(loss.detach().cpu())
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits = model(features.to(target_device))
        final_loss = float(
            torch.nn.functional.cross_entropy(logits, labels.to(target_device)).cpu()
        )
        accuracy = float((logits.argmax(dim=1) == labels.to(target_device)).float().mean().cpu())
    return {
        "passed": bool(accuracy >= threshold),
        "steps": steps,
        "threshold": threshold,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_accuracy": accuracy,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(target_device),
    }
