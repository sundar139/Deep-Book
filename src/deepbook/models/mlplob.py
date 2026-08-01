"""Minimal MLP-LOB comparator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn


class MLPLOB(nn.Module):
    """Flattened 100-by-40 LOB MLP returning three logits."""

    def __init__(
        self,
        input_shape: tuple[int, int] = (100, 40),
        hidden_sizes: Sequence[int] = (128, 64),
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if len(input_shape) != 2 or any(size <= 0 for size in input_shape):
            raise ValueError("input_shape must contain two positive dimensions")
        layers: list[nn.Module] = [nn.Flatten()]
        input_size = input_shape[0] * input_shape[1]
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU()])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return three unnormalized class logits."""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return cast(torch.Tensor, self.network(x))
