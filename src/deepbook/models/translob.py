"""Source-backed TransLOB for the controlled FI-2010 comparison."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn.functional as functional
from torch import nn


class _CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=2, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv(functional.pad(x, (self.dilation, 0))))


class _MaskedSelfAttention(nn.Module):
    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError("model dimension must be divisible by attention heads")
        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads
        # ponytail: QKV uses bare weight — official repo has no bias.
        # The paper does not specify bias; aligning to repo.
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=False)
        # ponytail: W_O output projection retained per paper equation MultiHead(X)=concat(heads)W^O.
        # Official repo module omits it. Classified AMBIGUOUS_SOURCE_CONFLICT.
        self.output = nn.Linear(model_dim, model_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, steps, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, steps, self.model_dim)
        return self.output(output), weights


class _TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, heads: int, feedforward_multiplier: int) -> None:
        super().__init__()
        self.attention = _MaskedSelfAttention(model_dim, heads)
        self.norm1 = nn.LayerNorm(model_dim, eps=1e-5)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, model_dim * feedforward_multiplier),
            nn.ReLU(),
            nn.Linear(model_dim * feedforward_multiplier, model_dim),
        )
        self.norm2 = nn.LayerNorm(model_dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(x)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.feedforward(x))
        return x, weights


class TransLOB(nn.Module):
    """Causal dilated convolution plus shared masked Transformer blocks.

    Input is ``[batch, 1, 100, 40]`` or ``[batch, 100, 40]`` and output is
    three unnormalized class logits. The two transformer iterations share one
    block, matching the official FI-2010 source description.
    """

    def __init__(
        self,
        input_features: int = 40,
        sequence_length: int = 100,
        num_classes: int = 3,
        convolution_channels: int = 14,
        attention_heads: int = 3,
        transformer_blocks: int = 2,
        feedforward_multiplier: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_features <= 0 or sequence_length <= 0 or num_classes <= 0:
            raise ValueError("input dimensions and num_classes must be positive")
        if transformer_blocks != 2:
            raise ValueError("the frozen TransLOB contract uses two shared transformer blocks")
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.convolution_channels = convolution_channels
        self.convolutions = nn.ModuleList(
            [
                _CausalConv1d(
                    input_features if index == 0 else convolution_channels,
                    convolution_channels,
                    2**index,
                )
                for index in range(5)
            ]
        )
        self.convolution_norm = nn.LayerNorm(convolution_channels, eps=1e-5)
        position = torch.linspace(-1.0, 1.0, sequence_length).reshape(1, sequence_length, 1)
        self.register_buffer("position", position)
        model_dim = convolution_channels + 1
        self.transformer_block = _TransformerBlock(
            model_dim, attention_heads, feedforward_multiplier
        )
        self.classifier = nn.Sequential(
            nn.Linear(sequence_length * model_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d | nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError("TransLOB expects a singleton channel dimension")
            x = x[:, 0]
        if x.ndim != 3 or x.shape[1:] != (self.sequence_length, self.input_features):
            raise ValueError(
                f"TransLOB expects [B, {self.sequence_length}, {self.input_features}] "
                "or [B, 1, sequence, features]"
            )
        return x

    def encode_with_trace(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return causal contextual representations and attention diagnostics."""
        x = self._input(x)
        x = x.transpose(1, 2)
        for convolution in self.convolutions:
            x = torch.relu(convolution(x))
        x = x.transpose(1, 2)
        convolution_shape = (x.shape[0], self.sequence_length, self.convolution_channels)
        x = self.convolution_norm(x)
        position = self.get_buffer("position")
        x = torch.cat((x, position.expand(x.shape[0], -1, -1)), dim=-1)
        attentions: list[torch.Tensor] = []
        for _ in range(2):
            x, attention = self.transformer_block(x)
            attentions.append(attention)
        return x, {
            "convolution": convolution_shape,
            "encoded": tuple(x.shape),
            "attention": attentions,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the contextual representation for every event in the window."""
        encoded, _ = self.encode_with_trace(x)
        return encoded

    def forward_with_trace(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return logits and architecture diagnostics."""
        encoded, trace = self.encode_with_trace(x)
        logits = self.classifier(encoded.reshape(encoded.shape[0], -1))
        return logits, trace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return three unnormalized class logits."""
        logits, _ = self.forward_with_trace(x)
        return logits


def parameter_count(model: nn.Module) -> int:
    """Return trainable parameter count."""
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
