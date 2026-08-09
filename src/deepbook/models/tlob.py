"""Source-backed TLOB dual-attention model for FI-2010."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn


class _BilinearNormalization(nn.Module):
    """PyTorch translation of the official TLOB BiN layer."""

    def __init__(self, features: int, steps: int) -> None:
        super().__init__()
        self.features = features
        self.steps = steps
        self.bias_time = nn.Parameter(torch.zeros(steps, 1))
        self.scale_time = nn.Parameter(torch.empty(steps, 1))
        self.bias_feature = nn.Parameter(torch.zeros(features, 1))
        self.scale_feature = nn.Parameter(torch.empty(features, 1))
        self.time_weight = nn.Parameter(torch.full((1,), 0.5))
        self.feature_weight = nn.Parameter(torch.full((1,), 0.5))
        nn.init.xavier_normal_(self.scale_time)
        nn.init.xavier_normal_(self.scale_feature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_ones = x.new_ones(self.steps, 1)
        time_mean = x.mean(dim=2, keepdim=True)
        time_std = x.std(dim=2, keepdim=True).clamp_min(1e-4)
        centered_time = x - time_mean @ time_ones.transpose(0, 1)
        time_normalized = centered_time / (time_std @ time_ones.transpose(0, 1))
        time_scale = self.scale_feature @ time_ones.transpose(0, 1)
        time_bias = self.bias_feature @ time_ones.transpose(0, 1)
        normalized_time = time_scale * time_normalized + time_bias

        feature_ones = x.new_ones(self.features, 1)
        feature_mean = x.mean(dim=1, keepdim=True)
        feature_std = x.std(dim=1, keepdim=True).clamp_min(1e-4)
        centered_feature = (x - feature_mean) / feature_std
        feature_scale = feature_ones @ self.scale_time.transpose(0, 1)
        feature_bias = feature_ones @ self.bias_time.transpose(0, 1)
        normalized_feature = feature_scale * centered_feature + feature_bias
        return self.time_weight * normalized_feature + self.feature_weight * normalized_time


class _AxisMLP(nn.Module):
    def __init__(self, start_dim: int, final_dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(final_dim)
        self.fc = nn.Linear(start_dim, start_dim * 4)
        self.fc2 = nn.Linear(start_dim * 4, final_dim)
        self.gelu = nn.GELU()
        self.start_dim = start_dim
        self.final_dim = final_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc(x)
        x = self.gelu(x)
        x = self.fc2(x)
        if x.shape[-1] == residual.shape[-1]:
            x = x + residual
        return cast(torch.Tensor, self.gelu(self.layer_norm(x)))


class _AxisAttention(nn.Module):
    def __init__(self, axis_dim: int, num_heads: int, final_dim: int) -> None:
        super().__init__()
        self.axis_dim = axis_dim
        self.final_dim = final_dim
        self.num_heads = num_heads
        self.qkv = nn.ModuleList([nn.Linear(axis_dim, axis_dim * num_heads) for _ in range(3)])
        self.attention = nn.MultiheadAttention(axis_dim * num_heads, num_heads, batch_first=True)
        self.output = nn.Linear(axis_dim * num_heads, axis_dim)
        self.norm = nn.LayerNorm(axis_dim)
        self.mlp = _AxisMLP(axis_dim, final_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        query, key, value = (projection(x) for projection in self.qkv)
        attended, weights = self.attention(
            query,
            key,
            value,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.output(attended)
        x = self.norm(x + residual)
        return self.mlp(x), weights


class TLOB(nn.Module):
    """TLOB's temporal-then-spatial attention blocks and classifier.

    The FI-2010 DeepBook adaptation keeps the existing 100-event, 40-feature
    raw LOB window. It therefore uses the source repository's book-only
    ``hidden_dim=40`` interpretation rather than its all-feature 144-column
    training path. Attention is intentionally non-causal, as specified by the
    source architecture.
    """

    def __init__(
        self,
        input_features: int = 40,
        sequence_length: int = 100,
        hidden_dim: int = 40,
        num_layers: int = 4,
        num_heads: int = 1,
        is_sin_emb: bool = True,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if input_features <= 0 or sequence_length <= 0 or hidden_dim <= 0:
            raise ValueError("input dimensions must be positive")
        if num_layers <= 0 or num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive")
        if hidden_dim % 4 or sequence_length % 4:
            raise ValueError("hidden_dim and sequence_length must be divisible by four")
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.norm_layer = _BilinearNormalization(input_features, sequence_length)
        self.embedding = nn.Linear(input_features, hidden_dim)
        if is_sin_emb:
            self.register_buffer("position", self._sinusoidal(sequence_length, hidden_dim))
        else:
            self.position = nn.Parameter(torch.randn(1, sequence_length, hidden_dim))
        self.temporal_layers = nn.ModuleList()
        self.spatial_layers = nn.ModuleList()
        for index in range(num_layers):
            temporal_final = hidden_dim if index != num_layers - 1 else hidden_dim // 4
            spatial_final = sequence_length if index != num_layers - 1 else sequence_length // 4
            self.temporal_layers.append(_AxisAttention(hidden_dim, num_heads, temporal_final))
            self.spatial_layers.append(_AxisAttention(sequence_length, num_heads, spatial_final))
        total_dim = (hidden_dim // 4) * (sequence_length // 4)
        final_layers: list[nn.Module] = []
        while total_dim > 128:
            final_layers.extend([nn.Linear(total_dim, total_dim // 4), nn.GELU()])
            total_dim //= 4
        final_layers.append(nn.Linear(total_dim, num_classes))
        self.final_layers = nn.ModuleList(final_layers)

    @staticmethod
    def _sinusoidal(steps: int, dimension: int) -> torch.Tensor:
        positions = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
        indices = torch.arange(0, dimension, 2, dtype=torch.float32)
        denominator = torch.pow(torch.tensor(10000.0), indices / dimension)
        encoding = torch.zeros(steps, dimension)
        encoding[:, 0::2] = torch.sin(positions / denominator)
        encoding[:, 1::2] = torch.cos(positions / denominator)
        return encoding.unsqueeze(0)

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError("TLOB expects a singleton channel dimension")
            x = x[:, 0]
        if x.ndim != 3 or x.shape[1:] != (self.sequence_length, self.input_features):
            raise ValueError(
                f"TLOB expects [B, {self.sequence_length}, {self.input_features}] "
                "or [B, 1, sequence, features]"
            )
        return x

    def forward_with_trace(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return logits and temporal/spatial attention diagnostics."""
        x = self._input(x)
        x = self.norm_layer(x.transpose(1, 2)).transpose(1, 2)
        x = self.embedding(x) + self.position
        temporal_attention: list[torch.Tensor] = []
        spatial_attention: list[torch.Tensor] = []
        attention_order: list[str] = []
        for temporal, spatial in zip(self.temporal_layers, self.spatial_layers, strict=True):
            x, temporal_weights = temporal(x)
            temporal_attention.append(temporal_weights)
            attention_order.append("temporal")
            x, spatial_weights = spatial(x.transpose(1, 2))
            spatial_attention.append(spatial_weights)
            attention_order.append("spatial")
            x = x.transpose(1, 2)
        x = x.transpose(1, 2).reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x, {
            "temporal_attention": temporal_attention,
            "spatial_attention": spatial_attention,
            "attention_order": attention_order,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return three unnormalized class logits."""
        logits, _ = self.forward_with_trace(x)
        return logits


def parameter_count(model: nn.Module) -> int:
    """Return trainable parameter count."""
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
