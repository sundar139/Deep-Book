"""Faithful PyTorch DeepLOB architecture from the frozen contract."""

from __future__ import annotations

import torch
from torch import nn


class DeepLOB(nn.Module):
    """Official-author PyTorch CNN/Inception/LSTM model returning three logits."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, num_classes)

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        trace: dict[str, tuple[int, ...]] = {}
        x = self.conv1(x)
        trace["conv1"] = tuple(x.shape)
        x = self.conv2(x)
        trace["conv2"] = tuple(x.shape)
        x = self.conv3(x)
        trace["conv3"] = tuple(x.shape)
        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)
        trace["inception"] = tuple(x.shape)
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], x.shape[1])
        hidden = x.new_zeros((1, x.shape[0], 64))
        cell = x.new_zeros((1, x.shape[0], 64))
        x, _ = self.lstm(x, (hidden, cell))
        trace["lstm"] = tuple(x.shape)
        return x, trace

    def forward_with_trace(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        """Return logits and major tensor shapes for architecture verification."""
        recurrent, trace = self._features(x)
        return self.fc1(recurrent[:, -1, :]), trace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return exactly three unnormalized class logits."""
        logits, _ = self.forward_with_trace(x)
        return logits


def parameter_count(model: nn.Module) -> int:
    """Return trainable parameter count."""
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
