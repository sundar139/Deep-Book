from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from deepbook.models.mlplob import MLPLOB
from deepbook.training.fi2010 import SegmentedWindowDataset
from deepbook.training.loop import fit_torch_classifier


def test_torch_fit_has_validation_metrics_and_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    lob = rng.normal(size=(40, 40)).astype(np.float32)
    source_labels = np.vstack([np.ones(40, dtype=np.int8), np.ones((4, 40), dtype=np.int8)])
    source_labels[0, 20:] = 2
    train = SegmentedWindowDataset([(lob[:, :25], source_labels[:, :25])], 0, 3)
    validation = SegmentedWindowDataset([(lob[:, 25:], source_labels[:, 25:])], 0, 3)
    result = fit_torch_classifier(
        MLPLOB(input_shape=(3, 40), hidden_sizes=(8,)),
        train,
        validation,
        seed=1337,
        max_epochs=2,
        patience=2,
        batch_size=4,
        learning_rate=1e-3,
        device="cpu",
        checkpoint_path=tmp_path / "best.pt",
    )

    assert result.best_epoch in {1, 2}
    assert {"accuracy", "macro_f1", "mcc", "nll", "brier", "ece"}.issubset(
        result.validation_metrics
    )
    assert result.checkpoint_round_trip
    assert result.training_seconds >= 0.0
    assert result.peak_gpu_memory_bytes == 0
    assert torch.isfinite(torch.tensor(result.validation_metrics["nll"]))
