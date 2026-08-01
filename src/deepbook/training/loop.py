"""Deterministic, validation-selected Torch training primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from deepbook.evaluation.classification import classification_metrics
from deepbook.training.fi2010 import load_checkpoint, save_checkpoint_atomic, seed_everything


@dataclass(frozen=True)
class TorchFitResult:
    """Auditable outcome of one validation-selected training run."""

    best_epoch: int
    validation_metrics: dict[str, Any]
    training_seconds: float
    inference_latency_ms_per_sample: float
    peak_gpu_memory_bytes: int
    checkpoint_round_trip: bool


def predict_raw(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return raw predictions, labels, probabilities, and per-sample latency."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    true_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for features, labels in loader:
            logits = model(features.to(device))
            batch_probabilities = torch.softmax(logits, dim=1)
            true_batches.append(labels.numpy())
            prediction_batches.append(batch_probabilities.argmax(dim=1).cpu().numpy())
            probability_batches.append(batch_probabilities.cpu().numpy())
    elapsed = time.perf_counter() - started
    true = np.concatenate(true_batches)
    predictions = np.concatenate(prediction_batches)
    probabilities = np.concatenate(probability_batches)
    latency = elapsed * 1000.0 / max(1, len(true))
    return true, predictions, probabilities, latency


def evaluate_torch_classifier(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str,
    batch_size: int,
) -> tuple[dict[str, Any], float]:
    """Evaluate a frozen model once and return metrics plus per-sample latency."""
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(target_device)
    model.eval()
    true, predictions, probabilities, latency = predict_raw(
        model, dataset, target_device, batch_size
    )
    return classification_metrics(true, predictions, probabilities), latency


def fit_torch_classifier(
    model: nn.Module,
    train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    validation_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    checkpoint_path: Path | None = None,
    configuration_hash: str = "",
    data_fingerprint: str = "",
    protocol_hash: str = "",
    resume_from: Path | None = None,
) -> TorchFitResult:
    """Fit using training-only data and select checkpoint by validation macro-F1.

    When resume_from is set, restore full state from the checkpoint and
    continue training from the next epoch.
    """
    if len(train_dataset) == 0 or len(validation_dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("training and validation datasets must both be non-empty")
    if max_epochs <= 0 or patience <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("training hyperparameters must be positive")
    seed_everything(seed)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 1
    stale_epochs = 0
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    if resume_from is not None and resume_from.is_file():
        metadata = load_checkpoint(resume_from, model, optimizer)
        if metadata.get("configuration_hash", "") != configuration_hash:
            raise ValueError("resume rejected: configuration hash mismatch")
        if metadata.get("data_fingerprint", "") != data_fingerprint:
            raise ValueError("resume rejected: data fingerprint mismatch")
        if metadata.get("protocol_hash", "") != protocol_hash:
            raise ValueError("resume rejected: protocol hash mismatch")
        start_epoch = int(metadata["epoch"]) + 1
        best_epoch = int(metadata.get("best_epoch", metadata["epoch"]))
        best_score = float(metadata.get("best_validation_metric", float("-inf")))
        stale_epochs = int(metadata.get("patience_counter", 0))
        # Restore best state from the checkpoint model
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        if start_epoch > max_epochs:
            # Already completed; return current state
            pass

    generator = torch.Generator().manual_seed(seed + start_epoch - 1)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    started = time.perf_counter()

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(target_device))
            loss = torch.nn.functional.cross_entropy(logits, labels.to(target_device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        model.eval()
        validation_true, validation_predictions, validation_probabilities, _ = predict_raw(
            model, validation_dataset, target_device, batch_size
        )
        validation_metrics = classification_metrics(
            validation_true, validation_predictions, validation_probabilities
        )
        score = float(validation_metrics["macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            if checkpoint_path is not None:
                save_checkpoint_atomic(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    seed=seed,
                    configuration_hash=configuration_hash,
                    data_fingerprint=data_fingerprint,
                    best_validation_metric=score,
                    best_epoch=best_epoch,
                    patience_counter=stale_epochs,
                    protocol_hash=protocol_hash,
                )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None or best_epoch == 0:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.eval()
    validation_features, _ = validation_dataset[0]
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer)
    else:
        model.load_state_dict(best_state)
    expected_output = model(validation_features.unsqueeze(0).to(target_device)).detach().cpu()
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer)
    else:
        model.load_state_dict(best_state)
    model.to(target_device)
    actual_output = model(validation_features.unsqueeze(0).to(target_device)).detach().cpu()
    checkpoint_round_trip = bool(
        torch.allclose(expected_output, actual_output, atol=1e-6, rtol=1e-5)
    )
    if not checkpoint_round_trip:
        raise RuntimeError("checkpoint round-trip changed model output")
    model.eval()
    true, predictions, probabilities, latency = predict_raw(
        model, validation_dataset, target_device, batch_size
    )
    metrics = classification_metrics(true, predictions, probabilities)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(target_device)) if target_device.type == "cuda" else 0
    )
    return TorchFitResult(
        best_epoch=best_epoch,
        validation_metrics=metrics,
        training_seconds=time.perf_counter() - started,
        inference_latency_ms_per_sample=latency,
        peak_gpu_memory_bytes=peak_memory,
        checkpoint_round_trip=checkpoint_round_trip,
    )
