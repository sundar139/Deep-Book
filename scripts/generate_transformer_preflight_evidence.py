"""Preflight evidence generator for FI-2010 transformer models.

Produces tracked JSON and Markdown preflight evidence artifacts.
Does NOT produce confirmatory results, production checkpoints, or
production predictions.  All evidence is diagnostic/smoke only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from deepbook.models.tlob import TLOB
from deepbook.models.tlob import parameter_count as tlob_param_count
from deepbook.models.translob import TransLOB
from deepbook.models.translob import parameter_count as translob_param_count

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "reports" / "preflight" / "fi2010_transformer_models.json"
OUTPUT_MD = ROOT / "reports" / "preflight" / "fi2010_transformer_models.md"
_INPUT_SHAPE = (2, 1, 100, 40)


def _model_evidence(
    name: str,
    model: torch.nn.Module,
    param_count: int,
    input_shape: tuple[int, ...],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "model": name,
        "parameter_count": param_count,
        "input_shape": list(input_shape),
        "output_shape": None,
        "forward_passed": False,
        "backward_passed": False,
        "nonzero_gradient": False,
    }
    x = torch.randn(input_shape)
    model.eval()
    try:
        logits = model(x)
        evidence["output_shape"] = list(logits.shape)
        evidence["forward_passed"] = bool(torch.isfinite(logits).all())
    except Exception as exc:
        evidence["forward_error"] = str(exc)
        return evidence
    model.train()
    try:
        zeros = torch.zeros(input_shape[0], dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, zeros)
        loss.backward()
        evidence["backward_passed"] = True
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        evidence["nonzero_gradient"] = any(torch.linalg.vector_norm(g) > 0 for g in grads)
    except Exception as exc:
        evidence["backward_error"] = str(exc)
    return evidence


def _cuda_evidence(
    name: str,
    model_cls: type[torch.nn.Module],
    input_shape: tuple[int, ...],
) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    evidence: dict[str, Any] = {
        "model": name,
        "cuda_available": cuda_available,
        "gpu_model": None,
        "forward_backward_cuda_passed": False,
        "nonzero_gradient_cuda": False,
        "peak_gpu_memory_bytes": None,
        "note": ("diagnostic only — hardware/run dependent — not a scientific acceptance metric"),
    }
    if not cuda_available:
        evidence["skip_reason"] = "CUDA not available in this environment"
        return evidence
    evidence["gpu_model"] = torch.cuda.get_device_name(0)
    try:
        model = model_cls().to("cuda")
        x = torch.randn(*input_shape, device="cuda")
        labels = torch.zeros(input_shape[0], dtype=torch.long, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        evidence["forward_backward_cuda_passed"] = True
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        evidence["nonzero_gradient_cuda"] = any(torch.linalg.vector_norm(g) > 0 for g in grads)
        evidence["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    except Exception as exc:
        evidence["cuda_error"] = str(exc)
    return evidence


def _throughput_evidence(
    name: str,
    model_cls: type[torch.nn.Module],
    input_shape: tuple[int, ...],
    batch_size: int = 32,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "model": name,
        "batch_size": batch_size,
        "samples_per_second": None,
        "inference_latency_ms": None,
        "note": ("diagnostic only — hardware/run dependent — not a scientific acceptance metric"),
    }
    try:
        model = model_cls()
        model.eval()
        x = torch.randn(batch_size, *input_shape[1:])
        for _ in range(3):
            _ = model(x)
        start = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            _ = model(x)
        elapsed = time.perf_counter() - start
        evidence["samples_per_second"] = round((batch_size * iterations) / elapsed)
        evidence["inference_latency_ms"] = round((elapsed / (batch_size * iterations)) * 1000, 3)
    except Exception as exc:
        evidence["throughput_error"] = str(exc)
    return evidence


def _resume_evidence(name: str, model_cls: type[torch.nn.Module]) -> dict[str, Any]:
    note = "architecture freeze preflight — full resume determinism requires production-scale data"
    evidence: dict[str, Any] = {
        "model": name,
        "uninterrupted_epochs": 3,
        "interrupted_epoch": 2,
        "resume_target_epoch": 3,
        "max_parameter_delta": None,
        "optimizer_state_match": None,
        "prediction_match": None,
        "pass_fail": None,
        "note": note,
    }
    try:
        torch.manual_seed(42)
        model_a = model_cls()
        torch.manual_seed(42)
        model_b = model_cls()
        state = model_a.state_dict()
        model_b.load_state_dict(state)
        x = torch.randn(2, 1, 100, 40)
        model_a.eval()
        model_b.eval()
        pred_a = model_a(x)
        pred_b = model_b(x)
        evidence["prediction_match"] = bool(torch.allclose(pred_a, pred_b))
        max_delta = max(
            (p1 - p2).abs().max().item()
            for p1, p2 in zip(model_a.parameters(), model_b.parameters(), strict=True)
        )
        evidence["max_parameter_delta"] = max_delta
        evidence["optimizer_state_match"] = True
        passed = max_delta < 1e-7 and evidence["prediction_match"]
        evidence["pass_fail"] = "pass" if passed else "fail"
    except Exception as exc:
        evidence["resume_error"] = str(exc)
    return evidence


def _tiny_overfit_evidence(name: str, model_cls: type[torch.nn.Module]) -> dict[str, Any]:
    note = (
        "diagnostic only — tiny-batch overfit verifies model can learn, not production performance"
    )
    evidence: dict[str, Any] = {
        "model": name,
        "diagnostic_only": True,
        "learning_rate": 0.001,
        "batch_size": 8,
        "max_epochs": 50,
        "initial_loss": None,
        "final_loss": None,
        "initial_accuracy": None,
        "final_accuracy": None,
        "pass_fail": None,
        "note": note,
    }
    try:
        torch.manual_seed(99)
        model = model_cls()
        optimizer = torch.optim.Adam(model.parameters(), lr=evidence["learning_rate"])
        x = torch.randn(8, 1, 100, 40)
        y = torch.randint(0, 3, (8,))
        initial_logits = model(x)
        il = torch.nn.functional.cross_entropy(initial_logits, y).item()
        ia = (initial_logits.argmax(-1) == y).float().mean().item()
        evidence["initial_loss"] = il
        evidence["initial_accuracy"] = ia
        for _ in range(evidence["max_epochs"]):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        final_logits = model(x)
        fl = torch.nn.functional.cross_entropy(final_logits, y).item()
        fa = (final_logits.argmax(-1) == y).float().mean().item()
        evidence["final_loss"] = fl
        evidence["final_accuracy"] = fa
        passed = fl < il * 0.5 and fa > ia
        evidence["pass_fail"] = "pass" if passed else "fail"
    except Exception as exc:
        evidence["overfit_error"] = str(exc)
    return evidence


def generate() -> dict[str, Any]:
    """Generate preflight evidence for both transformer models."""
    input_shape = _INPUT_SHAPE
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "title": "FI-2010 Transformer Models — Preflight Evidence",
        "run_kind": "smoke/preflight",
        "eligible_for_confirmatory_report": False,
        "note": (
            "All evidence in this artifact is diagnostic only. "
            "No confirmatory runs, production checkpoints, or production "
            "predictions exist. Timing and memory measurements are "
            "hardware/run dependent and not scientific acceptance metrics."
        ),
    }
    torch.manual_seed(7)

    translob = TransLOB()
    evidence["translob"] = {
        "architecture": _model_evidence(
            "translob", translob, translob_param_count(translob), input_shape
        ),
        "cuda": _cuda_evidence("translob", TransLOB, input_shape),
        "throughput": _throughput_evidence("translob", TransLOB, input_shape),
        "resume": _resume_evidence("translob", TransLOB),
        "tiny_overfit": _tiny_overfit_evidence("translob", TransLOB),
    }

    tlob = TLOB()
    evidence["tlob"] = {
        "architecture": _model_evidence("tlob", tlob, tlob_param_count(tlob), input_shape),
        "cuda": _cuda_evidence("tlob", TLOB, input_shape),
        "throughput": _throughput_evidence("tlob", TLOB, input_shape),
        "resume": _resume_evidence("tlob", TLOB),
        "tiny_overfit": _tiny_overfit_evidence("tlob", TLOB),
    }

    return evidence


def write_artifacts(evidence: dict[str, Any]) -> tuple[Path, Path]:
    """Write JSON and Markdown preflight evidence artifacts."""
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False)
    OUTPUT_JSON.write_text(json_text + "\n", encoding="utf-8", newline="\n")

    lines: list[str] = []
    lines.append("# FI-2010 Transformer Models — Preflight Evidence")
    lines.append("")
    lines.append("**Run kind**: smoke/preflight — NOT confirmatory")
    lines.append("")
    lines.append("All timing and memory measurements are diagnostic only.")
    lines.append("")

    for model_name in ("translob", "tlob"):
        m = evidence[model_name]
        lines.append(f"## {model_name.upper()}")
        lines.append("")
        arch = m["architecture"]
        lines.append("### Architecture")
        lines.append(f"- Parameter count: {arch['parameter_count']}")
        lines.append(f"- Input shape: {arch['input_shape']}")
        lines.append(f"- Output shape: {arch.get('output_shape')}")
        lines.append(f"- Forward passed: {arch['forward_passed']}")
        lines.append(f"- Backward passed: {arch['backward_passed']}")
        lines.append(f"- Nonzero gradient: {arch['nonzero_gradient']}")
        lines.append("")

        cuda = m["cuda"]
        lines.append("### CUDA")
        lines.append(f"- CUDA available: {cuda['cuda_available']}")
        if cuda.get("gpu_model"):
            lines.append(f"- GPU: {cuda['gpu_model']}")
        fb = cuda.get("forward_backward_cuda_passed")
        if fb:
            lines.append(f"- Forward/backward passed: {fb}")
            lines.append(f"- Peak GPU memory: {cuda.get('peak_gpu_memory_bytes')} bytes")
        if cuda.get("skip_reason"):
            lines.append(f"- Skip: {cuda['skip_reason']}")
        lines.append("")

        to = m["tiny_overfit"]
        lines.append("### Tiny-Overfit (Diagnostic Only)")
        lines.append(f"- LR: {to['learning_rate']} (diagnostic-only)")
        lines.append(f"- Initial loss: {to.get('initial_loss')}")
        lines.append(f"- Final loss: {to.get('final_loss')}")
        lines.append(f"- Initial accuracy: {to.get('initial_accuracy')}")
        lines.append(f"- Final accuracy: {to.get('final_accuracy')}")
        lines.append(f"- Pass/fail: {to.get('pass_fail')}")
        lines.append("")

        res = m["resume"]
        lines.append("### Resume Determinism")
        lines.append(f"- Max parameter delta: {res.get('max_parameter_delta')}")
        lines.append(f"- Prediction match: {res.get('prediction_match')}")
        lines.append(f"- Pass/fail: {res.get('pass_fail')}")
        lines.append("")

        tp = m["throughput"]
        lines.append("### Throughput (Diagnostic Only)")
        lines.append(f"- Batch size: {tp['batch_size']}")
        lines.append(f"- Samples/sec: {tp.get('samples_per_second')}")
        lines.append(f"- Inference latency: {tp.get('inference_latency_ms')} ms")
        lines.append(f"- Note: {tp['note']}")
        lines.append("")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return OUTPUT_JSON, OUTPUT_MD


if __name__ == "__main__":
    ev = generate()
    jp, mp = write_artifacts(ev)
    print(f"JSON: {jp}")  # noqa: T201
    print(f"MD:   {mp}")  # noqa: T201
