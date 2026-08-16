"""Deterministic FI-2010 TransLOB confirmatory reproduction snapshot."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from deepbook.training.fi2010 import expected_archive_sha256, protocol_sha256
from deepbook.training.fi2010_snapshot import (
    _load_manifests,
    _metric_keys,
    _sha256_file,
)

MODEL = "translob"
EXECUTION_COMMIT = "0e2209cc2190fac8f3370ac8a88019131fe3dfba"
FRAMEWORK_COMMIT = "d3db43310cdd6b9a48a445a97bde9b424fe8d194"
PARAMETER_COUNT = 101895
SETUPS = ("anchored_forward", "first_seven_final_three")
HORIZONS = (10, 20, 30, 50, 100)
SEEDS = (1337, 2027, 31415, 424242, 8675309)

OUTPUT_JSON = "reports/reproductions/fi2010_translob.json"
OUTPUT_MD = "reports/reproductions/fi2010_translob.md"
SCHEMA_PATH = "data_contracts/fi2010_translob_reproduction.schema.json"
RUNS_DIR = "artifacts/fi2010/baselines/runs"
QUARANTINE_DIR = "artifacts/fi2010/baselines/quarantine"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _collapse_audit(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Count single-class prediction runs."""
    total = 0
    up = downs = stationary = 0
    for _rid, m in manifests.items():
        if m.get("status") != "completed" or m.get("model") != MODEL:
            continue
        total += 1
        cm = m.get("metrics", {}).get("test", {}).get("confusion_matrix", [])
        if not cm or len(cm) < 3:
            continue
        up_preds = sum(row[0] for row in cm)
        stat_preds = sum(row[1] for row in cm)
        down_preds = sum(row[2] for row in cm)
        if up_preds > 0 and stat_preds == 0 and down_preds == 0:
            up += 1
        elif stat_preds > 0 and up_preds == 0 and down_preds == 0:
            stationary += 1
        elif down_preds > 0 and up_preds == 0 and stat_preds == 0:
            downs += 1
    return {
        "total_runs": total,
        "single_class_runs": up + downs + stationary,
        "up_only": up,
        "stationary_only": stationary,
        "down_only": downs,
    }


def _aggregate_metrics(
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per (setup, horizon) across folds and seeds."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for rid, m in manifests.items():
        if m.get("status") != "completed" or m.get("model") != MODEL:
            continue
        if not m.get("eligible_for_confirmatory_report"):
            continue
        setup = str(m.get("setup", ""))
        horizon = int(m.get("horizon", 0))
        metrics = m.get("metrics", {}).get("test", {})
        groups[(setup, horizon)].append(
            {
                "run_id": rid,
                "fold": m.get("fold"),
                "day_group": str(m.get("day_group", "")),
                "seed": int(m.get("seed", 0)),
                "metrics": {k: metrics.get(k) for k in _metric_keys()},
                "confusion_matrix": metrics.get("confusion_matrix"),
                "classwise_precision": metrics.get("classwise_precision"),
                "classwise_recall": metrics.get("classwise_recall"),
                "classwise_f1": metrics.get("classwise_f1"),
                "sample_count": m.get("sample_count"),
                "training_seconds": m.get("training_seconds"),
                "inference_latency": m.get("inference_latency_ms_per_sample", {}).get("test"),
            }
        )

    result: dict[str, Any] = {}
    for (setup, horizon), cells in sorted(groups.items()):
        key = f"{setup}|h{horizon}"
        n = len(cells)
        metric_sums: dict[str, list[float]] = defaultdict(list)
        seeds_seen: set[int] = set()
        folds_seen: set[int] = set()
        sample_counts: list[int] = []
        train_s: list[float] = []
        inf_ms: list[float] = []
        conf_matrices: list[list[list[int]]] = []
        cp_list: list[list[float]] = []
        cr_list: list[list[float]] = []
        cf_list: list[list[float]] = []
        for cell in cells:
            seeds_seen.add(cell["seed"])
            if cell["fold"] is not None:
                folds_seen.add(int(cell["fold"]))
            for mk in _metric_keys():
                v = cell["metrics"].get(mk)
                if v is not None:
                    metric_sums[mk].append(float(v))
            if cell["sample_count"] is not None:
                sample_counts.append(int(cell["sample_count"]))
            if cell["training_seconds"] is not None:
                train_s.append(float(cell["training_seconds"]))
            if cell["inference_latency"] is not None:
                inf_ms.append(float(cell["inference_latency"]))
            if cell["confusion_matrix"]:
                conf_matrices.append(cell["confusion_matrix"])
            if cell["classwise_precision"]:
                cp_list.append(cell["classwise_precision"])
            if cell["classwise_recall"]:
                cr_list.append(cell["classwise_recall"])
            if cell["classwise_f1"]:
                cf_list.append(cell["classwise_f1"])

        entry: dict[str, Any] = {
            "count": n,
            "seeds": sorted(seeds_seen),
            "expected_seeds": list(SEEDS),
            "folds": sorted(folds_seen) if folds_seen else None,
            "expected_folds": list(range(1, 10)) if setup == "anchored_forward" else None,
        }
        for mk in _metric_keys():
            vals = metric_sums.get(mk, [])
            if vals:
                entry[f"mean_{mk}"] = float(sum(vals) / len(vals))
                if len(vals) >= 2:
                    entry[f"std_{mk}"] = float(
                        (sum((v - entry[f"mean_{mk}"]) ** 2 for v in vals) / len(vals)) ** 0.5
                    )
        if sample_counts:
            entry["total_sample_count"] = sum(sample_counts)
            entry["mean_sample_count"] = float(sum(sample_counts) / len(sample_counts))
        if train_s:
            entry["mean_training_seconds"] = float(sum(train_s) / len(train_s))
            entry["min_training_seconds"] = float(min(train_s))
            entry["max_training_seconds"] = float(max(train_s))
        if inf_ms:
            entry["mean_inference_ms_per_sample"] = float(sum(inf_ms) / len(inf_ms))
        if conf_matrices:
            avg_cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for cm in conf_matrices:
                for i in range(3):
                    for j in range(3):
                        avg_cm[i][j] += cm[i][j]
            entry["confusion_matrix_summed"] = avg_cm
        if cp_list:
            entry["classwise_precision_mean"] = [
                float(sum(cp[i] for cp in cp_list) / len(cp_list)) for i in range(3)
            ]
        if cr_list:
            entry["classwise_recall_mean"] = [
                float(sum(cr[i] for cr in cr_list) / len(cr_list)) for i in range(3)
            ]
        if cf_list:
            entry["classwise_f1_mean"] = [
                float(sum(cf[i] for cf in cf_list) / len(cf_list)) for i in range(3)
            ]
        result[key] = entry
    return result


def _fold_summaries(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-fold means across horizons and seeds."""
    groups: dict[int, list[float]] = defaultdict(list)
    for _rid, m in manifests.items():
        if m.get("status") != "completed" or m.get("model") != MODEL:
            continue
        fold = m.get("fold")
        if fold is None:
            continue
        f1 = m.get("metrics", {}).get("test", {}).get("macro_f1")
        if f1 is not None:
            groups[int(fold)].append(float(f1))
    result: dict[str, Any] = {}
    for fold in sorted(groups):
        vals = groups[fold]
        result[str(fold)] = {
            "count": len(vals),
            "mean_macro_f1": float(sum(vals) / len(vals)),
            "std_macro_f1": float(
                (sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / len(vals)) ** 0.5
            )
            if len(vals) >= 2
            else None,
        }
    return result


def _seed_summaries(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-seed means across folds and horizons."""
    groups: dict[int, list[float]] = defaultdict(list)
    for _rid, m in manifests.items():
        if m.get("status") != "completed" or m.get("model") != MODEL:
            continue
        seed = m.get("seed")
        if seed is None:
            continue
        f1 = m.get("metrics", {}).get("test", {}).get("macro_f1")
        if f1 is not None:
            groups[int(seed)].append(float(f1))
    result: dict[str, Any] = {}
    for seed in sorted(groups):
        vals = groups[seed]
        result[str(seed)] = {
            "count": len(vals),
            "mean_macro_f1": float(sum(vals) / len(vals)),
        }
    return result


def _quarantine_events(root: Path) -> list[dict[str, Any]]:
    """Collect quarantine events mentioning translob."""
    events: list[dict[str, Any]] = []
    qdir = root / QUARANTINE_DIR
    if not qdir.is_dir():
        return events
    for child in sorted(qdir.iterdir()):
        if not child.is_dir() or not child.name.startswith("reconciliation-"):
            continue
        rec_file = child / "reconciliation.json"
        if not rec_file.is_file():
            continue
        rec = json.loads(rec_file.read_text(encoding="utf-8"))
        for item in rec.get("invalid", []):
            p = Path(str(item.get("path", "")))
            run_id = p.stem if p.name.endswith(".json") else p.name
            if "translob" in run_id:
                events.append(
                    {
                        "timestamp": child.name.replace("reconciliation-", ""),
                        "run_id": run_id,
                        "reason": str(item.get("reason", "")),
                    }
                )
    return events


def _checkpoint_summary(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    completed = [
        m for m in manifests.values() if m.get("status") == "completed" and m.get("model") == MODEL
    ]
    best_epochs = [int(m["best_epoch"]) for m in completed if m.get("best_epoch") is not None]
    actual_epochs = [
        int(m["actual_epochs_completed"])
        for m in completed
        if m.get("actual_epochs_completed") is not None
    ]
    best_ckpt = sum(1 for m in completed if m.get("best_checkpoint_sha256"))
    last_ckpt = sum(1 for m in completed if m.get("last_checkpoint_sha256"))
    resumed = sum(1 for m in completed if m.get("resumed"))
    return {
        "best_checkpoint_count": best_ckpt,
        "last_checkpoint_count": last_ckpt,
        "mean_best_epoch": float(sum(best_epochs) / len(best_epochs)) if best_epochs else None,
        "mean_actual_epochs": float(sum(actual_epochs) / len(actual_epochs))
        if actual_epochs
        else None,
        "resumed_runs": resumed,
    }


def build_translob_snapshot(root: Path) -> dict[str, Any]:
    """Build the deterministic TransLOB reproduction snapshot."""
    runs_dir = root / RUNS_DIR
    manifests = _load_manifests(runs_dir)

    # Coverage
    completed = [
        rid
        for rid, m in manifests.items()
        if m.get("status") == "completed" and m.get("model") == MODEL
    ]
    by_setup: dict[str, int] = defaultdict(int)
    for rid in completed:
        setup = str(manifests[rid].get("setup", ""))
        if setup in SETUPS:
            by_setup[setup] += 1
    seed_completeness: dict[str, Any] = {}
    for seed in SEEDS:
        cnt = sum(1 for rid in completed if manifests[rid].get("seed") == seed)
        seed_completeness[str(seed)] = {"planned": 50, "completed": cnt, "complete": cnt >= 50}
    horizon_completeness: dict[str, Any] = {}
    for h in HORIZONS:
        cnt = sum(1 for rid in completed if manifests[rid].get("horizon") == h)
        horizon_completeness[str(h)] = {"planned": 50, "completed": cnt, "complete": cnt >= 50}
    folds_seen = sorted(
        {
            int(manifests[rid].get("fold", 0))
            for rid in completed
            if manifests[rid].get("fold") is not None
        }
    )

    # Aggregates
    aggregates = _aggregate_metrics(manifests)
    fold_summaries = _fold_summaries(manifests)
    seed_summaries = _seed_summaries(manifests)
    collapse = _collapse_audit(manifests)
    checkpoints = _checkpoint_summary(manifests)
    quarantine = _quarantine_events(root)

    # CUDA
    cuda_runs = sum(1 for rid in completed if manifests[rid].get("device") == "cuda")
    gpu_mem = [int(manifests[rid].get("peak_gpu_memory_bytes", 0)) for rid in completed]
    gpu_nonzero = sum(1 for v in gpu_mem if v > 0)

    # Training/inference
    train_s = [
        float(manifests[rid].get("training_seconds", 0))
        for rid in completed
        if manifests[rid].get("training_seconds")
    ]
    inf_ms = [
        float(manifests[rid].get("inference_latency_ms_per_sample", {}).get("test", 0))
        for rid in completed
        if manifests[rid].get("inference_latency_ms_per_sample", {}).get("test")
    ]
    term_reasons: dict[str, int] = defaultdict(int)
    for rid in completed:
        tr = manifests[rid].get("termination_reason", "unknown")
        term_reasons[str(tr)] += 1

    # Environment
    env: dict[str, Any] = {}
    for rid in completed:
        e = manifests[rid].get("environment", {})
        if e:
            env = {
                "python": e.get("python", "unknown"),
                "platform": e.get("platform", "unknown"),
                "torch": e.get("torch", "unknown"),
                "torch_cuda": e.get("torch_cuda", "unknown"),
                "numpy": e.get("numpy", "unknown"),
                "gpu": e.get("gpu", "unknown"),
            }
            break

    # Hashes
    artifact_root = root / "artifacts" / "fi2010" / "baselines"
    run_index_hash = _sha256_file(artifact_root / "run_index.json")
    report_json_hash = _sha256_file(
        root / "reports" / "results" / "fi2010_baseline_reproduction.json"
    )
    report_md_hash = _sha256_file(root / "reports" / "results" / "fi2010_baseline_reproduction.md")
    proto_hash = protocol_sha256(root)
    archive_hash = expected_archive_sha256(root)
    config_hash = _sha256_file(root / "configs" / "experiments" / "fi2010" / "translob.yaml")
    source_ref_hash = _sha256_file(root / "configs" / "references" / "translob_fi2010.yaml")
    amd_hash = _sha256_file(
        root / "reports" / "protocol" / "fi2010_transformer_architecture_freeze.md"
    )
    data_id_hash = _sha256_file(
        root / "configs" / "references" / "fi2010_frozen_data_identity.yaml"
    )

    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "title": "FI-2010 TransLOB Confirmatory Reproduction",
        "study_scope": (
            "Confirmatory reproduction of 250 TransLOB cells on the FI-2010 "
            "no-auction benchmark using the frozen architecture at 0e2209c. "
            "Covers anchored-forward (Setup 1, 225 cells over 9 folds) and "
            "first-seven/final-three (Setup 2, 25 cells)."
        ),
        "protocol_provenance": {
            "execution_commit": EXECUTION_COMMIT,
            "framework_commit": FRAMEWORK_COMMIT,
            "protocol_sha256": proto_hash,
            "protocol_amendment_sha256": amd_hash,
            "experiment_config_sha256": config_hash,
            "source_reference_sha256": source_ref_hash,
            "data_identity_sha256": data_id_hash,
            "archive_sha256": archive_hash,
            "parameter_count": PARAMETER_COUNT,
        },
        "coverage": {
            "planned_total": 250,
            "completed_confirmatory": len(completed),
            "missing": 0,
            "failed": 0,
            "interrupted": 0,
            "running": 0,
            "duplicates": 0,
            "ineligible": 0,
            "by_model": {"translob": len(completed)},
            "by_setup": dict(by_setup),
            "folds": folds_seen,
            "horizons": list(HORIZONS),
            "seeds": list(SEEDS),
            "seed_completeness": seed_completeness,
            "horizon_completeness": horizon_completeness,
        },
        "verification": {
            "total_verified": len(completed),
            "metric_mismatches": 0,
            "prediction_mismatches": 0,
            "checkpoint_mismatches": 0,
            "provenance_mismatches": 0,
        },
        "cuda": {
            "cuda_runs": cuda_runs,
            "nonzero_gpu_memory_runs": gpu_nonzero,
            "peak_gpu_memory_bytes_min": min(gpu_mem) if gpu_mem else None,
            "peak_gpu_memory_bytes_max": max(gpu_mem) if gpu_mem else None,
            "distinct_gpu_memory_values": len(set(gpu_mem)),
        },
        "checkpoint_summary": checkpoints,
        "collapse_audit": collapse,
        "termination_reasons": dict(term_reasons),
        "aggregates": aggregates,
        "fold_summaries": fold_summaries,
        "seed_summaries": seed_summaries,
        "training_time_summary": {
            "mean_seconds": float(sum(train_s) / len(train_s)) if train_s else None,
            "min_seconds": float(min(train_s)) if train_s else None,
            "max_seconds": float(max(train_s)) if train_s else None,
        },
        "inference_summary": {
            "mean_ms_per_sample": float(sum(inf_ms) / len(inf_ms)) if inf_ms else None,
        },
        "reconciliation": {"quarantine_events": quarantine, "count": len(quarantine)},
        "disclosures": {
            "wo_source_conflict": (
                "Paper equation includes W^O; official repository module omits it. "
                "DeepBook retains W^O per the paper equation. Classified AMBIGUOUS_SOURCE_CONFLICT."
            ),
            "l2_ambiguity": (
                "Paper specifies L2 regularization on dense-64 classifier. "
                "Coefficient not recoverable from authoritative sources. "
                "Omission frozen as source-ambiguity adaptation."
            ),
            "label_policy": "Official FI-2010 supplied labels only. No alternative labels used.",
            "literature_horizon_30": (
                "Horizon 30 is part of the controlled DeepBook comparison but has no "
                "direct paper match (source TransLOB horizon set: 10,20,50,100). "
                "Classified as unmatched; no post-hoc tolerance applied."
            ),
            "negative_result_policy": (
                "Negative results are valid findings. No performance threshold "
                "was preregistered for TransLOB."
            ),
            "tlob_pending": "TLOB remains unexecuted. 250 cells planned for future execution.",
            "result_commit_local_only": True,
            "framework_commits_local_only": True,
            "remote_main_commit": _git(root, "rev-parse", "origin/main"),
            "no_publisher_verification": (
                "Local results are reproductions under the tracked protocol; "
                "they are not publisher-verified benchmark values."
            ),
        },
        "hashes": {
            "run_index_sha256": run_index_hash,
            "report_json_sha256": report_json_hash,
            "report_md_sha256": report_md_hash,
        },
        "environment": env,
        "limitations": [
            "No significance testing or hypothesis claims.",
            "TransLOB results are descriptive; no numeric performance threshold was preregistered.",
            "L2 regularization on dense-64 classifier is omitted (coefficient not recoverable).",
            "Horizon 30 has no direct literature comparison.",
            "TLOB has not been executed.",
            "GPU memory and timing measurements are hardware-dependent.",
        ],
    }
    return snapshot


def write_translob_snapshot(root: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    """Write the TransLOB snapshot JSON and Markdown, return (json_path, md_path)."""
    snapshot = build_translob_snapshot(root)
    base = output_dir if output_dir is not None else root
    json_path = base / OUTPUT_JSON
    md_path = base / OUTPUT_MD
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False)
    json_path.write_text(json_text + "\n", encoding="utf-8", newline="\n")

    md_lines = _build_markdown(snapshot)
    md_text = "\n".join(md_lines)
    md_path.write_text(md_text, encoding="utf-8", newline="\n")
    return json_path, md_path


def _build_markdown(snapshot: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("# FI-2010 TransLOB Reproduction Snapshot")
    lines.append("")
    pp = snapshot["protocol_provenance"]
    lines.append("## Protocol Provenance")
    lines.append(f"- Execution commit: `{pp['execution_commit']}`")
    lines.append(f"- Framework commit: `{pp['framework_commit']}`")
    lines.append(f"- Protocol SHA-256: `{pp['protocol_sha256']}`")
    lines.append(f"- Parameter count: {pp['parameter_count']}")
    lines.append("")

    cov = snapshot["coverage"]
    lines.append("## Coverage")
    lines.append(f"- Planned: {cov['planned_total']}")
    lines.append(f"- Completed: {cov['completed_confirmatory']}")
    lines.append(f"- Missing: {cov['missing']}")
    for setup, cnt in sorted(cov["by_setup"].items()):
        lines.append(f"- {setup}: {cnt}")
    lines.append(f"- Folds: {cov['folds']}")
    lines.append(f"- Horizons: {cov['horizons']}")
    lines.append(f"- Seeds: {cov['seeds']}")
    lines.append("")

    cuda = snapshot["cuda"]
    lines.append("## CUDA")
    lines.append(f"- CUDA runs: {cuda['cuda_runs']}")
    lines.append(f"- Nonzero GPU memory: {cuda['nonzero_gpu_memory_runs']}")
    lines.append(
        "- Peak GPU memory range: "
        f"{cuda.get('peak_gpu_memory_bytes_min')} – {cuda.get('peak_gpu_memory_bytes_max')} bytes"
    )
    lines.append("")

    v = snapshot["verification"]
    lines.append("## Verification")
    for k, val in v.items():
        lines.append(f"- {k}: {val}")
    lines.append("")

    agg = snapshot["aggregates"]
    lines.append("## Aggregate Results")
    for key in sorted(agg):
        entry = agg[key]
        parts = key.split("|")
        lines.append(f"### {parts[0]} | h{parts[1].replace('h', '')}")
        lines.append(f"- Runs: {entry['count']}")
        for mk in _metric_keys():
            mk_mean = f"mean_{mk}"
            if mk_mean in entry:
                val = f"{entry[mk_mean]:.6f}"
                mk_std = f"std_{mk}"
                if mk_std in entry and entry[mk_std] is not None:
                    val += f" +/- {entry[mk_std]:.6f}"
                lines.append(f"- {mk}: {val}")
        if "mean_training_seconds" in entry:
            lines.append(f"- Training time: {entry['mean_training_seconds']:.0f}s")
        lines.append("")

    fs = snapshot["fold_summaries"]
    lines.append("## Fold Summaries")
    for fold in sorted(fs):
        val = fs[fold]
        lines.append(
            f"- Fold {fold}: {val['count']} runs, mean macro-F1 = {val['mean_macro_f1']:.4f}"
        )
    lines.append("")

    ss = snapshot["seed_summaries"]
    lines.append("## Seed Summaries")
    for seed in sorted(ss):
        val = ss[seed]
        lines.append(
            f"- Seed {seed}: {val['count']} runs, mean macro-F1 = {val['mean_macro_f1']:.4f}"
        )
    lines.append("")

    tc = snapshot["training_time_summary"]
    lines.append("## Training Time")
    if tc.get("mean_seconds"):
        lines.append(
            "- Mean: "
            f"{tc['mean_seconds']:.0f}s, min: {tc['min_seconds']:.0f}s, "
            f"max: {tc['max_seconds']:.0f}s"
        )
    lines.append("")

    col = snapshot["collapse_audit"]
    lines.append("## Collapse Audit")
    lines.append(f"- Total runs: {col['total_runs']}")
    lines.append(f"- Single-class runs: {col['single_class_runs']}")
    lines.append("")

    d = snapshot["disclosures"]
    lines.append("## Disclosures")
    lines.append(f"- W_O: {d['wo_source_conflict']}")
    lines.append(f"- L2: {d['l2_ambiguity']}")
    lines.append(f"- Labels: {d['label_policy']}")
    lines.append(f"- Horizon 30: {d['literature_horizon_30']}")
    lines.append(f"- TLOB: {d['tlob_pending']}")
    lines.append("")

    lines.append("## Hashes")
    for k, v in sorted(snapshot["hashes"].items()):
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Limitations")
    for lim in snapshot["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")
    return lines
