"""FI-2010 baseline experiment CLI.

Usage:
    python -m deepbook.cli.fi2010_baselines --help
    python -m deepbook.cli.fi2010_baselines prepare
    python -m deepbook.cli.fi2010_baselines report
    python -m deepbook.cli.fi2010_baselines verify-run --run-id <id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from deepbook.evaluation.classification import classification_metrics
from deepbook.evaluation.prediction import load_prediction_artifact, sha256_file
from deepbook.training.fi2010 import (
    FROZEN_DATA_IDENTITY_PATH,
    check_protocol_ancestry,
    code_commit_provenance_reasons,
    configuration_hash,
    expected_archive_sha256,
    expected_data_fingerprint,
    protocol_sha256,
    validate_run_manifest,
)


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _cmd_prepare(root: Path) -> int:
    """Verify data contracts, frozen identity, and planning, then persist the run index."""
    from deepbook.training.runner import planned_run_specs, write_run_index

    config_path = root / "configs" / "data" / "fi2010.yaml"
    if not config_path.is_file():
        print(f"ERROR: data config not found: {config_path}", file=sys.stderr)
        return 1
    split_path = root / "data" / "interim" / "fi2010" / "fi2010_split_manifest.json"
    if not split_path.is_file():
        print(f"ERROR: split manifest not found: {split_path}", file=sys.stderr)
        return 1
    frozen_path = root / FROZEN_DATA_IDENTITY_PATH
    if not frozen_path.is_file():
        print(f"ERROR: frozen data identity not found: {frozen_path}", file=sys.stderr)
        return 1
    specs = planned_run_specs(root)
    index_path = write_run_index(root)
    print(f"Planned matrix cells: {len(specs)}")
    print(f"Run index: {index_path}")
    print("Preparation checks passed.")
    return 0


def _cmd_report(root: Path) -> int:
    """Generate the deterministic run index and JSON/Markdown reports from manifests."""
    from deepbook.training.runner import write_report, write_run_index

    paths = write_report(root)
    index_path = write_run_index(root)
    print(f"Run index: {index_path}")
    print(f"Report JSON: {paths[0]}")
    print(f"Report Markdown: {paths[1]}")
    return 0


_NEURAL_MODELS = {"mlplob", "deeplob", "translob", "tlob"}
_REQUIRED_METRIC_KEYS = frozenset(
    {
        "macro_f1",
        "mcc",
        "accuracy",
        "balanced_accuracy",
        "nll",
        "brier",
        "ece",
        "confusion_matrix",
        "classwise_precision",
        "classwise_recall",
        "classwise_f1",
        "class_order",
        "sample_count",
    }
)
_RTOL = 1e-10
_ATOL = 1e-8


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def _cmd_verify_run(root: Path, run_id: str, artifact_root: Path | None = None) -> int:
    """Comprehensively verify a run manifest against its artifacts and protocol."""
    manifest_path = (
        (artifact_root or (root / "artifacts" / "fi2010" / "baselines"))
        / "runs"
        / (f"{run_id}.json")
    )
    if not manifest_path.is_file():
        return _fail(f"run manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Validate full manifest schema
    schema_path = root / "data_contracts" / "fi2010_run_manifest.schema.json"
    try:
        validate_run_manifest(manifest, schema_path)
    except Exception as exc:
        return _fail(f"schema validation failed: {exc}")

    if manifest.get("status") != "completed":
        return _fail(f"run status is not completed: {manifest.get('status')!r}")

    # Commit existence, ancestry, tree identity, and execution-time ordering are
    # checked independently of the manifest's self-declared eligibility.
    code_reasons = code_commit_provenance_reasons(root, manifest)
    if code_reasons:
        return _fail("; ".join(code_reasons))
    print(f"Code commit provenance verified: {manifest['code_commit'][:12]}...")

    # 2. Reject empty metrics
    metrics_block = manifest.get("metrics", {})
    if not metrics_block or not isinstance(metrics_block, dict):
        return _fail("metrics block is empty or not a dict")

    # 3. Verify protocol SHA-256 — accept the frozen baseline-suite epoch hash
    # (the pre-transformer contract) OR the current transformer-era hash.
    # Baseline runs legitimately record the frozen baseline protocol identity.
    from deepbook.training.fi2010_suite_snapshot import _BASELINE_SUITE_PROTOCOL_SHA256

    recorded_protocol_hash = manifest.get("protocol_sha256", "")
    current_protocol_hash = protocol_sha256(root)
    if recorded_protocol_hash not in (current_protocol_hash, _BASELINE_SUITE_PROTOCOL_SHA256):
        return _fail(
            f"protocol SHA-256 mismatch: manifest={recorded_protocol_hash[:16]}... "
            f"current={current_protocol_hash[:16]}... "
            f"baseline-epoch={_BASELINE_SUITE_PROTOCOL_SHA256[:16]}..."
        )
    print(f"Protocol SHA-256 verified: {recorded_protocol_hash[:16]}...")

    # 4. Verify protocol ancestry
    protocol_commit_val = manifest.get("protocol_commit", "")
    if not check_protocol_ancestry(root, protocol_commit_val):
        return _fail("code commit does not descend from protocol commit")
    print(f"Protocol ancestry verified: {protocol_commit_val[:12]}...")

    # 5. Verify configuration hash
    config_path_str = manifest.get("configuration_path", "")
    if config_path_str:
        config_path = root / config_path_str
        if config_path.is_file():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            computed_config_hash = configuration_hash(config)
            recorded_config_hash = manifest.get("configuration_hash", "")
            if computed_config_hash != recorded_config_hash:
                return _fail(
                    f"configuration hash mismatch: manifest={recorded_config_hash[:16]}... "
                    f"computed={computed_config_hash[:16]}..."
                )
            print(f"Configuration hash verified: {computed_config_hash[:16]}...")
        else:
            print(f"WARNING: configuration file not found: {config_path}")
    else:
        print("WARNING: no configuration_path in manifest; skipping config hash check")

    # 6. Verify frozen FI-2010 data identity
    setup = str(manifest.get("setup", ""))
    try:
        frozen_fingerprint = expected_data_fingerprint(
            root,
            setup=setup,
            fold=manifest.get("fold"),
            day_group=manifest.get("day_group"),
        )
    except (KeyError, ValueError) as exc:
        return _fail(f"frozen data identity could not be resolved: {exc}")
    if manifest.get("data_fingerprint", "") != frozen_fingerprint:
        return _fail(
            f"data fingerprint does not match the frozen FI-2010 data identity: "
            f"manifest={str(manifest.get('data_fingerprint', ''))[:16]}... "
            f"frozen={frozen_fingerprint[:16]}..."
        )
    if manifest.get("archive_sha256", "") != expected_archive_sha256(root):
        return _fail("archive SHA-256 does not match the frozen authoritative archive")
    print(f"Frozen data identity verified: {frozen_fingerprint[:16]}...")

    # 7. Verify checkpoints
    model_name = manifest.get("model", "")
    is_neural = model_name in _NEURAL_MODELS
    checkpoint_path_str = manifest.get("best_checkpoint_path") or manifest.get("checkpoint_path")
    checkpoint_sha = manifest.get("best_checkpoint_sha256") or manifest.get("checkpoint_sha256")

    if is_neural and not checkpoint_path_str:
        return _fail("neural model requires best_checkpoint_path")
    if is_neural and not checkpoint_sha:
        return _fail("neural model requires best_checkpoint_sha256")

    if checkpoint_path_str:
        cp_path = Path(checkpoint_path_str)
        if not cp_path.is_file():
            return _fail(f"best-model checkpoint not found: {cp_path}")
        actual = sha256_file(cp_path)
        if checkpoint_sha and actual != checkpoint_sha:
            return _fail(
                f"best checkpoint SHA-256 mismatch: manifest={checkpoint_sha[:16]}... "
                f"actual={actual[:16]}..."
            )
        print(f"Best checkpoint SHA-256 verified: {actual[:16]}...")
    elif checkpoint_sha is not None:
        print("Best checkpoint SHA-256 recorded as null (classical model)")
    else:
        print("Best checkpoint SHA-256: not recorded")

    last_path_str = manifest.get("last_checkpoint_path")
    last_sha = manifest.get("last_checkpoint_sha256")
    if last_path_str and last_sha:
        last_path = Path(last_path_str)
        if not last_path.is_file():
            return _fail(f"last-state checkpoint not found: {last_path}")
        actual_last = sha256_file(last_path)
        if actual_last != last_sha:
            return _fail(
                f"last checkpoint SHA-256 mismatch: manifest={last_sha[:16]}... "
                f"actual={actual_last[:16]}..."
            )
        print(f"Last-state checkpoint SHA-256 verified: {actual_last[:16]}...")

    # 8. Verify prediction artifact
    pred_path_str = manifest.get("prediction_path")
    pred_sha = manifest.get("prediction_sha256")
    if manifest.get("status") == "completed" and not (pred_path_str and pred_sha):
        return _fail("a completed run must record prediction_path and prediction_sha256")
    if pred_path_str and pred_sha:
        pp = Path(pred_path_str)
        if not pp.is_file():
            return _fail(f"prediction artifact not found: {pp}")
        actual = sha256_file(pp)
        if actual != pred_sha:
            return _fail(
                f"prediction SHA-256 mismatch: manifest={pred_sha[:16]}... actual={actual[:16]}..."
            )
        print(f"Prediction SHA-256 verified: {actual[:16]}...")

        try:
            preds = load_prediction_artifact(pp)
        except (ValueError, OSError) as exc:
            return _fail(f"prediction artifact could not be loaded: {exc}")

        # Verify sample count
        manifest_sample_count = manifest.get("sample_count")
        pred_n = int(preds["y_true"].shape[0])
        if manifest_sample_count is not None and manifest_sample_count != pred_n:
            return _fail(
                f"sample count mismatch: manifest={manifest_sample_count} prediction={pred_n}"
            )
        # All arrays have matching lengths
        for key in ("y_pred", "probabilities", "sample_index", "source_file_id", "day_boundary_id"):
            if preds[key].shape[0] != pred_n:
                return _fail(
                    f"prediction array {key} has wrong length: {preds[key].shape[0]} != {pred_n}"
                )
        # Class order
        if tuple(preds.get("class_order", [])) != ("up", "stationary", "down"):
            return _fail(f"class_order mismatch: {tuple(preds.get('class_order', []))}")

        # Day boundary identity must match the manifest's audited source mapping
        day_index_map = manifest.get("day_index_map") or {}
        if day_index_map:
            observed_days = {int(value) for value in preds["day_boundary_id"].tolist()}
            declared_days = {int(key) for key in day_index_map}
            if observed_days != declared_days:
                return _fail(
                    f"day_boundary_id values {sorted(observed_days)} do not match "
                    f"the manifest day_index_map {sorted(declared_days)}"
                )
            for day, entry in sorted(day_index_map.items(), key=lambda item: int(item[0])):
                mask = preds["day_boundary_id"] == int(day)
                sources = {int(value) for value in preds["source_file_id"][mask].tolist()}
                if sources != {int(entry["source_fold"])}:
                    return _fail(
                        f"day {day} carries source_file_id {sorted(sources)}, "
                        f"expected {entry['source_fold']}"
                    )
            print(f"Day boundary identity verified for days {sorted(declared_days)}.")

        # Verify probabilities are valid
        proba = np.asarray(preds["probabilities"], dtype=np.float64)
        if not np.isfinite(proba).all():
            return _fail("probabilities contain non-finite values")
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-8):
            return _fail("probability rows do not sum to 1.0")

        # 8. Recompute metrics
        recomputed = classification_metrics(
            preds["y_true"], preds["y_pred"], preds["probabilities"]
        )
        manifest_metrics = metrics_block.get("test", {})
        if not manifest_metrics:
            return _fail("manifest test metrics are empty")

        # Check required metric keys exist
        for key in _REQUIRED_METRIC_KEYS:
            if key not in manifest_metrics:
                return _fail(f"required metric key missing from manifest: {key}")

        metric_keys = [
            "macro_f1",
            "mcc",
            "accuracy",
            "balanced_accuracy",
            "nll",
            "brier",
            "ece",
        ]
        for key in metric_keys:
            if key in manifest_metrics and key in recomputed:
                expected = float(manifest_metrics[key])
                actual_val = float(recomputed[key])
                if not np.allclose(expected, actual_val, rtol=_RTOL, atol=_ATOL):
                    return _fail(
                        f"metric mismatch for {key}: "
                        f"manifest={expected:.12f} recomputed={actual_val:.12f}"
                    )
                print(f"  {key}: manifest={expected:.6f} recomputed={actual_val:.6f} OK")
        # Confusion matrix
        if "confusion_matrix" in manifest_metrics and "confusion_matrix" in recomputed:
            cm_manifest = np.array(manifest_metrics["confusion_matrix"], dtype=np.int64)
            cm_recomputed = np.array(recomputed["confusion_matrix"], dtype=np.int64)
            if not np.array_equal(cm_manifest, cm_recomputed):
                return _fail("confusion matrix mismatch")
            print("  confusion_matrix OK")
        # Class-level metrics
        for arr_key in ("classwise_precision", "classwise_recall", "classwise_f1"):
            if arr_key in manifest_metrics and arr_key in recomputed:
                if not np.allclose(
                    np.array(manifest_metrics[arr_key], dtype=np.float64),
                    np.array(recomputed[arr_key], dtype=np.float64),
                    rtol=_RTOL,
                    atol=_ATOL,
                ):
                    return _fail(f"{arr_key} mismatch")
                print(f"  {arr_key} OK")
        print("All metrics verified.")
    elif pred_path_str is None and pred_sha is None:
        print("No prediction artifact recorded.")
    else:
        return _fail("prediction_path and prediction_sha256 inconsistent")

    print(f"Run {run_id} verification passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch FI-2010 baseline subcommands."""
    parser = argparse.ArgumentParser(
        description="FI-2010 baseline experiment framework",
        prog="python -m deepbook.cli.fi2010_baselines",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("prepare", help="verify data contracts and cache readiness")
    sub.add_parser("report", help="generate deterministic JSON and Markdown reports")
    sub.add_parser("snapshot", help="generate deterministic tracked reproduction snapshot")
    sub.add_parser(
        "snapshot-suite", help="generate complete 900-cell FI-2010 baseline suite snapshot"
    )
    sub.add_parser("snapshot-translob", help="generate the 250-cell TransLOB reproduction snapshot")

    verify = sub.add_parser("verify-run", help="verify run manifest, hashes, and metrics")
    verify.add_argument("--run-id", required=True, help="run ID to verify")

    args = parser.parse_args(argv)
    root = _find_repo_root()

    if args.command == "prepare":
        return _cmd_prepare(root)
    if args.command == "report":
        return _cmd_report(root)
    if args.command == "snapshot":
        from deepbook.training.fi2010_snapshot import write_snapshot

        json_path, md_path = write_snapshot(root)
        print(f"Snapshot JSON: {json_path}")
        print(f"Snapshot Markdown: {md_path}")
        return 0
    if args.command == "snapshot-suite":
        from deepbook.training.fi2010_suite_snapshot import write_suite_snapshot

        json_path, md_path = write_suite_snapshot(root)
        print(f"Suite Snapshot JSON: {json_path}")
        print(f"Suite Snapshot Markdown: {md_path}")
        return 0
    if args.command == "snapshot-translob":
        from deepbook.training.fi2010_translob_snapshot import write_translob_snapshot

        json_path, md_path = write_translob_snapshot(root)
        print(f"TransLOB Snapshot JSON: {json_path}")
        print(f"TransLOB Snapshot Markdown: {md_path}")
        return 0
    if args.command == "verify-run":
        return _cmd_verify_run(root, args.run_id)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
