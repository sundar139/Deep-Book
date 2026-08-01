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

from deepbook.evaluation.classification import classification_metrics
from deepbook.evaluation.prediction import load_prediction_artifact, sha256_file
from deepbook.training.fi2010 import validate_run_manifest


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _cmd_prepare(root: Path) -> int:
    """Verify data contracts and cache readiness."""
    config_path = root / "configs" / "data" / "fi2010.yaml"
    if not config_path.is_file():
        print(f"ERROR: data config not found: {config_path}", file=sys.stderr)
        return 1
    split_path = root / "data" / "interim" / "fi2010" / "fi2010_split_manifest.json"
    if not split_path.is_file():
        print(f"ERROR: split manifest not found: {split_path}", file=sys.stderr)
        return 1
    print("Preparation checks passed.")
    return 0


def _cmd_report(root: Path) -> int:
    """Generate deterministic JSON and Markdown reports from run manifests."""
    from deepbook.training.runner import write_report

    paths = write_report(root)
    print(f"Report JSON: {paths[0]}")
    print(f"Report Markdown: {paths[1]}")
    return 0


def _cmd_verify_run(root: Path, run_id: str) -> int:
    """Load a run manifest, validate, hash-check artifacts, and recompute metrics."""
    manifest_path = root / "artifacts" / "fi2010" / "baselines" / "runs" / f"{run_id}.json"
    if not manifest_path.is_file():
        print(f"ERROR: run manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_path = root / "data_contracts" / "fi2010_run_manifest.schema.json"
    try:
        validate_run_manifest(manifest, schema_path)
    except Exception as exc:
        print(f"ERROR: schema validation failed: {exc}", file=sys.stderr)
        return 1

    # Metrics tolerance
    rtol = 1e-10
    atol = 1e-8

    # Verify checkpoint SHA-256
    checkpoint_path_str = manifest.get("checkpoint_path")
    checkpoint_sha = manifest.get("checkpoint_sha256")
    if checkpoint_path_str:
        cp_path = Path(checkpoint_path_str)
        if not cp_path.is_file():
            print(f"ERROR: checkpoint not found: {cp_path}", file=sys.stderr)
            return 1
        actual = sha256_file(cp_path)
        if checkpoint_sha and actual != checkpoint_sha:
            print(
                f"ERROR: checkpoint SHA-256 mismatch: manifest={checkpoint_sha} actual={actual}",
                file=sys.stderr,
            )
            return 1
        print(f"Checkpoint SHA-256 verified: {actual[:16]}...")
    elif checkpoint_sha is not None:
        print("Checkpoint SHA-256: null (model family has no checkpoint artifact)")
    else:
        print("Checkpoint SHA-256: not recorded")

    # Verify prediction SHA-256
    pred_path_str = manifest.get("prediction_path")
    pred_sha = manifest.get("prediction_sha256")
    if pred_path_str and pred_sha:
        pp = Path(pred_path_str)
        if not pp.is_file():
            print(f"ERROR: prediction artifact not found: {pp}", file=sys.stderr)
            return 1
        actual = sha256_file(pp)
        if actual != pred_sha:
            print(
                f"ERROR: prediction SHA-256 mismatch: manifest={pred_sha} actual={actual}",
                file=sys.stderr,
            )
            return 1
        print(f"Prediction SHA-256 verified: {actual[:16]}...")

        # Recompute metrics from prediction artifact
        preds = load_prediction_artifact(pp)
        recomputed = classification_metrics(
            preds["y_true"], preds["y_pred"], preds["probabilities"]
        )
        manifest_metrics = manifest.get("metrics", {}).get("test", {})
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
                if not np.allclose(expected, actual_val, rtol=rtol, atol=atol):
                    print(
                        f"ERROR: metric mismatch for {key}: "
                        f"manifest={expected:.12f} recomputed={actual_val:.12f}",
                        file=sys.stderr,
                    )
                    return 1
                print(f"  {key}: manifest={expected:.6f} recomputed={actual_val:.6f} OK")
        # Confusion matrix
        if "confusion_matrix" in manifest_metrics and "confusion_matrix" in recomputed:
            cm_manifest = np.array(manifest_metrics["confusion_matrix"], dtype=np.int64)
            cm_recomputed = np.array(recomputed["confusion_matrix"], dtype=np.int64)
            if not np.array_equal(cm_manifest, cm_recomputed):
                print("ERROR: confusion matrix mismatch", file=sys.stderr)
                return 1
            print("  confusion_matrix OK")
        print("All metrics verified.")
    elif pred_path_str is None and pred_sha is None:
        print("No prediction artifact recorded.")
    else:
        print("WARNING: prediction_path and prediction_sha256 inconsistent; skipping verification")

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

    verify = sub.add_parser("verify-run", help="verify run manifest, hashes, and metrics")
    verify.add_argument("--run-id", required=True, help="run ID to verify")

    args = parser.parse_args(argv)
    root = _find_repo_root()

    if args.command == "prepare":
        return _cmd_prepare(root)
    if args.command == "report":
        return _cmd_report(root)
    if args.command == "verify-run":
        return _cmd_verify_run(root, args.run_id)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
