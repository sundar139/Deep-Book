"""Repository policy checks for DeepBook.

Verifies: .env not tracked, .env.example tracked, prohibited paths ignored,
paid-data authorization defaults to false, no staged implementation files,
no stage terminology in tracked content.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROHIBITED_IMPORTS = [
    "numpy",
    "scipy",
    "sklearn",
    "torch",
    "pandas",
    "gymnasium",
    "stable_baselines3",
]

PROHIBITED_PATHS = [
    "src/deepbook/features/hawkes.py",
    "src/deepbook/eval/stats.py",
    "src/deepbook/validation/invariants.py",
]

STAGE_TERMS = ["phase 0", "phase 1", "phase 2", "phase 3", "phase I", "phase II"]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def ok(msg: str) -> None:
    print(f"OK: {msg}")


def main() -> int:
    errors = 0

    # --- .env not tracked ---
    r = _git("ls-files", "--error-unmatch", ".env")
    if r.returncode == 0:
        errors += fail(".env is tracked by git")
    else:
        ok(".env is not tracked")

    # --- .env.example tracked ---
    r = _git("ls-files", "--error-unmatch", ".env.example")
    if r.returncode != 0:
        errors += fail(".env.example is NOT tracked by git")
    else:
        ok(".env.example is tracked")

    # --- .env is gitignored ---
    r = _git("check-ignore", "-v", ".env")
    if r.returncode != 0:
        errors += fail(".env is not git-ignored")
    else:
        ok(f".env is git-ignored ({r.stdout.strip()})")

    # --- Raw data paths ignored ---
    for path in ["data/raw/example.jsonl", "artifacts/test.txt"]:
        r = _git("check-ignore", "-v", path)
        if r.returncode != 0:
            errors += fail(f"{path} is NOT git-ignored")
    ok("data/raw/ and artifacts/ are git-ignored")

    # --- Paid-data authorization defaults to false ---
    import yaml  # noqa: E402

    policy = yaml.safe_load((ROOT / "configs" / "spending_policy.yaml").read_text())
    if policy["paid_data"]["default_authorization"] is not False:
        errors += fail("paid_data.default_authorization is not False")
    else:
        ok("paid_data.default_authorization is False")
    if policy["core"]["research_budget_usd"] != 0:
        errors += fail("core.research_budget_usd is not 0")
    else:
        ok("core.research_budget_usd is 0")

    # --- No prohibited implementation files ---
    for path in PROHIBITED_PATHS:
        if (ROOT / path).exists():
            errors += fail(f"prohibited file exists: {path}")
    ok("no prohibited later-stage implementation files")

    # --- No prohibited imports in tracked Python files ---
    r = _git(
        "grep",
        "-n",
        "-I",
        r"import (numpy|scipy|sklearn|torch|pandas|gymnasium|stable_baselines3)"
        r"|from (numpy|scipy|sklearn|torch|pandas|gymnasium|stable_baselines3)",
        "--",
        "*.py",
    )
    if r.stdout.strip():
        errors += fail(f"prohibited imports found:\n{r.stdout}")
    else:
        ok("no prohibited imports in tracked Python files")

    # --- No credential strings in tracked text files ---
    r = _git("ls-files", "-z")
    tracked = [f for f in r.stdout.strip("\0").split("\0") if f]
    for rel in tracked:
        fpath = ROOT / rel
        if not fpath.is_file():
            continue
        if fpath.suffix in (".pt", ".pth", ".ckpt", ".png", ".jpg", ".lock"):
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Detect Databento key patterns that aren't example placeholders
        if (
            "db-" in content
            and "DATABENTO_API_KEY" in content
            and "db-YOUR" not in content
            and "db-example" not in content
        ):
            errors += fail(f"suspicious Databento key pattern in {rel} (manual review needed)")
    ok("no suspicious credential patterns")

    # --- No stage terminology in tracked files ---
    r = _git("ls-files", "-z")
    for rel in tracked:
        # Skip the policy script itself — it defines the banned terms
        if rel == "scripts/check_repository_policy.py":
            continue
        fpath = ROOT / rel
        if not fpath.is_file():
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace").lower()
        except Exception:
            continue
        for term in STAGE_TERMS:
            if term in content:
                errors += fail(f"stage terminology '{term}' found in {rel}")
    ok("no stage terminology in tracked files")

    if errors:
        print(f"\n{errors} policy check(s) FAILED")
    else:
        print("\nAll policy checks PASSED")
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
