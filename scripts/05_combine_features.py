#!/usr/bin/env python
"""
05_combine_features.py
-----------------------
Merge acoustic, syntactic, and lexical feature CSVs into a single combined
CSV per task in Pitt/processed/combined/.

Inputs (all optional — present columns are included)
------
    Pitt/processed/acoustic/{task}_features.csv    — stem, label, 88 acoustic
    Pitt/processed/syntactic/{task}_features.csv   — stem, label, 14-23 syntactic
    Pitt/processed/lexical/{task}_features.csv     — stem, label, 10+ lexical

Output
------
    Pitt/processed/combined/{task}_features.csv

CSV schema
----------
    stem       — e.g. "001-0"
    label      — "Control" or "Dementia"
    [acoustic features...]
    [syntactic features...]
    [lexical features...]

Notes
-----
- The merge key is (stem, label).  Rows present in only a subset of the
  source CSVs are kept (outer join) with NaN for missing features.
- If a source CSV doesn't exist for a task it is silently skipped.

Run
---
    conda run -n dementia-detection python scripts/05_combine_features.py
    conda run -n dementia-detection python scripts/05_combine_features.py --task cookie
    conda run -n dementia-detection python scripts/05_combine_features.py --force
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "Pitt" / "processed"

_ALL_TASKS = ("cookie", "fluency", "recall", "sentence")

# Source folder names in merge order
_CATEGORIES = ("acoustic", "syntactic", "lexical")

try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "pandas not found. Run:  .venv/bin/pip install pandas"
    ) from e


# ---------------------------------------------------------------------------
# Core merge function
# ---------------------------------------------------------------------------

def combine_task(task: str, processed_dir: Path, force: bool = False) -> None:
    """Merge all available feature CSVs for a single task."""
    out_dir = processed_dir / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{task}_features.csv"

    if out_csv.exists() and not force:
        print(f"  [skip] {out_csv.relative_to(processed_dir.parent)} already exists")
        return

    merged: "pd.DataFrame | None" = None
    found: list[str] = []

    for category in _CATEGORIES:
        src = processed_dir / category / f"{task}_features.csv"
        if not src.exists():
            print(f"  [warn] missing {category}/{task}_features.csv — skipping")
            continue

        df = pd.read_csv(src)
        # Normalise key columns
        df["stem"] = df["stem"].astype(str)
        df["label"] = df["label"].astype(str)

        # Drop 'filename' if present (from linguistic.py output)
        if "filename" in df.columns:
            df = df.drop(columns=["filename"])

        # Rename feature columns to avoid clashes across categories
        feat_cols = [c for c in df.columns if c not in ("stem", "label")]
        df = df.rename(columns={c: f"{category}__{c}" for c in feat_cols})

        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=["stem", "label"], how="outer")

        found.append(category)

    if merged is None or merged.empty:
        print(f"  [warn] Nothing to combine for task '{task}'")
        return

    # Sort for determinism
    merged = merged.sort_values(["label", "stem"]).reset_index(drop=True)

    merged.to_csv(out_csv, index=False)
    print(
        f"  Combined {' + '.join(found):30s}  →  {out_csv.relative_to(PROJECT_ROOT)}"
        f"  ({len(merged)} rows, {len(merged.columns)} cols)"
    )


def combine_all(
    processed_dir: Path,
    tasks: list[str],
    force: bool = False,
) -> None:
    for task in tasks:
        sep = "═" * 60
        print(f"\n{sep}\nTask: {task}\n{sep}")
        combine_task(task, processed_dir, force=force)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge acoustic + syntactic + lexical feature CSVs."
    )
    p.add_argument(
        "--task",
        choices=list(_ALL_TASKS),
        default=None,
        help="Merge a single task (default: all four tasks).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing combined CSVs.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = [args.task] if args.task else list(_ALL_TASKS)

    print(f"Processed dir: {PROCESSED_DIR}")
    print(f"Tasks        : {tasks}")
    print(f"Force        : {args.force}")

    combine_all(PROCESSED_DIR, tasks, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
