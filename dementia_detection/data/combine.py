"""
combine.py
----------
Merge per-modality feature CSVs (acoustic, syntactic, lexical) into a single
combined CSV per task.

Each modality column is prefixed with its category name (e.g. ``acoustic__``)
before merging so that column names remain unambiguous across modalities.

Source layout::

    Pitt/processed/acoustic/{task}_features.csv
    Pitt/processed/syntactic/{task}_features.csv
    Pitt/processed/lexical/{task}_features.csv

Output::

    Pitt/processed/combined/{task}_features.csv

Public API
----------
combine_task(task, processed_dir, force) -> None
    Merge all available modalities for a single task.

combine_all(tasks, processed_dir, force)  -> None
    Run combine_task for every task in *tasks*.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.config import CATEGORIES, PROCESSED_DIR, TASKS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def combine_task(
    task: str,
    processed_dir: Path = PROCESSED_DIR,
    force: bool = False,
) -> None:
    """
    Merge acoustic + syntactic + lexical feature CSVs for *task* into a
    single combined CSV.

    The merge key is ``(stem, label)``; an outer join is used so rows
    present in only a subset of source CSVs are retained (NaN for missing
    features).  Existing combined CSVs are skipped unless *force* is True.
    """
    out_dir = processed_dir / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{task}_features.csv"

    if out_csv.exists() and not force:
        print(f"  [skip] {out_csv.relative_to(processed_dir.parent)} already exists")
        return

    merged: "pd.DataFrame | None" = None
    found: list[str] = []

    for category in CATEGORIES:
        src = processed_dir / category / f"{task}_features.csv"
        if not src.exists():
            print(f"  [warn] missing {category}/{task}_features.csv — skipping")
            continue

        df = pd.read_csv(src)
        df["stem"]  = df["stem"].astype(str)
        df["label"] = df["label"].astype(str)

        # Drop auxiliary columns that appear in linguistic output
        df = df.drop(columns=["filename"], errors="ignore")

        # Prefix all feature columns to avoid cross-category clashes
        feat_cols = [c for c in df.columns if c not in ("stem", "label")]
        df = df.rename(columns={c: f"{category}__{c}" for c in feat_cols})

        merged = df if merged is None else pd.merge(
            merged, df, on=["stem", "label"], how="outer"
        )
        found.append(category)

    if merged is None or merged.empty:
        print(f"  [warn] nothing to combine for task '{task}'")
        return

    merged = merged.sort_values(["label", "stem"]).reset_index(drop=True)
    merged.to_csv(out_csv, index=False)
    print(
        f"  Combined {' + '.join(found):30s}  →  {out_csv.relative_to(processed_dir.parent)}"
        f"  ({len(merged)} rows, {len(merged.columns)} cols)"
    )


def combine_all(
    tasks: tuple[str, ...] = TASKS,
    processed_dir: Path = PROCESSED_DIR,
    force: bool = False,
) -> None:
    """Run :func:`combine_task` for every task in *tasks*."""
    for task in tasks:
        sep = "═" * 60
        print(f"\n{sep}\nTask: {task}\n{sep}")
        combine_task(task, processed_dir, force=force)
