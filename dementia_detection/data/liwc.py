"""
liwc.py
-------
Integrate LIWC-22 output CSVs into combined feature tables.

LIWC-22 must be run externally on the cleaned participant transcripts and
its per-task output CSVs placed at::

    Pitt/processed/LIWC/{Control,Dementia}/LIWC-22 Results - {task} - LIWC Analysis.csv

All LIWC feature columns are prefixed with ``liwc__`` before merging.

Public API
----------
load_liwc_for_task(task)          -> pd.DataFrame | None
    Load and concatenate Control + Dementia LIWC CSVs for one task.

integrate_task(task, combined_dir, liwc_dir) -> bool
    Merge LIWC features into an existing combined CSV in-place.

integrate_all(tasks, combined_dir, liwc_dir) -> None
    Convenience wrapper: run integrate_task for every task.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.config import COMBINED_DIR, LIWC_DIR, LIWC_META, TASKS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_liwc_for_task(
    task: str,
    liwc_dir: Path = LIWC_DIR,
) -> "pd.DataFrame | None":
    """
    Load and concatenate Control + Dementia LIWC-22 CSVs for *task*.

    Returns a DataFrame with columns ``[stem, liwc__*, …]``, or ``None``
    when the expected CSV files do not exist.
    """
    ctrl_path = liwc_dir / "Control" / f"LIWC-22 Results - {task} - LIWC Analysis.csv"
    dem_path  = liwc_dir / "Dementia" / f"LIWC-22 Results - {task} - LIWC Analysis.csv"

    if not ctrl_path.exists() or not dem_path.exists():
        return None

    liwc = pd.concat(
        [pd.read_csv(ctrl_path), pd.read_csv(dem_path)],
        ignore_index=True,
    )

    # "021-3.txt" → "021-3"
    liwc["stem"] = liwc["Filename"].str.replace(r"\.txt$", "", regex=True)

    feature_cols = [c for c in liwc.columns if c not in LIWC_META and c != "stem"]
    liwc = liwc.rename(columns={c: f"liwc__{c}" for c in feature_cols})

    return liwc[["stem"] + [f"liwc__{c}" for c in feature_cols]]


def integrate_task(
    task: str,
    combined_dir: Path = COMBINED_DIR,
    liwc_dir: Path = LIWC_DIR,
) -> bool:
    """
    Merge LIWC-22 features into ``combined_dir/{task}_features.csv`` in-place.

    Idempotent: existing ``liwc__*`` columns are dropped and re-merged on
    every call.  Returns *True* when the merge succeeds, *False* when either
    source file is missing.
    """
    combined_path = combined_dir / f"{task}_features.csv"
    if not combined_path.exists():
        print(f"  [{task}] combined CSV not found — skip")
        return False

    liwc_df = load_liwc_for_task(task, liwc_dir)
    if liwc_df is None:
        print(f"  [{task}] LIWC output not found — skip")
        return False

    combined = pd.read_csv(combined_path)

    # Drop any existing liwc__ columns so the merge is idempotent
    existing = [c for c in combined.columns if c.startswith("liwc__")]
    if existing:
        print(f"  [{task}] dropping {len(existing)} stale liwc__ columns before re-merge")
        combined = combined.drop(columns=existing)

    before_rows = len(combined)
    merged = combined.merge(liwc_df, on="stem", how="left")

    liwc_cols = [c for c in merged.columns if c.startswith("liwc__")]
    n_missing = merged[liwc_cols].isna().any(axis=1).sum()

    print(
        f"  [{task}] {before_rows} rows + {len(liwc_df)} LIWC rows → "
        f"{len(liwc_cols)} liwc__ features  |  {n_missing} rows missing LIWC"
    )

    merged.to_csv(combined_path, index=False)
    print(f"  [{task}] saved → {combined_path}  ({len(merged.columns)} cols total)")
    return True


def integrate_all(
    tasks: tuple[str, ...] = TASKS,
    combined_dir: Path = COMBINED_DIR,
    liwc_dir: Path = LIWC_DIR,
) -> None:
    """Run :func:`integrate_task` for every task in *tasks*."""
    print("=" * 60)
    print("Integrating LIWC-22 features into combined CSVs")
    print("=" * 60)
    for task in tasks:
        integrate_task(task, combined_dir, liwc_dir)
    print("\nDone.")
