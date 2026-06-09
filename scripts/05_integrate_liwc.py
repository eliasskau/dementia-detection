"""
07_integrate_liwc.py
--------------------
Reads LIWC-22 output CSVs from Pitt/processed/LIWC/{Control,Dementia}/
and merges them into the existing combined feature CSVs in
Pitt/processed/combined/, prefixing all LIWC columns with "liwc__".

Currently only cookie task has LIWC output — the script will skip
tasks where LIWC files are not found.

Usage:
    python scripts/07_integrate_liwc.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).resolve().parents[1]
LIWC_DIR    = ROOT / "Pitt/processed/LIWC"
COMBINED    = ROOT / "Pitt/processed/combined"

# Columns that are metadata in LIWC output — NOT features
_LIWC_META = {"Filename", "Segment"}

TASKS = ("cookie", "fluency", "recall", "sentence")


def load_liwc_for_task(task: str) -> pd.DataFrame | None:
    """
    Load and concatenate Control + Dementia LIWC CSVs for a given task.
    Returns None if files are not found.
    """
    ctrl_path = LIWC_DIR / "Control"   / f"LIWC-22 Results - {task} - LIWC Analysis.csv"
    dem_path  = LIWC_DIR / "Dementia"  / f"LIWC-22 Results - {task} - LIWC Analysis.csv"

    if not ctrl_path.exists() or not dem_path.exists():
        return None

    ctrl = pd.read_csv(ctrl_path)
    dem  = pd.read_csv(dem_path)
    liwc = pd.concat([ctrl, dem], ignore_index=True)

    # Derive stem from Filename (e.g. "021-3.txt" → "021-3")
    liwc["stem"] = liwc["Filename"].str.replace(r"\.txt$", "", regex=True)

    # Drop metadata columns, keep only feature columns
    feature_cols = [c for c in liwc.columns if c not in _LIWC_META and c != "stem"]

    # Prefix all LIWC feature columns
    rename_map = {c: f"liwc__{c}" for c in feature_cols}
    liwc = liwc.rename(columns=rename_map)

    return liwc[["stem"] + [f"liwc__{c}" for c in feature_cols]]


def integrate_task(task: str) -> bool:
    combined_path = COMBINED / f"{task}_features.csv"
    if not combined_path.exists():
        print(f"  [{task}] combined CSV not found — skip")
        return False

    liwc_df = load_liwc_for_task(task)
    if liwc_df is None:
        print(f"  [{task}] LIWC output not found — skip")
        return False

    combined = pd.read_csv(combined_path)

    # Check if already merged
    if any(c.startswith("liwc__") for c in combined.columns):
        print(f"  [{task}] LIWC already merged — re-merging (overwrite)")
        combined = combined[[c for c in combined.columns if not c.startswith("liwc__")]]

    before_rows = len(combined)
    merged = combined.merge(liwc_df, on="stem", how="left")

    liwc_cols  = [c for c in merged.columns if c.startswith("liwc__")]
    n_missing  = merged[liwc_cols].isna().any(axis=1).sum()

    print(f"  [{task}] {before_rows} rows → merged {len(liwc_df)} LIWC rows → "
          f"{len(liwc_cols)} liwc__ features  |  {n_missing} rows missing LIWC")

    merged.to_csv(combined_path, index=False)
    print(f"  [{task}] saved → {combined_path}  (total cols: {len(merged.columns)})")
    return True


def main() -> None:
    print("=" * 60)
    print("Integrating LIWC-22 features into combined CSVs")
    print("=" * 60)
    for task in TASKS:
        integrate_task(task)
    print("\nDone.")


if __name__ == "__main__":
    main()
