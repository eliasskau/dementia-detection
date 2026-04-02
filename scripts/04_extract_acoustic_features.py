#!/usr/bin/env python
"""
04_extract_acoustic_features.py
--------------------------------
Extract eGeMAPS v02 Functional features from participant-only WAV files
and write one CSV per task to Pitt/processed/acoustic/.

Input
-----
    Pitt/intermediate/participant_only_audio/{Control,Dementia}/{task}/*.wav

Output
------
    Pitt/processed/acoustic/cookie_features.csv
    Pitt/processed/acoustic/fluency_features.csv
    Pitt/processed/acoustic/recall_features.csv
    Pitt/processed/acoustic/sentence_features.csv

CSV schema: stem, label, [88 eGeMAPS feature columns]

Run:
    conda run -n dementia-detection python scripts/04_extract_acoustic_features.py
    conda run -n dementia-detection python scripts/04_extract_acoustic_features.py --task cookie
    conda run -n dementia-detection python scripts/04_extract_acoustic_features.py --force
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

AUDIO_DIR = PROJECT_ROOT / "Pitt" / "intermediate" / "participant_only_audio"
OUTPUT_DIR = PROJECT_ROOT / "Pitt" / "processed"

_ALL_TASKS = ("cookie", "fluency", "recall", "sentence")

# ---------------------------------------------------------------------------
# Import feature extractor
# ---------------------------------------------------------------------------
from src.feature_extraction.acoustic import extract_all_acoustic  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch eGeMAPS extraction from participant-only WAV files."
    )
    p.add_argument(
        "--task",
        choices=list(_ALL_TASKS),
        default=None,
        help="Process a single task only (default: all four tasks).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-extract and overwrite existing output CSVs.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    tasks = [args.task] if args.task else list(_ALL_TASKS)

    print(f"Audio dir : {AUDIO_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Tasks     : {tasks}")
    print(f"Force     : {args.force}")

    extract_all_acoustic(
        audio_dir=AUDIO_DIR,
        output_dir=OUTPUT_DIR,
        tasks=tasks,
        force=args.force,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
