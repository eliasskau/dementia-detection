#!/usr/bin/env python
"""
05_integrate_liwc.py
--------------------
Merge LIWC-22 output CSVs into the combined feature tables.

LIWC-22 must be run externally on the cleaned participant transcripts
(Pitt/intermediate/cleaned_transcripts/) and the per-task output CSVs
placed at:

    Pitt/processed/LIWC/{Control,Dementia}/LIWC-22 Results - {task} - LIWC Analysis.csv

All logic lives in dementia_detection.data.liwc.

Usage
-----
    python scripts/05_integrate_liwc.py
    python scripts/05_integrate_liwc.py --task cookie
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import TASKS
from dementia_detection.data.liwc import integrate_all, integrate_task


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrate LIWC-22 features.")
    p.add_argument("--task", choices=list(TASKS), default=None,
                   help="Integrate a single task (default: all tasks).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.task:
        integrate_task(args.task)
    else:
        integrate_all()


if __name__ == "__main__":
    main()
