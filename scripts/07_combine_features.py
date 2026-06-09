#!/usr/bin/env python
"""
07_combine_features.py
-----------------------
Merge per-modality feature CSVs (acoustic, syntactic, lexical) into a
single combined CSV per task at Pitt/processed/combined/.

All logic lives in dementia_detection.data.combine.

Usage
-----
    python scripts/07_combine_features.py
    python scripts/07_combine_features.py --task cookie
    python scripts/07_combine_features.py --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import TASKS
from dementia_detection.data.combine import combine_all, combine_task


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge per-modality feature CSVs.")
    p.add_argument("--task", choices=list(TASKS), default=None,
                   help="Merge a single task (default: all tasks).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing combined CSVs.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.task:
        combine_task(args.task, force=args.force)
    else:
        combine_all(force=args.force)


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
