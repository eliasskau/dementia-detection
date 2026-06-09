#!/usr/bin/env python
"""
06_add_response_length.py
--------------------------
Add two response-length features to the combined feature CSVs:

    response_length__word_count          — whitespace-token count from the
                                           participant-only cleaned transcript
    response_length__audio_duration_sec  — duration of the participant-only
                                           WAV file after silence removal

All logic lives in dementia_detection.data.response_length.

Usage
-----
    python scripts/06_add_response_length.py
    python scripts/06_add_response_length.py --task cookie
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import TASKS
from dementia_detection.data.response_length import add_all, add_to_combined


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add response-length features.")
    p.add_argument("--task", choices=list(TASKS), default=None,
                   help="Process a single task (default: all tasks).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.task:
        add_to_combined(args.task)
    else:
        add_all()


if __name__ == "__main__":
    main()

