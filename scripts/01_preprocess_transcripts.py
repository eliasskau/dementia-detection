"""
01_preprocess_transcripts.py
-----------------------------
Converts all Pitt Corpus .cha transcript files to cleaned plain-text,
outputting one .txt file per .cha file under:

    Pitt/intermediate/cleaned_transcripts/
        Control/
            cookie/
                002-0.txt
                ...
        Dementia/
            cookie/
                ...

The folder structure from Pitt/Pitt-transcript/ is mirrored exactly.

Each output .txt file contains only the participant's (*PAR) spoken words,
one utterance per line, with all CHAT metadata stripped.

Usage
-----
    python scripts/01_preprocess_transcripts.py

    # Dry-run (list files without writing):
    python scripts/01_preprocess_transcripts.py --dry-run

    # Limit to a single sub-task folder, e.g. cookie only:
    python scripts/01_preprocess_transcripts.py --task cookie
"""

import argparse
import sys
from pathlib import Path

# ── Resolve project root so the script works from any cwd ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.cha_to_txt import cha_to_txt  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = PROJECT_ROOT / "Pitt" / "raw" / "Pitt-transcript"
OUTPUT_DIR = PROJECT_ROOT / "Pitt" / "intermediate" / "cleaned_transcripts"


def main(dry_run: bool = False, task_filter: str | None = None) -> None:
    # Collect all .cha files recursively
    all_cha = sorted(INPUT_DIR.glob("**/*.cha"))

    if not all_cha:
        print(f"[ERROR] No .cha files found under {INPUT_DIR}")
        sys.exit(1)

    # Optionally filter by task sub-folder name (e.g. "cookie")
    if task_filter:
        all_cha = [f for f in all_cha if task_filter.lower() in f.parts]
        if not all_cha:
            print(f"[ERROR] No .cha files matched task filter '{task_filter}'")
            sys.exit(1)

    print(f"Found {len(all_cha)} .cha files under {INPUT_DIR}")
    if task_filter:
        print(f"Task filter applied: '{task_filter}'")
    if dry_run:
        print("[DRY-RUN] No files will be written.\n")

    converted = 0
    skipped   = 0
    empty     = 0

    for cha_file in all_cha:
        # Mirror path under OUTPUT_DIR
        rel      = cha_file.relative_to(INPUT_DIR)
        out_file = OUTPUT_DIR / rel.with_suffix(".txt")

        if dry_run:
            print(f"  would write → {out_file.relative_to(PROJECT_ROOT)}")
            continue

        # Convert
        cha_to_txt(cha_file, out_file)

        # Check the output isn't empty (completely silent / interviewer-only file)
        content = out_file.read_text(encoding="utf-8").strip()
        if not content:
            empty += 1
            print(f"  [WARN] Empty output: {cha_file.name}")
        else:
            converted += 1

    if not dry_run:
        print(f"\n{'─'*60}")
        print(f"Converted : {converted}")
        print(f"Empty     : {empty}  (no *PAR speech found)")
        print(f"Output dir: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Pitt .cha transcripts to plain text.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be converted without writing anything.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        metavar="TASK",
        help="Only process files inside a sub-folder with this name (e.g. cookie).",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, task_filter=args.task)
