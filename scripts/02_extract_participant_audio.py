"""
02_extract_participant_audio.py
--------------------------------
Uses *PAR: timestamps from .cha files to slice each recording down to
participant-speech-only, then concatenates those segments into a single
.wav per session.

Output structure mirrors the .cha transcript tree:

    Pitt/intermediate/participant_only_audio/
        Control/
            cookie/
                002-0.wav
                002-1.wav
                ...
        Dementia/
            cookie/
                001-0.wav
                ...

Usage
-----
    # Extract everything
    python scripts/02_extract_participant_audio.py

    # Dry-run (show matches without writing audio)
    python scripts/02_extract_participant_audio.py --dry-run

    # Only cookie task
    python scripts/02_extract_participant_audio.py --task cookie
"""

import argparse
import sys
from pathlib import Path

# ── Resolve project root so the script works from any cwd ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.audio_extractor import extract_participant_audio  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
CHA_DIR    = PROJECT_ROOT / "Pitt" / "raw" / "Pitt-transcript"
AUDIO_DIR  = PROJECT_ROOT / "Pitt" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "Pitt" / "intermediate" / "participant_only_audio"

SAMPLE_RATE = 16_000   # Hz — standard for speech processing


def main(
    dry_run:     bool       = False,
    task_filter: str | None = None,
    skip_existing: bool     = True,
) -> None:

    pattern   = "**/*.cha"
    cha_files = sorted(CHA_DIR.glob(pattern))

    if not cha_files:
        print(f"[ERROR] No .cha files found under {CHA_DIR}")
        sys.exit(1)

    # Optional task filter (e.g. "cookie")
    if task_filter:
        cha_files = [f for f in cha_files if task_filter.lower() in f.parts]
        if not cha_files:
            print(f"[ERROR] No .cha files matched task filter '{task_filter}'")
            sys.exit(1)

    # Build (stem, task) → audio path lookup
    # Key is (file_stem, task_folder_name) e.g. ("016-0", "cookie")
    # This avoids collisions where the same participant ID exists across tasks.
    audio_lookup: dict[tuple[str, str], Path] = {}
    for mp3 in AUDIO_DIR.rglob("*.mp3"):
        # mp3 lives at e.g. Pitt/raw/dementia/cookie/016-0.mp3
        # task = the folder immediately containing the file
        task_folder = mp3.parent.name
        audio_lookup[(mp3.stem, task_folder)] = mp3

    print(f"Found {len(cha_files)} .cha files")
    print(f"Found {len(audio_lookup)} .mp3 files (keyed by stem+task)")
    if task_filter:
        print(f"Task filter : '{task_filter}'")
    if dry_run:
        print("[DRY-RUN] No files will be written.\n")
    if skip_existing:
        print("[INFO] Skipping already-written outputs (use --force to overwrite).\n")

    written  = 0
    skipped  = 0
    errors   = 0
    existing = 0
    error_list: list[str] = []

    for cha_file in cha_files:
        stem        = cha_file.stem           # e.g. "016-0"
        task_folder = cha_file.parent.name    # e.g. "cookie"
        key         = (stem, task_folder)

        if key not in audio_lookup:
            print(f"  [SKIP] No audio found for {cha_file.name} (task={task_folder})")
            skipped += 1
            continue

        audio_file = audio_lookup[key]
        rel        = cha_file.relative_to(CHA_DIR)
        out_file   = OUTPUT_DIR / rel.with_suffix(".wav")

        if dry_run:
            print(f"  would write → {out_file.relative_to(PROJECT_ROOT)}")
            continue

        if skip_existing and out_file.exists():
            existing += 1
            continue

        try:
            extract_participant_audio(
                cha_path   = cha_file,
                audio_path = audio_file,
                out_path   = out_file,
                sr_out     = SAMPLE_RATE,
            )
            written += 1
            print(f"  ✓  {stem}")
        except Exception as exc:
            msg = f"{task_folder}/{stem}  —  {exc}"
            print(f"  ✗  {msg}")
            error_list.append(msg)
            errors += 1

    if not dry_run:
        print(f"\n{'─' * 60}")
        print(f"Written  : {written}")
        if existing:
            print(f"Existing : {existing}  (skipped, already done)")
        print(f"Skipped  : {skipped}  (no matching .mp3)")
        print(f"Errors   : {errors}")
        if error_list:
            print("\nFailed files:")
            for e in error_list:
                print(f"  ✗  {e}")
        print(f"Output   : {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
        print(f"Rate     : {SAMPLE_RATE} Hz mono .wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract participant-only audio from Pitt Corpus recordings."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without writing anything.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        metavar="TASK",
        help="Only process files inside a sub-folder with this name (e.g. cookie).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite already-written output files (default: skip existing).",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, task_filter=args.task, skip_existing=not args.force)
