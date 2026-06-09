#!/usr/bin/env python
"""
08_add_response_length.py
--------------------------
Add two response-length features to the combined feature CSVs:

    response_length__word_count        — number of whitespace-separated tokens
                                         in the participant-only cleaned transcript
    response_length__audio_duration_sec — duration (seconds) of the participant-only
                                         WAV after silence/interviewer removal

These are added as a new prefix group so they can be included in any feature
group combination without requiring LIWC.  Note: liwc__WC already captures raw
word count from LIWC-22 for samples where LIWC was run; this feature provides
the same quantity independently of LIWC, and also covers non-LIWC tasks.

Run
---
    conda run -n dementia-detection python scripts/08_add_response_length.py
    conda run -n dementia-detection python scripts/08_add_response_length.py --task cookie
"""

import argparse
import wave
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_DIR = PROJECT_ROOT / "Pitt" / "intermediate" / "cleaned_transcripts"
AUDIO_DIR      = PROJECT_ROOT / "Pitt" / "intermediate" / "participant_only_audio"
COMBINED_DIR   = PROJECT_ROOT / "Pitt" / "processed" / "combined"
# ---------------------------------------------------------------------------


def _wav_duration(path: Path) -> float | None:
    """Return duration in seconds of a WAV file, or None on error."""
    try:
        with wave.open(str(path)) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def _word_count(path: Path) -> int | None:
    """Return number of whitespace-separated tokens in a text file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return len(text.split())
    except Exception:
        return None


def build_response_length(task: str) -> pd.DataFrame:
    """
    Walk Control + Dementia directories for the given task and build a
    DataFrame with columns: stem, response_length__word_count,
    response_length__audio_duration_sec.
    """
    rows = []
    for label in ("Control", "Dementia"):
        txt_dir = TRANSCRIPT_DIR / label / task
        wav_dir = AUDIO_DIR      / label / task

        if not txt_dir.exists():
            print(f"  [WARN] transcript dir missing: {txt_dir}")
            continue

        for txt_path in sorted(txt_dir.glob("*.txt")):
            stem = txt_path.stem
            wc   = _word_count(txt_path)

            wav_path = wav_dir / f"{stem}.wav"
            dur = _wav_duration(wav_path) if wav_path.exists() else None

            rows.append({
                "stem":                              stem,
                "response_length__word_count":       wc,
                "response_length__audio_duration_sec": dur,
            })

    return pd.DataFrame(rows)


def add_to_combined(task: str) -> None:
    combined_path = COMBINED_DIR / f"{task}_features.csv"
    if not combined_path.exists():
        print(f"  [SKIP] combined CSV not found: {combined_path}")
        return

    df = pd.read_csv(combined_path)

    # Drop old response_length columns if re-running
    old_cols = [c for c in df.columns if c.startswith("response_length__")]
    if old_cols:
        df = df.drop(columns=old_cols)
        print(f"  [INFO] dropped {len(old_cols)} existing response_length__ columns")

    rl = build_response_length(task)
    print(f"  Response-length rows built: {len(rl)} "
          f"| WC NaN={rl['response_length__word_count'].isna().sum()} "
          f"| DUR NaN={rl['response_length__audio_duration_sec'].isna().sum()}")

    before = len(df)
    df = df.merge(rl, on="stem", how="left")
    assert len(df) == before, "Row count changed after merge — check stem key"

    wc_nan  = df["response_length__word_count"].isna().sum()
    dur_nan = df["response_length__audio_duration_sec"].isna().sum()
    print(f"  Merged -> {len(df)} rows | WC NaN={wc_nan} | DUR NaN={dur_nan}")

    df.to_csv(combined_path, index=False)
    print(f"  Saved  -> {combined_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None,
                        help="Single task name (default: all tasks with combined CSV)")
    args = parser.parse_args()

    if args.task:
        tasks = [args.task]
    else:
        tasks = [p.stem.replace("_features", "")
                 for p in sorted(COMBINED_DIR.glob("*_features.csv"))]

    print(f"Tasks: {tasks}")
    for task in tasks:
        print(f"\n── {task} ──")
        add_to_combined(task)

    print("\nDone.")


if __name__ == "__main__":
    main()
