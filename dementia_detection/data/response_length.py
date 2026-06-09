"""
response_length.py
------------------
Compute and attach two response-length features to combined feature tables:

    response_length__word_count          — whitespace-token count from the
                                           participant-only cleaned transcript
    response_length__audio_duration_sec  — duration of the participant-only
                                           WAV file after silence removal

These features are independent of LIWC and cover all four tasks.

Public API
----------
wav_duration(path) -> float | None
    Return duration in seconds for a WAV file, or None on error.

word_count(path)   -> int | None
    Return whitespace-token count of a text file, or None on error.

build_response_length(task, transcript_dir, audio_dir) -> pd.DataFrame
    Walk Control + Dementia directories for *task* and return a DataFrame
    with columns ``[stem, response_length__word_count,
    response_length__audio_duration_sec]``.

add_to_combined(task, combined_dir, transcript_dir, audio_dir) -> None
    Merge response-length features into an existing combined CSV in-place.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pandas as pd

from config.config import AUDIO_DIR, COMBINED_DIR, TASKS, TRANSCRIPT_DIR


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def wav_duration(path: Path) -> "float | None":
    """Return WAV duration in seconds, or *None* on any error."""
    try:
        with wave.open(str(path)) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def word_count(path: Path) -> "int | None":
    """Return number of whitespace-separated tokens in a text file."""
    try:
        return len(path.read_text(encoding="utf-8", errors="replace").split())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_response_length(
    task: str,
    transcript_dir: Path = TRANSCRIPT_DIR,
    audio_dir: Path = AUDIO_DIR,
) -> pd.DataFrame:
    """
    Build a response-length DataFrame for all participants in *task*.

    Returns
    -------
    pd.DataFrame with columns:
        ``stem``, ``response_length__word_count``,
        ``response_length__audio_duration_sec``
    """
    rows: list[dict] = []
    for label in ("Control", "Dementia"):
        txt_dir = transcript_dir / label / task
        wav_dir = audio_dir / label / task

        if not txt_dir.exists():
            print(f"  [WARN] transcript directory missing: {txt_dir}")
            continue

        for txt_path in sorted(txt_dir.glob("*.txt")):
            stem     = txt_path.stem
            wc       = word_count(txt_path)
            wav_path = wav_dir / f"{stem}.wav"
            dur      = wav_duration(wav_path) if wav_path.exists() else None

            rows.append({
                "stem":                                stem,
                "response_length__word_count":         wc,
                "response_length__audio_duration_sec": dur,
            })

    return pd.DataFrame(rows)


def add_to_combined(
    task: str,
    combined_dir: Path = COMBINED_DIR,
    transcript_dir: Path = TRANSCRIPT_DIR,
    audio_dir: Path = AUDIO_DIR,
) -> None:
    """
    Merge response-length features into ``combined_dir/{task}_features.csv``
    in-place.  Idempotent: existing ``response_length__*`` columns are
    dropped and recomputed on every call.
    """
    combined_path = combined_dir / f"{task}_features.csv"
    if not combined_path.exists():
        print(f"  [SKIP] combined CSV not found: {combined_path}")
        return

    df = pd.read_csv(combined_path)

    # Drop stale columns so the update is idempotent
    old_cols = [c for c in df.columns if c.startswith("response_length__")]
    if old_cols:
        df = df.drop(columns=old_cols)
        print(f"  [INFO] dropped {len(old_cols)} stale response_length__ columns")

    rl = build_response_length(task, transcript_dir, audio_dir)
    print(
        f"  Built {len(rl)} rows"
        f"  | WC NaN={rl['response_length__word_count'].isna().sum()}"
        f"  | DUR NaN={rl['response_length__audio_duration_sec'].isna().sum()}"
    )

    before = len(df)
    df = df.merge(rl, on="stem", how="left")
    assert len(df) == before, "Row count changed after merge — check stem key"

    df.to_csv(combined_path, index=False)
    print(f"  Saved  → {combined_path}")


def add_all(
    tasks: tuple[str, ...] = TASKS,
    combined_dir: Path = COMBINED_DIR,
    transcript_dir: Path = TRANSCRIPT_DIR,
    audio_dir: Path = AUDIO_DIR,
) -> None:
    """Run :func:`add_to_combined` for every task that has a combined CSV."""
    for task in tasks:
        path = combined_dir / f"{task}_features.csv"
        if path.exists():
            print(f"\n── {task} ──")
            add_to_combined(task, combined_dir, transcript_dir, audio_dir)
