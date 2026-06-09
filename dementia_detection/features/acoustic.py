"""
acoustic.py
-----------
Batch eGeMAPS v02 acoustic feature extraction using openSMILE.

Reads participant-only WAV files from:
    Pitt/intermediate/participant_only_audio/
        {Control,Dementia}/{task}/*.wav

Outputs one CSV per task to:
    Pitt/processed/acoustic/{task}_features.csv

CSV schema
----------
    stem      — file stem, e.g. "001-0"
    label     — "Control" or "Dementia"
    [88 eGeMAPS Functionals features]

Public API
----------
    extract_egemaps(wav_path)  -> dict  (88 features, keyed by feature name)
    extract_all_acoustic(audio_dir, output_dir, tasks=None, force=False)

Usage (CLI)
-----------
    .venv/bin/python -m src.feature_extraction.acoustic \\
        [--task cookie] [--force]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# opensmile must be importable — lives in .venv, NOT the conda env
try:
    import opensmile
except ImportError as e:
    raise ImportError(
        "opensmile not found. Run:  .venv/bin/pip install opensmile"
    ) from e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_TASKS = ("cookie", "fluency", "recall", "sentence")

# Lazy-initialised global smile instance (shared across calls in a process)
_SMILE: "opensmile.Smile | None" = None


def _get_smile() -> "opensmile.Smile":
    """Return (and lazily create) the shared openSMILE extractor."""
    global _SMILE
    if _SMILE is None:
        _SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE


# ---------------------------------------------------------------------------
# Single-file extraction
# ---------------------------------------------------------------------------

def extract_egemaps(wav_path: str | Path) -> dict:
    """
    Extract eGeMAPS v02 Functional features from a WAV file.

    Parameters
    ----------
    wav_path : str | Path
        Path to a WAV audio file (16 kHz mono recommended).

    Returns
    -------
    dict
        88 feature names → float values.
        Returns an empty dict {} on failure.
    """
    wav_path = Path(wav_path)
    try:
        smile = _get_smile()
        df = smile.process_file(str(wav_path))
        # process_file returns a DataFrame with one row; flatten to dict
        return {col: float(df.iloc[0][col]) for col in df.columns}
    except Exception as exc:
        print(f"[acoustic] ERROR extracting {wav_path.name}: {exc}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_all_acoustic(
    audio_dir: str | Path,
    output_dir: str | Path,
    tasks: tuple[str, ...] | list[str] | None = None,
    force: bool = False,
) -> None:
    """
    Batch-extract eGeMAPS features from all participant WAVs.

    Directory layout expected under ``audio_dir``::

        {Control,Dementia}/{task}/*.wav

    For each task one CSV is written to ``output_dir/acoustic/{task}_features.csv``
    with columns: stem, label, [88 eGeMAPS features].

    Parameters
    ----------
    audio_dir : str | Path
        Root of the participant-only audio tree
        (e.g. ``Pitt/intermediate/participant_only_audio``).
    output_dir : str | Path
        Root of the processed outputs (e.g. ``Pitt/processed``).
        The ``acoustic/`` sub-folder is created automatically.
    tasks : sequence of str, optional
        Which tasks to process.  Defaults to all four:
        cookie, fluency, recall, sentence.
    force : bool
        Re-extract even if the output CSV already exists.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    tasks = list(tasks) if tasks else list(_ALL_TASKS)

    acoustic_out = output_dir / "acoustic"
    acoustic_out.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        sep = "═" * 60
        print(f"\n{sep}\nTask: {task}\n{sep}")

        out_csv = acoustic_out / f"{task}_features.csv"
        if out_csv.exists() and not force:
            print(f"  [skip] {out_csv} already exists (use --force to overwrite)")
            continue

        rows: list[dict] = []
        errors = 0
        feature_cols: list[str] | None = None

        for label in ("Control", "Dementia"):
            task_dir = audio_dir / label / task
            if not task_dir.is_dir():
                continue

            wav_files = sorted(task_dir.glob("*.wav"))
            print(f"\n  {label}: {len(wav_files)} files")

            for wav in wav_files:
                stem = wav.stem
                feats = extract_egemaps(wav)
                if not feats:
                    print(f"    ✗  {wav.name}")
                    errors += 1
                    continue

                if feature_cols is None:
                    feature_cols = list(feats.keys())

                row = {"stem": stem, "label": label, **feats}
                rows.append(row)
                print(f"    ✓  {wav.name}")

        if not rows:
            print(f"  [warn] No features extracted for task '{task}'")
            continue

        # Write CSV
        cols = ["stem", "label"] + (feature_cols or [])
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"\n  Acoustic written: {len(rows)} | errors: {errors}"
            f" → {out_csv}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract eGeMAPS acoustic features from participant WAVs."
    )
    p.add_argument(
        "--task",
        choices=list(_ALL_TASKS),
        default=None,
        help="Process a single task (default: all four tasks).",
    )
    p.add_argument(
        "--audio-dir",
        default=None,
        help="Root of the participant audio tree. "
             "Default: Pitt/intermediate/participant_only_audio (relative to project root).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Root of the processed outputs. "
             "Default: Pitt/processed (relative to project root).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-extract and overwrite existing CSV files.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    # Resolve paths relative to project root (two levels up from this file)
    _root = Path(__file__).resolve().parents[2]
    _audio_dir = Path(args.audio_dir) if args.audio_dir else _root / "Pitt/intermediate/participant_only_audio"
    _output_dir = Path(args.output_dir) if args.output_dir else _root / "Pitt/processed"
    _tasks = [args.task] if args.task else list(_ALL_TASKS)

    extract_all_acoustic(
        audio_dir=_audio_dir,
        output_dir=_output_dir,
        tasks=_tasks,
        force=args.force,
    )
