"""
audio_extractor.py
------------------
Parses *PAR: timestamps from a .cha file and extracts only the participant's
audio segments from the corresponding .mp3, concatenating them into a single
output .wav file.

Timestamps in CHAT files appear at the end of *PAR: lines in the form:
    *PAR:   some words here . 2821_3211
where 2821 = segment start (ms) and 3211 = segment end (ms).

Speed: uses ffmpeg (via static-ffmpeg) to seek directly to each segment
rather than decoding the full file -- ~10x faster than librosa for mp3.

Public API
----------
extract_participant_audio(cha_path, audio_path, out_path=None) -> str
    Extract PAR-only audio for a single file pair.

extract_all(cha_dir, audio_dir, output_dir, recursive=True) -> list[str]
    Batch-extract PAR-only audio for all matched .cha / .mp3 pairs.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


# -- Locate ffmpeg (prefer static-ffmpeg wheel, fall back to system) ----------
def _ffmpeg_bin() -> str:
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except ImportError:
        pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install it with:\n"
            "  pip install static-ffmpeg"
        )
    return ffmpeg


_FFMPEG = None   # resolved lazily on first use


def _get_ffmpeg() -> str:
    global _FFMPEG
    if _FFMPEG is None:
        _FFMPEG = _ffmpeg_bin()
    return _FFMPEG


# -- Public API ----------------------------------------------------------------

def extract_participant_audio(
    cha_path,
    audio_path,
    out_path=None,
    sr_out: int = 16000,
) -> str:
    cha_path   = Path(cha_path)
    audio_path = Path(audio_path)

    if out_path is None:
        out_path = audio_path.with_suffix(".wav")
    out_path = Path(out_path)

    segments = _parse_par_timestamps(cha_path)
    if not segments:
        raise ValueError(f"No *PAR timestamps found in {cha_path.name}")

    ffmpeg = _get_ffmpeg()
    chunks = []

    for start_ms, end_ms in segments:
        duration_ms = end_ms - start_ms
        if duration_ms <= 0:
            continue

        start_s    = start_ms    / 1000.0
        duration_s = duration_ms / 1000.0

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            ffmpeg, "-y",
            "-ss", f"{start_s:.3f}",
            "-i",  str(audio_path),
            "-t",  f"{duration_s:.3f}",
            "-ac", "1",
            "-ar", str(sr_out),
            "-f",  "wav",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not Path(tmp_path).exists():
            Path(tmp_path).unlink(missing_ok=True)
            continue

        try:
            data, _ = sf.read(tmp_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if len(data) > 0:
                chunks.append(data)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if not chunks:
        raise ValueError(f"All PAR segments were empty after extraction: {cha_path.name}")

    participant_audio = np.concatenate(chunks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), participant_audio, sr_out, subtype="PCM_16")
    return str(out_path)


def extract_all(
    cha_dir,
    audio_dir,
    output_dir,
    recursive: bool = True,
    sr_out:    int  = 16000,
):
    cha_dir    = Path(cha_dir)
    audio_dir  = Path(audio_dir)
    output_dir = Path(output_dir)

    pattern   = "**/*.cha" if recursive else "*.cha"
    cha_files = sorted(cha_dir.glob(pattern))

    if not cha_files:
        print(f"No .cha files found under {cha_dir}")
        return []

    # Build lookup: (task_folder_lowercase, stem) -> mp3 path
    audio_lookup = {}
    for mp3 in audio_dir.rglob("*.mp3"):
        task = mp3.parent.name.lower()
        audio_lookup[(task, mp3.stem)] = mp3

    written, skipped, errors = [], [], []

    for cha_file in cha_files:
        stem = cha_file.stem
        task = cha_file.parent.name.lower()
        key  = (task, stem)

        if key not in audio_lookup:
            skipped.append(cha_file.name)
            continue

        audio_file = audio_lookup[key]
        rel        = cha_file.relative_to(cha_dir)
        out_file   = output_dir / rel.with_suffix(".wav")

        try:
            extract_participant_audio(cha_file, audio_file, out_file, sr_out=sr_out)
            written.append(str(out_file))
            print(f"  ok  {task}/{stem}")
        except Exception as exc:
            errors.append(f"{cha_file.name}: {exc}")
            print(f"  ERR {task}/{stem} -- {exc}")

    print(f"\n{'─'*60}")
    print(f"Written : {len(written)}")
    print(f"Skipped : {len(skipped)}  (no matching .mp3)")
    print(f"Errors  : {len(errors)}")
    return written


# -- Internal helpers ----------------------------------------------------------

_TIMESTAMP_RE = re.compile(r'\x15?(\d+)_(\d+)\x15?')


def _parse_par_timestamps(cha_path: Path):
    raw = cha_path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace('\x15', ' ')
    segments = []
    for line in raw.splitlines():
        if not line.startswith("*PAR:"):
            continue
        m = _TIMESTAMP_RE.search(line)
        if m:
            segments.append((int(m.group(1)), int(m.group(2))))
    return segments
