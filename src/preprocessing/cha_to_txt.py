"""
cha_to_txt.py
-------------
Convert CHAT (.cha) transcript files to plain-text files containing
only the cleaned participant (*PAR) speech.

Public API
----------
cha_to_txt(cha_path, out_path=None) -> str
    Convert a single .cha file.

convert_all(input_dir, output_dir=None, recursive=True) -> list[str]
    Batch-convert every .cha file found under input_dir.
"""

import re
from pathlib import Path

from .text_cleaner import clean_participant_text


def cha_to_txt(
    cha_path: "str | Path",
    out_path: "str | Path | None" = None,
) -> str:
    """
    Convert a single .cha file to a plain .txt file.

    Each *PAR utterance becomes one line in the output.
    Continuation lines and %mor/%gra annotation tiers are handled correctly.

    Parameters
    ----------
    cha_path : path to the source .cha file
    out_path : path to write the .txt file.
               Defaults to the same location as cha_path with a .txt extension.

    Returns
    -------
    Absolute path of the written .txt file (str).
    """
    cha_path = Path(cha_path)

    if out_path is None:
        out_path = cha_path.with_suffix(".txt")
    out_path = Path(out_path)

    raw_utterances = _extract_par_lines(cha_path)
    cleaned = [clean_participant_text(u) for u in raw_utterances]
    # Drop utterances that are empty after cleaning
    cleaned = [line for line in cleaned if line]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(cleaned), encoding="utf-8")

    return str(out_path)


def convert_all(
    input_dir:  "str | Path",
    output_dir: "str | Path | None" = None,
    recursive:  bool = True,
) -> "list[str]":
    """
    Batch-convert every .cha file found under input_dir.

    Parameters
    ----------
    input_dir  : root directory to search for .cha files
    output_dir : root directory for .txt output.
                 The sub-directory structure under input_dir is mirrored.
                 If None, each .txt file is written next to its .cha file.
    recursive  : search sub-directories recursively (default True)

    Returns
    -------
    List of output file paths that were written.
    """
    input_dir = Path(input_dir)
    pattern   = "**/*.cha" if recursive else "*.cha"
    cha_files = sorted(input_dir.glob(pattern))

    if not cha_files:
        print(f"No .cha files found in {input_dir}")
        return []

    written = []
    for cha_file in cha_files:
        if output_dir is not None:
            rel      = cha_file.relative_to(input_dir)
            out_file = Path(output_dir) / rel.with_suffix(".txt")
        else:
            out_file = None

        out = cha_to_txt(cha_file, out_file)
        written.append(out)
        print(f"  ✓  {cha_file.name}  →  {out}")

    print(f"\nDone — {len(written)} files converted.")
    return written


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_par_lines(cha_path: Path) -> "list[str]":
    """
    Read a .cha file and return a list of raw *PAR utterance strings.

    Handles:
    - Multi-line utterances (continuation lines start with a tab)
    - %mor / %gra / other annotation tiers (skipped entirely)
    """
    raw   = cha_path.read_text(encoding="utf-8", errors="replace")
    # 0x15 (NAK) is used as an in-line field separator in some CHAT files;
    # replace it with a space so it doesn't bleed into the next tier.
    raw   = raw.replace('\x15', ' ')
    lines = raw.splitlines()
    par_lines: "list[str]" = []
    current:   "str | None" = None

    for line in lines:
        if line.startswith("*PAR:"):
            # Save any previous utterance
            if current is not None:
                par_lines.append(current)
            current = re.sub(r'^\*PAR:\s*', '', line)

        elif line.startswith("\t") and current is not None:
            # Continuation line — skip if it is an annotation tier (%mor, %gra …)
            stripped = line.strip()
            if not stripped.startswith("%"):
                current += " " + stripped

        else:
            # Any new speaker tier, header line, or blank line ends the utterance
            if current is not None:
                par_lines.append(current)
                current = None

    # Flush the last utterance
    if current is not None:
        par_lines.append(current)

    return par_lines
