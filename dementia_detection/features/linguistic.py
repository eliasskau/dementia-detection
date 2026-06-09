"""
linguistic.py
-------------
Wrapper around NeoSCA's LCA (lexical) and SCA (syntactic) analyzers to
batch-extract features from cleaned participant transcripts.

LCA — 34 lexical diversity / sophistication features
    wordtypes, swordtypes, lextypes, slextypes,
    wordtokens, swordtokens, lextokens, slextokens,
    LD, LS1, LS2, VS1, VS2, CVS1,
    NDW, NDW-50, NDW-ER50, NDW-ES50,
    TTR, MSTTR, CTTR, RTTR, LogTTR, Uber,
    LV, VV1, SVV1, CVV1, VV2, NV, AdjV, AdvV, ModV

SCA — 23 syntactic complexity features
    W, S, VP, C, T, DC, CT, CP, CN,
    MLS, MLT, MLC, C/S, VP/T, C/T, DC/C, DC/T,
    T/S, CT/T, CP/T, CP/C, CN/T, CN/C

Source transcripts
------------------
Pitt/intermediate/cleaned_transcripts/
    Control/   {cookie, fluency, recall, sentence}/*.txt
    Dementia/  {cookie, fluency, recall, sentence}/*.txt

Outputs
-------
Pitt/processed/
    lexical/               ← LCA outputs
        cookie_features.csv
        fluency_features.csv
        recall_features.csv
        sentence_features.csv
    syntactic/             ← SCA outputs (same structure)

Public API
----------
extract_lca_features(txt_path) -> dict
extract_sca_features(txt_path, stanford_parser_home, stanford_tregex_home) -> dict
extract_all(transcripts_dir, output_dir, tasks=None, run_lca=True, run_sca=True)

Usage (CLI)
-----------
    # Both LCA + SCA, cookie only:
    STANFORD_PARSER_HOME=~/neosca-stanford/stanford-parser-full-2020-11-17 \\
    STANFORD_TREGEX_HOME=~/neosca-stanford/stanford-tregex-2020-11-17 \\
    /path/to/conda/envs/dementia-detection/bin/python \\
        -m src.feature_extraction.linguistic --task cookie

    # LCA only (no Java needed):
    ... python -m src.feature_extraction.linguistic --lca-only
"""

import csv
import os
import tempfile
from pathlib import Path
from typing import Optional

_ALL_TASKS = ("cookie", "fluency", "recall", "sentence")

# ── Java / Stanford paths ─────────────────────────────────────────────────────

_OPENJDK17_JVM = (
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk"
    "/Contents/Home/lib/server/libjvm.dylib"
)

def _stanford_homes() -> tuple[str, str]:
    """Return (parser_home, tregex_home) from env vars."""
    parser = os.environ.get(
        "STANFORD_PARSER_HOME",
        str(Path.home() / "neosca-stanford" / "stanford-parser-full-2020-11-17"),
    )
    tregex = os.environ.get(
        "STANFORD_TREGEX_HOME",
        str(Path.home() / "neosca-stanford" / "stanford-tregex-2020-11-17"),
    )
    return parser, tregex


def _ensure_jvm(stanford_parser_home: str, stanford_tregex_home: str) -> None:
    """Start JPype JVM with Stanford jars on the classpath, if not already running."""
    import jpype
    if jpype.isJVMStarted():
        return

    from neosca import scaenv
    classpaths = scaenv.unite_classpaths(stanford_parser_home, stanford_tregex_home)

    jvm = os.environ.get("JPYPE_JVM", _OPENJDK17_JVM)
    if not Path(jvm).exists():
        raise RuntimeError(
            f"JVM not found at {jvm}.\n"
            "Install OpenJDK 17 with: brew install openjdk@17\n"
            "Or set JPYPE_JVM to your libjvm.dylib path."
        )
    jpype.startJVM(jvm, classpath=classpaths)


# ── LCA (lexical) ─────────────────────────────────────────────────────────────

def extract_lca_features(txt_path) -> dict:
    """
    Run NeoSCA LCA on a single cleaned transcript .txt file.
    Returns a dict of 34 lexical features, or {} on failure.
    """
    from neosca.lca.lca import LCA

    txt_path = Path(txt_path)
    if txt_path.stat().st_size == 0:
        return {}

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as f:
        tmp = f.name

    try:
        lca = LCA(ofile=tmp)
        ok, _ = lca.analyze(ifiles=[str(txt_path)])
        if not ok:
            return {}

        with open(tmp, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}

        row = dict(rows[0])
        row.pop("filename", None)
        return {k: _to_num(v) for k, v in row.items()}

    finally:
        Path(tmp).unlink(missing_ok=True)


# ── SCA (syntactic) ───────────────────────────────────────────────────────────

def extract_sca_features(
    txt_path,
    stanford_parser_home: Optional[str] = None,
    stanford_tregex_home: Optional[str] = None,
) -> dict:
    """
    Run NeoSCA SCA on a single cleaned transcript .txt file.
    Returns a dict of 23 syntactic features, or {} on failure.

    Requires Java (OpenJDK 17 arm64) and Stanford Parser/Tregex jars.
    """
    from neosca.neosca import NeoSCA

    txt_path = Path(txt_path)
    if txt_path.stat().st_size == 0:
        return {}

    if stanford_parser_home is None or stanford_tregex_home is None:
        stanford_parser_home, stanford_tregex_home = _stanford_homes()

    _ensure_jvm(stanford_parser_home, stanford_tregex_home)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as f:
        tmp = f.name

    try:
        sca = NeoSCA(
            ofile_freq=tmp,
            stanford_parser_home=stanford_parser_home,
            stanford_tregex_home=stanford_tregex_home,
        )
        sca.run_on_ifiles([str(txt_path)])

        with open(tmp, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}

        row = dict(rows[0])
        row.pop("Filename", None)
        return {k: _to_num(v) for k, v in row.items()}

    finally:
        Path(tmp).unlink(missing_ok=True)


# ── Batch extraction ──────────────────────────────────────────────────────────

def extract_all(
    transcripts_dir,
    output_dir,
    tasks: Optional[tuple] = None,
    run_lca: bool = True,
    run_sca: bool = True,
) -> dict:
    """
    Batch-extract LCA and/or SCA features for all tasks.

    Outputs:
        {output_dir}/lexical/{task}_features.csv    (if run_lca)
        {output_dir}/syntactic/{task}_features.csv  (if run_sca)

    Returns dict keyed by e.g. ("lca", "cookie") or ("sca", "cookie").
    """
    transcripts_dir = Path(transcripts_dir)
    output_dir      = Path(output_dir)

    if tasks is None:
        tasks = _ALL_TASKS

    parser_home, tregex_home = _stanford_homes()

    results = {}

    for task in tasks:
        print(f"\n{'═'*60}")
        print(f"Task: {task}")
        print(f"{'═'*60}")

        lca_rows, sca_rows = [], []
        lca_ok = lca_err = sca_ok = sca_err = 0

        for label in ("Control", "Dementia"):
            task_dir = transcripts_dir / label / task
            if not task_dir.exists():
                print(f"  [SKIP] {label}/{task} — directory not found")
                continue

            txt_files = sorted(task_dir.glob("*.txt"))
            print(f"\n  {label}: {len(txt_files)} files")

            for txt in txt_files:
                meta = {"filename": txt.name, "stem": txt.stem, "label": label}

                if run_lca:
                    feats = extract_lca_features(txt)
                    if feats:
                        lca_rows.append({**meta, **feats})
                        lca_ok += 1
                    else:
                        print(f"    ✗ LCA  {txt.name}")
                        lca_err += 1

                if run_sca:
                    feats = extract_sca_features(txt, parser_home, tregex_home)
                    if feats:
                        sca_rows.append({**meta, **feats})
                        sca_ok += 1
                    else:
                        print(f"    ✗ SCA  {txt.name}")
                        sca_err += 1

                if run_lca and run_sca and lca_rows and sca_rows:
                    # only print once per file if both ran
                    print(f"    ✓  {txt.name}")
                elif run_lca and lca_rows:
                    print(f"    ✓  {txt.name}")
                elif run_sca and sca_rows:
                    print(f"    ✓  {txt.name}")

        if run_lca and lca_rows:
            out = _write_csv(lca_rows, output_dir / "lexical" / f"{task}_features.csv")
            results[("lca", task)] = out
            print(f"\n  LCA written: {lca_ok} | errors: {lca_err} → {out}")

        if run_sca and sca_rows:
            out = _write_csv(sca_rows, output_dir / "syntactic" / f"{task}_features.csv")
            results[("sca", task)] = out
            print(f"  SCA written: {sca_ok} | errors: {sca_err} → {out}")

    return results


# ── Utilities ─────────────────────────────────────────────────────────────────

def _to_num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _write_csv(rows: list, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
    TRANSCRIPTS_DIR = PROJECT_ROOT / "Pitt" / "intermediate" / "cleaned_transcripts"
    OUTPUT_DIR      = PROJECT_ROOT / "Pitt" / "processed"

    parser = argparse.ArgumentParser(
        description="Extract LCA (lexical) and SCA (syntactic) features from Pitt transcripts."
    )
    parser.add_argument("--task", type=str, default=None,
        help="Only process this task (cookie|fluency|recall|sentence). Default: all.")
    parser.add_argument("--lca-only", action="store_true",
        help="Only run LCA (no Java/Stanford needed).")
    parser.add_argument("--sca-only", action="store_true",
        help="Only run SCA.")
    parser.add_argument("--transcripts-dir", type=Path, default=TRANSCRIPTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    run_lca = not args.sca_only
    run_sca = not args.lca_only
    tasks   = (args.task,) if args.task else None

    written = extract_all(
        transcripts_dir=args.transcripts_dir,
        output_dir=args.output_dir,
        tasks=tasks,
        run_lca=run_lca,
        run_sca=run_sca,
    )

    print(f"\n{'═'*60}")
    print(f"Done. {len(written)} CSV(s) written:")
    for (kind, task), path in written.items():
        print(f"  [{kind.upper()}] {task:10s} → {path}")
