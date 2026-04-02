"""
03_extract_linguistic_features.py
-----------------------------------
Batch-extract LCA (lexical) and SCA (syntactic) features from all
cleaned transcripts and write to:

    Pitt/processed/lexical/{task}_features.csv
    Pitt/processed/syntactic/{task}_features.csv

Usage
-----
    STANFORD_PARSER_HOME=~/neosca-stanford/stanford-parser-full-2020-11-17 \
    STANFORD_TREGEX_HOME=~/neosca-stanford/stanford-tregex-2020-11-17 \
    /path/to/conda/envs/dementia-detection/bin/python \
        scripts/03_extract_linguistic_features.py

    # LCA only (no Java needed):
    python scripts/03_extract_linguistic_features.py --lca-only

    # Single task:
    python scripts/03_extract_linguistic_features.py --task cookie
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.linguistic import extract_all  # noqa: E402

TRANSCRIPTS_DIR = PROJECT_ROOT / "Pitt" / "intermediate" / "cleaned_transcripts"
OUTPUT_DIR      = PROJECT_ROOT / "Pitt" / "processed"


def main():
    parser = argparse.ArgumentParser(
        description="Extract LCA + SCA linguistic features from Pitt transcripts."
    )
    parser.add_argument("--task", type=str, default=None,
        help="Only process this task (cookie|fluency|recall|sentence). Default: all.")
    parser.add_argument("--lca-only", action="store_true",
        help="Only run LCA (no Java/Stanford needed).")
    parser.add_argument("--sca-only", action="store_true",
        help="Only run SCA.")
    args = parser.parse_args()

    run_lca = not args.sca_only
    run_sca = not args.lca_only
    tasks   = (args.task,) if args.task else None

    written = extract_all(
        transcripts_dir=TRANSCRIPTS_DIR,
        output_dir=OUTPUT_DIR,
        tasks=tasks,
        run_lca=run_lca,
        run_sca=run_sca,
    )

    print(f"\n{'═'*60}")
    print(f"Done. {len(written)} CSV(s) written:")
    for (kind, task), path in written.items():
        print(f"  [{kind.upper()}] {task:10s} → {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
