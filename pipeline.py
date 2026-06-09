#!/usr/bin/env python
"""
pipeline.py
-----------
Single entrypoint for the dementia detection pipeline.

Each command maps to one stage and delegates entirely to the
``dementia_detection`` library.  Run a command with ``--help`` for details.

Usage
-----
    python pipeline.py preprocess              # .cha → transcripts + WAV
    python pipeline.py features                # transcripts/WAV → feature CSVs
    python pipeline.py train                   # CSVs → trained models
    python pipeline.py search                  # hyperparameter search
    python pipeline.py export                  # best model → best_model.pkl
    python pipeline.py evaluate                # SHAP, calibration, permutation test
    python pipeline.py all                     # full pipeline end-to-end

    python pipeline.py train --task cookie --model svm --features liwc
    python pipeline.py features --lca-only --force
    python pipeline.py search --model svm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import (
    AUDIO_DIR, COMBINED_DIR, FIGURES_DIR, MODELS_DIR,
    PROCESSED_DIR, RAW_DIR, TASK, TASKS, TRANSCRIPT_DIR,
)

# Hardcoded choices keeps --help fast (avoids importing sklearn at parse time)
_MODELS   = ("svm", "logistic_regression", "random_forest", "xgboost")
_FEATURES = (
    "acoustic", "syntactic", "lexical", "liwc", "response_length", "all",
    "ablation_no_acoustic", "ablation_no_ac_rl", "ablation_no_ac_rl_syn",
)


# ---------------------------------------------------------------------------
# Command handlers  (all imports are lazy — only load what's needed)
# ---------------------------------------------------------------------------

def cmd_preprocess(args: argparse.Namespace) -> None:
    from dementia_detection.data import convert_all, extract_all_audio

    cha_dir   = RAW_DIR / "Pitt-transcript"
    audio_dir = RAW_DIR

    print("── Step 1: Convert .cha transcripts ─────────────────────────")
    if args.dry_run:
        n = len(list(cha_dir.glob("**/*.cha")))
        print(f"  [dry-run] {n} .cha files found under {cha_dir} — nothing written")
    else:
        convert_all(cha_dir, TRANSCRIPT_DIR)

    if not args.skip_audio:
        print("\n── Step 2: Extract participant-only WAV audio ────────────────")
        if args.dry_run:
            print("  [dry-run] Would extract audio — nothing written")
        else:
            extract_all_audio(cha_dir, audio_dir, AUDIO_DIR)


def cmd_features(args: argparse.Namespace) -> None:
    from dementia_detection.features import extract_all, extract_all_acoustic
    from dementia_detection.data import combine_all, integrate_all, add_all

    tasks = (args.task,) if args.task else TASKS

    print("── Step 3: Linguistic features (LCA + SCA) ──────────────────")
    extract_all(
        TRANSCRIPT_DIR, PROCESSED_DIR, tasks=tasks,
        run_lca=not args.sca_only, run_sca=not args.lca_only,
    )

    print("\n── Step 4: Acoustic features (eGeMAPS) ──────────────────────")
    extract_all_acoustic(AUDIO_DIR, PROCESSED_DIR, tasks=tasks, force=args.force)

    print("\n── Step 5: Combine acoustic + syntactic + lexical ───────────")
    combine_all(tasks=tasks, force=args.force)

    print("\n── Step 6: Integrate LIWC-22 ────────────────────────────────")
    integrate_all(tasks=tasks)

    print("\n── Step 7: Add response-length features ─────────────────────")
    add_all(tasks=tasks)


def cmd_train(args: argparse.Namespace) -> None:
    from dementia_detection.models import train_all

    tasks          = [args.task]     if args.task     else list(TASKS)
    model_names    = [args.model]    if args.model    else None
    feature_groups = [args.features] if args.features else None

    print("=" * 60)
    print("Model Training")
    print("=" * 60)
    train_all(
        combined_dir=COMBINED_DIR, output_dir=MODELS_DIR,
        tasks=tasks, model_names=model_names, feature_groups=feature_groups,
    )


def cmd_search(args: argparse.Namespace) -> None:
    from dementia_detection.models import tune_all

    tune_all(model_names=[args.model] if args.model else None)


def cmd_export(args: argparse.Namespace) -> None:
    from dementia_detection.models import export_best_model

    export_best_model()


def cmd_evaluate(args: argparse.Namespace) -> None:
    import pandas as pd
    from dementia_detection.models import plot_shap, plot_calibration, permutation_test

    pkl_path = MODELS_DIR / TASK / "svm__liwc.pkl"
    if not pkl_path.exists():
        print(f"[ERROR] Model not found: {pkl_path}")
        print("Run:  python pipeline.py train  (or  make train)")
        sys.exit(1)

    df = pd.read_csv(COMBINED_DIR / f"{TASK}_features.csv")
    df["subject_id"] = df["stem"].apply(lambda s: s.split("-")[0])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Evaluating SVM + LIWC")
    print("=" * 60)

    print("\n── SHAP ──────────────────────────────────────────────────────")
    plot_shap(pkl_path, df)

    print("\n── Calibration ───────────────────────────────────────────────")
    plot_calibration(pkl_path, df)

    print("\n── Permutation test ──────────────────────────────────────────")
    permutation_test(pkl_path, df)

    print(f"\nFigures saved → {FIGURES_DIR}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_preprocess(args)
    cmd_features(args)
    cmd_train(args)
    cmd_search(args)
    cmd_export(args)
    cmd_evaluate(args)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Dementia detection pipeline — single entrypoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "commands:\n"
            "  preprocess   Convert .cha files + extract participant-only WAV audio\n"
            "  features     Extract linguistic/acoustic features, integrate LIWC, combine\n"
            "  train        Train all classifiers on all feature groups\n"
            "  search       Hyperparameter grid search (LIWC, all 4 classifiers)\n"
            "  export       Train best model on full dataset → best_model.pkl\n"
            "  evaluate     SHAP summary, calibration curve, permutation test\n"
            "  all          Run full pipeline end-to-end\n"
        ),
    )
    sub = root.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # ── preprocess ────────────────────────────────────────────────────
    p = sub.add_parser("preprocess", help="Convert .cha + extract participant audio")
    p.add_argument("--task",       default=None, metavar="TASK",
                   help="Filter to a single task sub-folder (e.g. cookie)")
    p.add_argument("--dry-run",    action="store_true", dest="dry_run",
                   help="List files without writing anything")
    p.add_argument("--skip-audio", action="store_true", dest="skip_audio",
                   help="Skip participant audio extraction")
    p.set_defaults(func=cmd_preprocess)

    # ── features ──────────────────────────────────────────────────────
    p = sub.add_parser("features", help="Extract + combine all feature groups")
    p.add_argument("--task",     choices=list(TASKS), default=None)
    p.add_argument("--lca-only", action="store_true", dest="lca_only",
                   help="Run LCA only (no Java/Stanford required)")
    p.add_argument("--sca-only", action="store_true", dest="sca_only",
                   help="Run SCA only")
    p.add_argument("--force",    action="store_true",
                   help="Overwrite existing output CSVs")
    p.set_defaults(func=cmd_features)

    # ── train ─────────────────────────────────────────────────────────
    p = sub.add_parser("train", help="Train classifiers on feature CSVs")
    p.add_argument("--task",     choices=list(TASKS),     default=None)
    p.add_argument("--model",    choices=list(_MODELS),   default=None)
    p.add_argument("--features", choices=list(_FEATURES), default=None, metavar="GROUP")
    p.set_defaults(func=cmd_train)

    # ── search ────────────────────────────────────────────────────────
    p = sub.add_parser("search", help="Hyperparameter grid search (LIWC group)")
    p.add_argument("--model", choices=list(_MODELS), default=None,
                   help="Tune a single model (default: all four)")
    p.set_defaults(func=cmd_search)

    # ── export ────────────────────────────────────────────────────────
    p = sub.add_parser("export", help="Export best model as DementiaRiskPredictor")
    p.set_defaults(func=cmd_export)

    # ── evaluate ──────────────────────────────────────────────────────
    p = sub.add_parser("evaluate", help="SHAP, calibration curve, permutation test")
    p.set_defaults(func=cmd_evaluate)

    # ── all ───────────────────────────────────────────────────────────
    p = sub.add_parser("all", help="Run full pipeline end-to-end")
    p.add_argument("--task",       choices=list(TASKS), default=None)
    p.add_argument("--dry-run",    action="store_true", dest="dry_run")
    p.add_argument("--skip-audio", action="store_true", dest="skip_audio")
    p.add_argument("--lca-only",   action="store_true", dest="lca_only")
    p.add_argument("--sca-only",   action="store_true", dest="sca_only")
    p.add_argument("--force",      action="store_true")
    p.add_argument("--model",      choices=list(_MODELS),   default=None)
    p.add_argument("--features",   choices=list(_FEATURES), default=None, metavar="GROUP")
    p.set_defaults(func=cmd_all)

    return root


def main() -> None:
    parser = _make_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
