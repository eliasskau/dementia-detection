#!/usr/bin/env python
"""
06_train_models.py
------------------
Train Logistic Regression, Random Forest, and XGBoost classifiers on
combined acoustic + syntactic + lexical features from the Pitt corpus.

Reads from : Pitt/processed/combined/{task}_features.csv
Writes to  : results/models/{task}/{model}__{feature_group}.pkl
             results/models/{task}/{model}__{feature_group}__cv_metrics.json
             results/models/training_summary.csv

Evaluation : Leave-One-Subject-Out (LOSO) cross-validation
             Metrics: AUC, F1, Accuracy (mean ± std across subjects)

Run
---
    conda run -n dementia-detection python scripts/06_train_models.py
    conda run -n dementia-detection python scripts/06_train_models.py --task cookie
    conda run -n dementia-detection python scripts/06_train_models.py --model xgboost
    conda run -n dementia-detection python scripts/06_train_models.py --model xgboost --features all
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

COMBINED_DIR = PROJECT_ROOT / "Pitt" / "processed" / "combined"
MODELS_DIR   = PROJECT_ROOT / "results" / "models"

from src.models.train import MODELS, FEATURE_GROUPS, train_all  # noqa: E402

_ALL_TASKS = ("cookie", "fluency", "recall", "sentence")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train dementia detection models.")
    p.add_argument(
        "--task",
        choices=list(_ALL_TASKS),
        default=None,
        help="Train on a single task (default: all four).",
    )
    p.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Train a single model type (default: all three).",
    )
    p.add_argument(
        "--features",
        choices=list(FEATURE_GROUPS.keys()),
        default=None,
        help="Use a single feature group (default: all groups).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    tasks          = [args.task]     if args.task     else list(_ALL_TASKS)
    model_names    = [args.model]    if args.model    else list(MODELS.keys())
    feature_groups = [args.features] if args.features else list(FEATURE_GROUPS.keys())

    print("=" * 60)
    print("Dementia Detection — Model Training")
    print("=" * 60)
    print(f"Tasks          : {tasks}")
    print(f"Models         : {model_names}")
    print(f"Feature groups : {feature_groups}")
    print(f"Combined dir   : {COMBINED_DIR}")
    print(f"Output dir     : {MODELS_DIR}")

    train_all(
        combined_dir=COMBINED_DIR,
        output_dir=MODELS_DIR,
        tasks=tasks,
        model_names=model_names,
        feature_groups=feature_groups,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
