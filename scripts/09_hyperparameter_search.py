#!/usr/bin/env python
"""
10_hyperparameter_search.py
---------------------------
Hyperparameter search for the best model+feature-group combo per classifier.

Design
------
    All four classifiers (SVM, LR, RF, XGBoost) are tuned on their best
    feature group (liwc, Gini-selected).  The same subject-aware 80/20 split
    used in 06_train_models.py is applied here, so the held-out 20% test set
    is never visible during tuning — only during final evaluation.

    Inside the 80% train set, a subject-aware 5-fold CV is used as the inner
    loop for GridSearchCV (scoring=roc_auc).

Protocol (per model)
--------------------
    1.  80/20 subject-aware split  (same random_state=42 → identical split as training)
    2.  GridSearchCV on train split
            inner CV : subject-aware 5-fold (custom CV splitter)
            scoring  : roc_auc
            refit    : True  → best estimator refitted on full 80% train set
    3.  Evaluate best estimator on held-out 20% → tuned_test_auc
    4.  Compare vs default_test_auc from training_summary.csv

Outputs
-------
    results/models/hyperparameter_search_results.csv
        Columns: model, feature_group, best_params, default_test_auc,
                 tuned_test_auc, delta_auc, cv_best_score, cv_best_std

    results/models/tuned_{model}__liwc.pkl
        Same DementiaRiskPredictor format as best_model.pkl.

Run
---
    conda run -n dementia-detection python scripts/10_hyperparameter_search.py
    conda run -n dementia-detection python scripts/10_hyperparameter_search.py --model svm
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dementia_detection.models.train import GiniSelector  # noqa: E402
from config.config import GINI_THRESHOLD as _GINI_THRESHOLD, RANDOM_STATE  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (search-specific — not part of the training library)
# ---------------------------------------------------------------------------

def _feature_cols(df: "pd.DataFrame", feature_group: str) -> "list[str]":
    """Return all columns whose prefix matches *feature_group*."""
    prefix = f"{feature_group}__"
    return [c for c in df.columns if c.startswith(prefix)]


def _subject_train_test_split(
    df: "pd.DataFrame",
    test_size: float = 0.2,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    80/20 subject-aware split preserving class balance.
    Deterministic given RANDOM_STATE.
    """
    subj_df = (
        df.groupby("subject_id")["label"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"label": "subject_label"})
    )
    skf = StratifiedKFold(
        n_splits=int(round(1 / test_size)),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    _, test_subj_idx = next(
        skf.split(subj_df["subject_id"], subj_df["subject_label"])
    )
    test_subjects = set(subj_df.loc[test_subj_idx, "subject_id"])
    mask = df["subject_id"].isin(test_subjects)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK           = "cookie"
FEATURE_GROUP  = "liwc"
FEATURE_PREFIX = "liwc__"
COMBINED_CSV   = PROJECT_ROOT / "Pitt" / "processed" / "combined" / f"{TASK}_features.csv"
SUMMARY_CSV    = PROJECT_ROOT / "results" / "models" / "training_summary.csv"
OUT_DIR        = PROJECT_ROOT / "results" / "models"
N_CV_FOLDS     = 5

# ---------------------------------------------------------------------------
# Default test AUCs (from 06_train_models.py, held-out 20% test set)
# These are used only for the delta comparison in the summary table.
# ---------------------------------------------------------------------------
DEFAULT_TEST_AUC = {
    "svm":                 0.789,
    "logistic_regression": 0.782,
    "random_forest":       0.768,
    "xgboost":             0.753,
}

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
# SVM (RBF kernel):
#   C     controls the penalty for misclassification.  Larger C = tighter
#         margin = more complex boundary.  Default=1.0.
#   gamma controls the reach of each training sample.  "scale" = 1/(n_features
#         * X.var()).  Smaller gamma = smoother boundary.
#
# LR:
#   C     inverse regularisation strength.  Smaller C = stronger L2 shrinkage.
#   solver "lbfgs" works for L2.  "saga" also supports L1 if added later.
#
# RF:
#   n_estimators  more trees → lower variance but slower.
#   max_depth     None = grow until pure leaves (can overfit).
#   min_samples_leaf  prevents very small leaves, acts as regulariser.
#   max_features  fraction of features per split.  "sqrt" is sklearn default.
#
# XGBoost:
#   n_estimators  number of boosting rounds.
#   max_depth     tree depth.  Shallow trees (3-5) generalise better.
#   learning_rate step size shrinkage.  Smaller = slower but better.
#   subsample     row subsampling per tree — reduces overfitting.
#   colsample_bytree  column subsampling — like RF's max_features.
#   reg_lambda    L2 regularisation on leaf weights (default=1).
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "svm": {
        "clf__C":     [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
    "logistic_regression": {
        "clf__C":       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__solver":  ["lbfgs", "saga"],
        "clf__penalty": ["l2"],
    },
    "random_forest": {
        "clf__n_estimators":    [100, 200, 400],
        "clf__max_depth":       [None, 5, 10, 20],
        "clf__min_samples_leaf":[1, 3, 5],
        "clf__max_features":    ["sqrt", 0.3, 0.5],
    },
    "xgboost": {
        "clf__n_estimators":      [100, 200, 400],
        "clf__max_depth":         [3, 4, 6],
        "clf__learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "clf__subsample":         [0.6, 0.8, 1.0],
        "clf__colsample_bytree":  [0.6, 0.8, 1.0],
        "clf__reg_lambda":        [0.1, 1.0, 10.0],
    },
}

# Grid sizes for printing
_GRID_SIZES = {k: int(np.prod([len(v) for v in g.values()]))
               for k, g in PARAM_GRIDS.items()}

# ---------------------------------------------------------------------------
# Base estimators (same defaults as train.py)
# ---------------------------------------------------------------------------
BASE_ESTIMATORS = {
    "svm": SVC(
        kernel="rbf", class_weight="balanced",
        probability=True, random_state=42,
    ),
    "logistic_regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "xgboost": XGBClassifier(
        eval_metric="logloss", random_state=42, n_jobs=-1,
    ),
}

NEEDS_SCALING = {"svm": True, "logistic_regression": True,
                 "random_forest": False, "xgboost": False}


# ---------------------------------------------------------------------------
# Subject-aware CV splitter for GridSearchCV
# ---------------------------------------------------------------------------

def make_subject_cv_splits(df_train: pd.DataFrame, n_splits: int = 5):
    """
    Returns a list of (train_idx, val_idx) tuples where splits are at
    subject level.  Passed as `cv=` to GridSearchCV.

    GridSearchCV expects indices into the X array, not boolean masks.
    """
    subject_ids = df_train["subject_id"].values
    unique_subj = np.unique(subject_ids)
    le_subj     = LabelEncoder()
    # majority-vote label per subject
    y_enc = LabelEncoder().fit_transform(df_train["label"].values)
    subj_label = np.array([
        int(y_enc[subject_ids == s].mean() >= 0.5)
        for s in unique_subj
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    for tr_subj_idx, val_subj_idx in skf.split(unique_subj, subj_label):
        val_subjects = set(unique_subj[val_subj_idx])
        val_mask  = np.array([s in val_subjects for s in subject_ids])
        train_idx = np.where(~val_mask)[0]
        val_idx   = np.where(val_mask)[0]
        splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
# Build pipeline (same structure as train.py)
# ---------------------------------------------------------------------------

def build_pipeline(model_name: str, estimator) -> Pipeline:
    gini_thresh = _GINI_THRESHOLD["liwc"]
    steps = [
        ("imputer",       SimpleImputer(strategy="constant", fill_value=0.0)),
        ("gini_selector", GiniSelector(threshold=gini_thresh)),
    ]
    if NEEDS_SCALING[model_name]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Single model tuning
# ---------------------------------------------------------------------------

def tune_model(
    model_name: str,
    df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """Run grid search for one model on the LIWC feature group."""

    print(f"\n{'─'*62}")
    print(f"  Model : {model_name}")
    print(f"  Grid  : {_GRID_SIZES[model_name]} combinations × {N_CV_FOLDS} folds"
          f" = {_GRID_SIZES[model_name] * N_CV_FOLDS} fits")

    # ── 80/20 split (identical to 06_train_models.py) ─────────────────
    df_train, df_test = _subject_train_test_split(df)

    cols  = _feature_cols(df_train, FEATURE_GROUP)
    le    = LabelEncoder().fit(df["label"].values)

    X_train = df_train[cols].values.astype(np.float32)
    y_train = le.transform(df_train["label"].values)
    X_test  = df_test[cols].values.astype(np.float32)
    y_test  = le.transform(df_test["label"].values)

    # XGBoost: set scale_pos_weight from training distribution
    estimator = clone(BASE_ESTIMATORS[model_name])
    if model_name == "xgboost":
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        estimator.set_params(scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0)

    pipe   = build_pipeline(model_name, estimator)
    cv_idx = make_subject_cv_splits(df_train, n_splits=N_CV_FOLDS)

    # ── GridSearchCV ──────────────────────────────────────────────────
    t0 = time.time()
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRIDS[model_name],
        cv=cv_idx,
        scoring="roc_auc",
        refit=True,          # refit best params on full train set
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
        error_score=np.nan,
    )
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0

    # ── Evaluate best estimator on held-out test ───────────────────────
    y_prob_tuned = gs.best_estimator_.predict_proba(X_test)[:, 1]
    tuned_test_auc = float(roc_auc_score(y_test, y_prob_tuned))

    default_test_auc = DEFAULT_TEST_AUC.get(model_name, None)
    delta = (tuned_test_auc - default_test_auc) if default_test_auc else None

    # ── CV score distribution for best params ─────────────────────────
    best_idx = gs.best_index_
    cv_mean  = float(gs.cv_results_["mean_test_score"][best_idx])
    cv_std   = float(gs.cv_results_["std_test_score"][best_idx])

    # ── Print results ─────────────────────────────────────────────────
    print(f"  Best params : {gs.best_params_}")
    print(f"  CV-AUC      : {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"  Default AUC : {default_test_auc:.3f}" if default_test_auc else "")
    print(f"  Tuned  AUC  : {tuned_test_auc:.3f}"
          + (f"  (Δ {delta:+.3f})" if delta is not None else ""))
    print(f"  Time        : {elapsed:.1f}s")

    # ── Save tuned pipeline as pkl ────────────────────────────────────
    out_path = OUT_DIR / f"tuned_{model_name}__liwc.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "pipeline":      gs.best_estimator_,
                "label_encoder": le,
                "feature_cols":  cols,
                "best_params":   gs.best_params_,
                "tuned_test_auc": tuned_test_auc,
                "cv_auc_mean":   cv_mean,
                "cv_auc_std":    cv_std,
            },
            f,
        )
    print(f"  Saved  : {out_path.relative_to(PROJECT_ROOT)}")

    return {
        "model":             model_name,
        "feature_group":     FEATURE_GROUP,
        "best_params":       json.dumps(gs.best_params_),
        "cv_best_score":     round(cv_mean, 4),
        "cv_best_std":       round(cv_std, 4),
        "default_test_auc":  default_test_auc,
        "tuned_test_auc":    round(tuned_test_auc, 4),
        "delta_auc":         round(delta, 4) if delta is not None else None,
        "n_fits":            _GRID_SIZES[model_name] * N_CV_FOLDS,
        "elapsed_sec":       round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(PARAM_GRIDS.keys()),
        default=None,
        help="Tune a single model (default: all four).",
    )
    args = parser.parse_args()

    models_to_tune = [args.model] if args.model else list(PARAM_GRIDS.keys())

    print("=" * 62)
    print("Hyperparameter Search — LIWC feature group")
    print("=" * 62)
    print(f"Models         : {models_to_tune}")
    print(f"CV folds       : {N_CV_FOLDS} (subject-aware)")
    print(f"Total grid fits: "
          f"{sum(_GRID_SIZES[m] * N_CV_FOLDS for m in models_to_tune)}")

    df = pd.read_csv(COMBINED_CSV)
    df["subject_id"] = df["stem"].apply(lambda s: s.split("-")[0])

    rows = []
    for model_name in models_to_tune:
        result = tune_model(model_name, df)
        rows.append(result)

    # ── Summary table ─────────────────────────────────────────────────
    summary = pd.DataFrame(rows)
    out_csv = OUT_DIR / "hyperparameter_search_results.csv"
    summary.to_csv(out_csv, index=False)

    print(f"\n{'='*62}")
    print("Results summary")
    print("=" * 62)
    display_cols = ["model", "cv_best_score", "cv_best_std",
                    "default_test_auc", "tuned_test_auc", "delta_auc"]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFull results saved → {out_csv.relative_to(PROJECT_ROOT)}")

    # ── Flag best overall tuned model ─────────────────────────────────
    if len(summary) > 1:
        best_row = summary.loc[summary["tuned_test_auc"].idxmax()]
        print(f"\n🏆  Best tuned model : {best_row['model']}"
              f"  Test-AUC = {best_row['tuned_test_auc']:.4f}"
              f"  (Δ {best_row['delta_auc']:+.4f} vs default)")


if __name__ == "__main__":
    main()
