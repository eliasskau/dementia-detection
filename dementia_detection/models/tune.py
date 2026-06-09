"""
tune.py
-------
Hyperparameter search for all classifiers on the LIWC feature group.

Uses a subject-aware 80/20 split for the held-out test set and a
subject-aware 5-fold inner CV for GridSearchCV, matching the evaluation
protocol in train.py.

Public API
----------
tune_model(model_name, df, ...)  -> dict
    Grid-search one model and return a results dictionary.

tune_all(model_names, task, out_dir) -> pd.DataFrame
    Tune every requested model and write a summary CSV.
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config.config import (
    COMBINED_DIR, GINI_THRESHOLD as _GINI_THRESHOLD,
    MODELS_DIR, RANDOM_STATE, TASK,
)
from .train import GiniSelector

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

PARAM_GRIDS: dict[str, dict] = {
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
        "clf__n_estimators":     [100, 200, 400],
        "clf__max_depth":        [None, 5, 10, 20],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features":     ["sqrt", 0.3, 0.5],
    },
    "xgboost": {
        "clf__n_estimators":     [100, 200, 400],
        "clf__max_depth":        [3, 4, 6],
        "clf__learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "clf__subsample":        [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda":       [0.1, 1.0, 10.0],
    },
}

_BASE_ESTIMATORS = {
    "svm": SVC(
        kernel="rbf", class_weight="balanced",
        probability=True, random_state=RANDOM_STATE,
    ),
    "logistic_regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
    ),
    "random_forest": RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "xgboost": XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
    ),
}

_NEEDS_SCALING = {
    "svm": True, "logistic_regression": True,
    "random_forest": False, "xgboost": False,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _feature_cols(df: pd.DataFrame, feature_group: str) -> list[str]:
    return [c for c in df.columns if c.startswith(f"{feature_group}__")]


def _subject_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic subject-aware 80/20 split (stratified by diagnosis)."""
    subj_df = (
        df.groupby("subject_id")["label"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"label": "subject_label"})
    )
    skf = StratifiedKFold(
        n_splits=int(round(1 / test_size)),
        shuffle=True, random_state=RANDOM_STATE,
    )
    _, test_idx = next(skf.split(subj_df["subject_id"], subj_df["subject_label"]))
    test_subjects = set(subj_df.loc[test_idx, "subject_id"])
    mask = df["subject_id"].isin(test_subjects)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def _subject_cv_splits(df_train: pd.DataFrame, n_splits: int = 5) -> list[tuple]:
    """Subject-aware CV index pairs for use as GridSearchCV ``cv=``."""
    subject_ids = df_train["subject_id"].values
    unique_subj = np.unique(subject_ids)
    y_enc       = LabelEncoder().fit_transform(df_train["label"].values)
    subj_label  = np.array([
        int(y_enc[subject_ids == s].mean() >= 0.5) for s in unique_subj
    ])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    splits = []
    for tr_idx, val_idx in skf.split(unique_subj, subj_label):
        val_subjects = set(unique_subj[val_idx])
        val_mask = np.array([s in val_subjects for s in subject_ids])
        splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return splits


def _build_pipeline(model_name: str, estimator) -> Pipeline:
    gini_thresh = _GINI_THRESHOLD.get("liwc", 0.008)
    steps: list = [
        ("imputer",       SimpleImputer(strategy="constant", fill_value=0.0)),
        ("gini_selector", GiniSelector(threshold=gini_thresh)),
    ]
    if _NEEDS_SCALING[model_name]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune_model(
    model_name: str,
    df: pd.DataFrame,
    feature_group: str = "liwc",
    n_cv_folds: int = 5,
    out_dir: Path = MODELS_DIR,
    verbose: bool = True,
) -> dict:
    """
    Grid-search *model_name* on *feature_group* columns.

    Returns a result dictionary with ``best_params``, ``cv_best_score``,
    ``tuned_test_auc``, ``elapsed_sec``, etc.  Also saves a tuned .pkl.
    """
    grid_size = int(np.prod([len(v) for v in PARAM_GRIDS[model_name].values()]))
    if verbose:
        print(f"\n{'─' * 62}")
        print(f"  Model : {model_name}")
        print(f"  Grid  : {grid_size} combos × {n_cv_folds} folds = {grid_size * n_cv_folds} fits")

    df_train, df_test = _subject_train_test_split(df)
    cols  = _feature_cols(df_train, feature_group)
    le    = LabelEncoder().fit(df["label"].values)

    X_train = df_train[cols].values.astype(np.float32)
    y_train = le.transform(df_train["label"].values)
    X_test  = df_test[cols].values.astype(np.float32)
    y_test  = le.transform(df_test["label"].values)

    estimator = clone(_BASE_ESTIMATORS[model_name])
    if model_name == "xgboost":
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        estimator.set_params(scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0)

    pipe   = _build_pipeline(model_name, estimator)
    cv_idx = _subject_cv_splits(df_train, n_splits=n_cv_folds)

    t0 = time.time()
    gs = GridSearchCV(
        estimator=pipe, param_grid=PARAM_GRIDS[model_name],
        cv=cv_idx, scoring="roc_auc", refit=True,
        n_jobs=-1, verbose=0, error_score=np.nan,
    )
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0

    tuned_test_auc = float(roc_auc_score(y_test, gs.best_estimator_.predict_proba(X_test)[:, 1]))
    cv_mean = float(gs.cv_results_["mean_test_score"][gs.best_index_])
    cv_std  = float(gs.cv_results_["std_test_score"][gs.best_index_])

    if verbose:
        print(f"  Best params : {gs.best_params_}")
        print(f"  CV-AUC      : {cv_mean:.3f} ± {cv_std:.3f}")
        print(f"  Tuned AUC   : {tuned_test_auc:.3f}")
        print(f"  Time        : {elapsed:.1f}s")

    # Save tuned pipeline
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"tuned_{model_name}__{feature_group}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "pipeline":       gs.best_estimator_,
            "label_encoder":  le,
            "feature_cols":   cols,
            "best_params":    gs.best_params_,
            "tuned_test_auc": tuned_test_auc,
            "cv_auc_mean":    cv_mean,
            "cv_auc_std":     cv_std,
        }, f)
    if verbose:
        print(f"  Saved  : {pkl_path}")

    return {
        "model":          model_name,
        "feature_group":  feature_group,
        "best_params":    json.dumps(gs.best_params_),
        "cv_best_score":  round(cv_mean, 4),
        "cv_best_std":    round(cv_std, 4),
        "tuned_test_auc": round(tuned_test_auc, 4),
        "n_fits":         grid_size * n_cv_folds,
        "elapsed_sec":    round(elapsed, 1),
    }


def tune_all(
    model_names: "list[str] | None" = None,
    task: str = TASK,
    feature_group: str = "liwc",
    out_dir: Path = MODELS_DIR,
) -> pd.DataFrame:
    """
    Tune every model in *model_names* (default: all four classifiers) and
    write ``out_dir/hyperparameter_search_results.csv``.

    Returns
    -------
    pd.DataFrame summary with one row per model.
    """
    if model_names is None:
        model_names = list(PARAM_GRIDS.keys())

    csv_path = COMBINED_DIR / f"{task}_features.csv"
    df = pd.read_csv(csv_path)
    df["subject_id"] = df["stem"].apply(lambda s: s.split("-")[0])

    total_fits = sum(
        int(np.prod([len(v) for v in PARAM_GRIDS[m].values()])) * 5
        for m in model_names
    )
    print("=" * 62)
    print(f"Hyperparameter Search — {feature_group.upper()} feature group")
    print("=" * 62)
    print(f"Models      : {model_names}")
    print(f"Total fits  : {total_fits}")

    rows = [
        tune_model(m, df, feature_group=feature_group, out_dir=out_dir)
        for m in model_names
    ]
    summary = pd.DataFrame(rows)
    out_csv = Path(out_dir) / "hyperparameter_search_results.csv"
    summary.to_csv(out_csv, index=False)

    print(f"\n{'=' * 62}")
    print("Results")
    print("=" * 62)
    print(summary[["model", "cv_best_score", "cv_best_std", "tuned_test_auc"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    print(f"\nFull results saved → {out_csv}")

    if len(summary) > 1:
        best = summary.loc[summary["tuned_test_auc"].idxmax()]
        print(f"\n🏆  Best tuned model: {best['model']}  AUC = {best['tuned_test_auc']:.4f}")

    return summary
