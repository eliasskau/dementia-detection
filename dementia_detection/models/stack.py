"""
stack.py
--------
Late-fusion stacking for multi-modal dementia detection.

Architecture (Balagopalan et al. 2021; Fraser et al. 2016)
----------------------------------------------------------
Level 0 — one SVM per modality, each trained on its own features:
    acoustic   → P(dementia | acoustic features)
    liwc       → P(dementia | LIWC features)
    lexical    → P(dementia | lexical features)
    syntactic  → P(dementia | syntactic features)

Level 1 — Logistic Regression meta-learner trained on the 4 probability
    outputs. Its coefficients reveal the relative contribution of each
    modality — the primary interpretable finding.

Evaluation
----------
Subject-aware RSKF (same protocol as train.py):
    - Level-0 models are fitted *only* on the training fold
    - Level-1 meta-learner receives out-of-fold predictions so it never
      sees level-0 train data → no leakage at either level
    - Final stack fitted on full dataset for deployment

Outputs
-------
    results/models/{task}/stack__cv_metrics.json
    results/models/{task}/stack.pkl
    Printed modality weights from the meta-learner

Public API
----------
run_stacking(task, combined_dir, output_dir) -> dict
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config.config import (
    BOOTSTRAP_CI, BOOTSTRAP_N, COMBINED_DIR, GINI_THRESHOLD,
    MODELS_DIR, RANDOM_STATE, RSKF_REPEATS, RSKF_SPLITS, TASK,
)
from .train import GiniSelector, _bootstrap_auc_ci

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Modality configuration
# Assign each modality its best-performing model class and feature prefix.
# SVM is used for all modalities for consistency — keeps the comparison clean.
# ---------------------------------------------------------------------------

MODALITIES: dict[str, dict] = {
    "liwc":      {"prefix": "liwc__",            "gini_threshold": GINI_THRESHOLD["liwc"]},
    "lexical":   {"prefix": "lexical__",          "gini_threshold": None},
    "syntactic": {"prefix": "syntactic__",        "gini_threshold": None},
    "acoustic":  {"prefix": "acoustic__",         "gini_threshold": GINI_THRESHOLD["acoustic"]},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feature_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def _build_level0_pipeline(gini_threshold: float | None) -> Pipeline:
    """One SVM pipeline per modality, with optional Gini selection."""
    steps: list = [("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    if gini_threshold is not None:
        steps.append(("gini_selector", GiniSelector(threshold=gini_threshold)))
    steps.append(("scaler", StandardScaler()))
    steps.append(("clf", SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", probability=True, random_state=RANDOM_STATE,
    )))
    return Pipeline(steps)


def _subject_rskf_splits(
    df: pd.DataFrame,
    n_splits: int = RSKF_SPLITS,
    n_repeats: int = RSKF_REPEATS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate subject-aware RSKF index pairs.
    Returns list of (train_sample_idx, test_sample_idx).
    """
    subject_ids = df["subject_id"].values
    unique_subj = np.unique(subject_ids)
    le = LabelEncoder()
    y_enc = le.fit_transform(df["label"].values)
    subj_label = np.array([
        int(y_enc[subject_ids == s].mean() >= 0.5) for s in unique_subj
    ])

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    splits = []
    for subj_tr_idx, subj_te_idx in rskf.split(unique_subj, subj_label):
        te_subjects = set(unique_subj[subj_te_idx])
        te_mask = np.array([s in te_subjects for s in subject_ids])
        splits.append((np.where(~te_mask)[0], np.where(te_mask)[0]))
    return splits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_stacking(
    task: str = TASK,
    combined_dir: Path = COMBINED_DIR,
    output_dir: Path = MODELS_DIR,
) -> dict:
    """
    Run subject-aware RSKF stacking and return a results dict.

    Level-0: one SVM per modality (out-of-fold probabilities).
    Level-1: Logistic Regression meta-learner on the 4 OOF probability columns.
    """
    output_dir = Path(output_dir) / task
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(combined_dir) / f"{task}_features.csv"
    df = pd.read_csv(csv_path)
    df["subject_id"] = df["stem"].apply(lambda s: s.split("-")[0])

    le = LabelEncoder()
    y_all = le.fit_transform(df["label"].values)
    n = len(df)

    modality_names = list(MODALITIES.keys())
    n_modalities   = len(modality_names)

    print("=" * 62)
    print(f"Stacking — {task} task")
    print("=" * 62)
    print(f"Modalities  : {modality_names}")
    print(f"Samples     : {n}  {dict(df['label'].value_counts())}")
    print(f"RSKF        : {RSKF_SPLITS} splits × {RSKF_REPEATS} repeats = "
          f"{RSKF_SPLITS * RSKF_REPEATS} folds")

    splits = _subject_rskf_splits(df)

    fold_aucs: list[float] = []
    fold_f1s:  list[float] = []

    # Accumulate OOF meta-features across folds (for final weight display)
    oof_meta  = np.zeros((n, n_modalities))
    oof_count = np.zeros(n, dtype=int)

    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)
        y_tr  = le.transform(df_tr["label"].values)
        y_te  = le.transform(df_te["label"].values)

        # ── Level 0: fit one SVM per modality on train, predict on test ──────
        meta_tr = np.zeros((len(df_tr), n_modalities))
        meta_te = np.zeros((len(df_te), n_modalities))

        for m_idx, (mod_name, cfg) in enumerate(MODALITIES.items()):
            cols = _feature_cols(df_tr, cfg["prefix"])
            pipe = _build_level0_pipeline(cfg["gini_threshold"])
            pipe.fit(df_tr[cols].values.astype(np.float32), y_tr)

            # Inner OOF for level-0 meta-features on training data
            inner_splits = _subject_rskf_splits(df_tr, n_splits=3, n_repeats=1)
            oof_tr = np.zeros(len(df_tr))
            for tr2, te2 in inner_splits:
                from sklearn.base import clone as _clone
                p = _clone(pipe)
                cols2 = _feature_cols(df_tr.iloc[tr2], cfg["prefix"])
                p.fit(df_tr.iloc[tr2][cols2].values.astype(np.float32), y_tr[tr2])
                oof_tr[te2] = p.predict_proba(
                    df_tr.iloc[te2][cols].values.astype(np.float32)
                )[:, 1]
            meta_tr[:, m_idx] = oof_tr

            # Level-0 test predictions from full-train-fold model
            meta_te[:, m_idx] = pipe.predict_proba(
                df_te[cols].values.astype(np.float32)
            )[:, 1]

        # ── Level 1: meta-learner ─────────────────────────────────────────
        meta_clf = LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        )
        meta_clf.fit(meta_tr, y_tr)
        y_prob = meta_clf.predict_proba(meta_te)[:, 1]
        y_pred = meta_clf.predict(meta_te)

        fold_aucs.append(roc_auc_score(y_te, y_prob))
        fold_f1s.append(f1_score(y_te, y_pred, zero_division=0))

        # Accumulate OOF meta-features
        oof_meta[te_idx]  += meta_te
        oof_count[te_idx] += 1

    # ── Aggregate ─────────────────────────────────────────────────────────
    fold_aucs_arr = np.array(fold_aucs)
    fold_f1s_arr  = np.array(fold_f1s)
    ci_low, ci_high = _bootstrap_auc_ci(y_all, (oof_meta / np.maximum(oof_count[:, None], 1)).mean(axis=1))

    rskf_auc_mean = float(fold_aucs_arr.mean())
    rskf_auc_std  = float(fold_aucs_arr.std())
    rskf_f1_mean  = float(fold_f1s_arr.mean())

    print(f"\nRSKF AUC  : {rskf_auc_mean:.4f} ± {rskf_auc_std:.4f}  "
          f"95% CI [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"RSKF F1   : {rskf_f1_mean:.4f}")

    # ── Final stack fitted on full dataset ────────────────────────────────
    print("\nFitting final stack on full dataset …")
    final_meta = np.zeros((n, n_modalities))
    final_pipes: dict[str, Pipeline] = {}

    for m_idx, (mod_name, cfg) in enumerate(MODALITIES.items()):
        cols = _feature_cols(df, cfg["prefix"])
        pipe = _build_level0_pipeline(cfg["gini_threshold"])
        pipe.fit(df[cols].values.astype(np.float32), y_all)
        final_meta[:, m_idx] = pipe.predict_proba(
            df[cols].values.astype(np.float32)
        )[:, 1]
        final_pipes[mod_name] = pipe

    final_meta_clf = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    )
    final_meta_clf.fit(final_meta, y_all)

    # ── Modality weights ──────────────────────────────────────────────────
    raw_coef    = final_meta_clf.coef_[0]
    coef_scaled = (raw_coef - raw_coef.min()) / (raw_coef.max() - raw_coef.min() + 1e-9) * 100

    print(f"\n{'Modality':<12} {'Raw coef':>10} {'Scaled (0-100)':>15}")
    print("─" * 42)
    for name, raw, scaled in zip(modality_names, raw_coef, coef_scaled):
        bar = "█" * int(scaled / 5)
        print(f"  {name:<10} {raw:>10.4f} {scaled:>12.1f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────
    metrics = {
        "task":              task,
        "model":             "stack",
        "feature_group":     "all_modalities",
        "modalities":        modality_names,
        "rskf_auc_mean":     round(rskf_auc_mean, 6),
        "rskf_auc_std":      round(rskf_auc_std, 6),
        "rskf_auc_ci_low":   round(ci_low, 6),
        "rskf_auc_ci_high":  round(ci_high, 6),
        "rskf_f1_mean":      round(rskf_f1_mean, 6),
        "rskf_n_folds":      len(splits),
        "modality_coefs":    {name: round(float(c), 6)
                              for name, c in zip(modality_names, raw_coef)},
        "modality_coefs_scaled": {name: round(float(c), 2)
                                  for name, c in zip(modality_names, coef_scaled)},
        "n_samples":         n,
    }

    json_path = output_dir / "stack__cv_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {json_path}")

    pkl_path = output_dir / "stack.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "level0_pipelines": final_pipes,
            "meta_clf":         final_meta_clf,
            "label_encoder":    le,
            "modality_names":   modality_names,
            "modalities":       MODALITIES,
            "metrics":          metrics,
        }, f)
    print(f"Model saved  → {pkl_path}")

    return metrics
