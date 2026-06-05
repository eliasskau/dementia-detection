#!/usr/bin/env python
"""
09_export_best_model.py
-----------------------
Trains the best-performing configuration (SVM + LIWC, Gini-selected) on the
FULL dataset and exports a DementiaRiskPredictor wrapper as best_model.pkl.

Why full dataset?
  The pkl files from 06_train_models.py are trained on the 80% train split
  only (the 20% test set is never seen during training, by design).  Once we
  have confirmed the best model via held-out evaluation, we retrain on all
  available data to maximise generalisation for deployment/inference.

Outputs
-------
  results/models/best_model.pkl
      Contains a DementiaRiskPredictor instance with:
        .predict_risk(df_row)   -> float  0-100  (dementia risk %)
        .predict_batch(df)      -> list[float]   (one score per row)
        .top_features(n=10)     -> list[(name, score_0_100)]
        .feature_names          -> list[str]     (Gini-selected LIWC feature names)
        .metadata               -> dict          (training info)

Run
---
    conda run -n dementia-detection python scripts/09_export_best_model.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import GiniSelector, _GINI_THRESHOLD  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — best model confirmed via held-out test set
# ---------------------------------------------------------------------------
TASK         = "cookie"
FEATURE_GROUP = "liwc"
FEATURE_PREFIX = "liwc__"
GINI_THRESHOLD = _GINI_THRESHOLD["liwc"]   # 0.008 — keeps 57/118 LIWC features

COMBINED_CSV = PROJECT_ROOT / "Pitt" / "processed" / "combined" / f"{TASK}_features.csv"
OUT_PATH     = PROJECT_ROOT / "results" / "models" / "best_model.pkl"


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------

class DementiaRiskPredictor:
    """
    Wrapper around the trained SVM+LIWC pipeline.

    Outputs
    -------
    predict_risk(df_row)   : float in [0, 100]
        Dementia risk score.  50 = decision boundary.
        > 50 → model predicts Dementia.

    predict_batch(df)      : list[float]
        Batch version of predict_risk.

    top_features(n)        : list[(feature_name, importance_0_100)]
        Global Gini importances of the selected features, scaled so that
        the most important feature = 100.  Useful for explaining which
        LIWC categories drive the model most.

    Notes
    -----
    - `df_row` / `df` must contain the raw LIWC feature columns (liwc__*).
    - Column ordering is handled automatically via self.feature_cols.
    - Missing values are imputed to 0.0 (same as training).
    """

    def __init__(
        self,
        pipeline,
        label_encoder: LabelEncoder,
        feature_cols: list[str],
        gini_importances: np.ndarray,
        metadata: dict,
    ):
        self.pipeline         = pipeline
        self.label_encoder    = label_encoder
        self.feature_cols     = feature_cols          # raw (pre-Gini) LIWC cols
        self._gini_imp        = gini_importances      # shape: (n_selected_features,)
        self.metadata         = metadata

        # Selected (post-Gini) feature names
        mask = pipeline.named_steps["gini_selector"].selected_mask_
        self.feature_names: list[str] = [
            c.replace(FEATURE_PREFIX, "")
            for c, keep in zip(feature_cols, mask) if keep
        ]

    # ------------------------------------------------------------------
    def _to_array(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and order feature columns from a DataFrame."""
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in input DataFrame: {missing[:5]} ..."
            )
        return df[self.feature_cols].values.astype(np.float32)

    # ------------------------------------------------------------------
    def predict_risk(self, df_row: pd.DataFrame) -> float:
        """
        Return a single dementia risk score in [0, 100].

        Parameters
        ----------
        df_row : pd.DataFrame with exactly 1 row (or any number of rows —
                 only the first row is used).

        Returns
        -------
        float : 0 = very likely Control, 100 = very likely Dementia.
        """
        X = self._to_array(df_row.iloc[[0]])
        prob = self.pipeline.predict_proba(X)[0, 1]   # P(Dementia)
        return round(float(prob) * 100, 2)

    # ------------------------------------------------------------------
    def predict_batch(self, df: pd.DataFrame) -> list[float]:
        """
        Return dementia risk scores [0, 100] for every row in df.

        Parameters
        ----------
        df : pd.DataFrame with N rows.

        Returns
        -------
        list[float] of length N.
        """
        X = self._to_array(df)
        probs = self.pipeline.predict_proba(X)[:, 1]
        return [round(float(p) * 100, 2) for p in probs]

    # ------------------------------------------------------------------
    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Return the top-n most important LIWC features, scaled so that
        the most important = 100.

        Parameters
        ----------
        n : number of features to return (default 10).

        Returns
        -------
        list of (feature_name, importance_0_100) sorted descending.
        """
        imp = self._gini_imp.copy()
        if imp.max() == 0:
            return [(name, 0.0) for name in self.feature_names[:n]]
        scaled = (imp / imp.max()) * 100.0
        pairs  = sorted(zip(self.feature_names, scaled), key=lambda x: -x[1])
        return [(name, round(float(score), 2)) for name, score in pairs[:n]]

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        meta = self.metadata
        return (
            f"DementiaRiskPredictor(\n"
            f"  model        = {meta.get('model_name')}\n"
            f"  feature_group= {meta.get('feature_group')}\n"
            f"  trained_on   = {meta.get('n_samples')} samples "
            f"({meta.get('n_subjects')} subjects)\n"
            f"  n_features   = {meta.get('n_features_raw')} raw → "
            f"{meta.get('n_features_selected')} Gini-selected\n"
            f"  test_auc     = {meta.get('test_auc')}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Build + export
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Export Best Model — SVM + LIWC (Gini-selected)")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────
    print(f"\nLoading : {COMBINED_CSV}")
    df = pd.read_csv(COMBINED_CSV)
    df["subject_id"] = df["stem"].apply(lambda s: s.split("-")[0])

    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    X = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(df["label"].values)

    print(f"  Samples  : {len(df)}  {dict(df['label'].value_counts())}")
    print(f"  Subjects : {df['subject_id'].nunique()}")
    print(f"  Features : {len(feature_cols)} raw LIWC columns")

    # ── Build pipeline (same as train.py) ─────────────────────────────
    estimator = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", probability=True, random_state=42,
    )
    steps = [
        ("imputer",       SimpleImputer(strategy="constant", fill_value=0.0)),
        ("gini_selector", GiniSelector(threshold=GINI_THRESHOLD)),
        ("scaler",        StandardScaler()),
        ("clf",           estimator),
    ]
    from sklearn.pipeline import Pipeline
    pipe = Pipeline(steps)

    # ── Fit on ALL data ────────────────────────────────────────────────
    print(f"\nTraining SVM+LIWC on all {len(df)} samples …")
    pipe.fit(X, y)

    n_selected = pipe.named_steps["gini_selector"].n_selected_
    print(f"  Gini selection: {len(feature_cols)} → {n_selected} features retained")

    # ── Extract Gini importances for the SELECTED features ─────────────
    # GiniSelector.importances_selected() returns importances for kept features.
    selector = pipe.named_steps["gini_selector"]
    gini_imp = selector.importances_selected()

    # ── Assemble metadata ──────────────────────────────────────────────
    metadata = {
        "model_name":         "svm",
        "feature_group":      FEATURE_GROUP,
        "task":               TASK,
        "n_samples":          len(df),
        "n_subjects":         df["subject_id"].nunique(),
        "n_features_raw":     len(feature_cols),
        "n_features_selected": n_selected,
        "gini_threshold":     GINI_THRESHOLD,
        "test_auc":           0.789,   # from held-out evaluation in 06_train_models.py
        "cv_auc_mean":        0.855,
        "cv_auc_std":         0.038,
        "label_classes":      list(le.classes_),
        "trained_on":         "full dataset (all 552 samples)",
    }

    # ── Wrap and save ─────────────────────────────────────────────────
    predictor = DementiaRiskPredictor(
        pipeline=pipe,
        label_encoder=le,
        feature_cols=feature_cols,
        gini_importances=gini_imp,
        metadata=metadata,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(predictor, f)
    print(f"\nSaved  : {OUT_PATH}")

    # ── Quick sanity check ─────────────────────────────────────────────
    print("\n── Sanity check ──────────────────────────────────────────────")
    sample_ctrl = df[df["label"] == "Control"].iloc[[0]]
    sample_dem  = df[df["label"] == "Dementia"].iloc[[0]]
    score_ctrl  = predictor.predict_risk(sample_ctrl)
    score_dem   = predictor.predict_risk(sample_dem)
    print(f"  Control sample  → risk score: {score_ctrl:.1f} / 100")
    print(f"  Dementia sample → risk score: {score_dem:.1f} / 100")

    print("\n── Top 10 features (Gini importance, scaled 0–100) ───────────")
    print(f"  {'Feature':<35} {'Importance':>12}")
    print("  " + "─" * 50)
    for name, score in predictor.top_features(10):
        bar = "█" * int(score / 5)
        print(f"  {name:<35} {score:>8.1f}  {bar}")

    print(f"\n{predictor}")
    print("\nDone.")


if __name__ == "__main__":
    main()
