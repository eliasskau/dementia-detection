"""
explain.py
----------
SHAP explanations for trained pipelines.

For each model type a different explainer is used:
    SVM / LR        -> KernelExplainer (model-agnostic, uses a background sample)
    RandomForest    -> TreeExplainer
    XGBoost         -> TreeExplainer

global_shap(pkl_path, df) returns a DataFrame of mean |SHAP| per feature,
sorted descending — one row per selected feature.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap


def _load(pkl_path: Path) -> tuple:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj["pipeline"], obj["label_encoder"], obj["feature_cols"]


def _preprocess(pipeline, X: np.ndarray) -> np.ndarray:
    """Run all pipeline steps except the final classifier."""
    steps = list(pipeline.steps[:-1])
    X_t = X.copy()
    for _, step in steps:
        X_t = step.transform(X_t)
    return X_t


def _feature_names_after_gini(pipeline, feature_cols: list[str]) -> list[str]:
    if "gini_selector" in pipeline.named_steps:
        mask = pipeline.named_steps["gini_selector"].selected_mask_
        return [c for c, keep in zip(feature_cols, mask) if keep]
    return list(feature_cols)


def global_shap(
    pkl_path: Path,
    df: pd.DataFrame,
    n_background: int = 50,
    max_samples: int = 200,
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values across the dataset.

    Parameters
    ----------
    pkl_path     : path to a .pkl saved by train_single()
    df           : combined features DataFrame (must contain feature_cols)
    n_background : background dataset size for KernelExplainer
    max_samples  : max rows explained (KernelExplainer is slow; subsampled)

    Returns
    -------
    DataFrame with columns [feature, mean_abs_shap], sorted descending.
    """
    pipeline, le, feature_cols = _load(pkl_path)
    clf_name = type(pipeline.named_steps["clf"]).__name__.lower()

    X_raw = df[feature_cols].values.astype(np.float32)
    X_pre = _preprocess(pipeline, X_raw)
    names = _feature_names_after_gini(pipeline, feature_cols)

    clf = pipeline.named_steps["clf"]

    if "randomforest" in clf_name or "xgb" in clf_name:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_pre)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        rng = np.random.RandomState(42)
        bg_idx = rng.choice(len(X_pre), size=min(n_background, len(X_pre)), replace=False)
        background = X_pre[bg_idx]

        def predict_proba_dementia(x):
            return clf.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(predict_proba_dementia, background)
        sample_idx = rng.choice(len(X_pre), size=min(max_samples, len(X_pre)), replace=False)
        shap_values = explainer.shap_values(X_pre[sample_idx], silent=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
