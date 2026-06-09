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


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_shap(
    pkl_path: Path,
    df: pd.DataFrame,
    top_n: int = 20,
    out_path: "Path | None" = None,
) -> Path:
    """
    Save a horizontal bar chart of mean |SHAP| values.

    Returns the path of the saved figure.
    """
    import matplotlib.pyplot as plt
    from config.config import FIGURES_DIR

    if out_path is None:
        out_path = FIGURES_DIR / "shap_summary.png"

    print("Computing SHAP values (KernelExplainer — ~2 min) …")
    shap_df = global_shap(pkl_path, df)
    top = shap_df.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#2196F3")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Features — global importance (SHAP)")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    print(f"\n{'Feature':<35} {'Mean |SHAP|':>12}")
    print("─" * 50)
    for _, row in top.iterrows():
        print(f"  {row['feature']:<33} {row['mean_abs_shap']:>10.4f}")

    return Path(out_path)


def plot_calibration(
    pkl_path: Path,
    df: pd.DataFrame,
    out_path: "Path | None" = None,
) -> Path:
    """
    Save a calibration curve (reliability diagram).

    Returns the path of the saved figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from config.config import FIGURES_DIR

    if out_path is None:
        out_path = FIGURES_DIR / "calibration_curve.png"

    pipeline, le, feature_cols = _load(pkl_path)
    X      = df[feature_cols].values.astype(np.float32)
    y      = le.transform(df["label"].values)
    y_prob = pipeline.predict_proba(X)[:, 1]

    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", color="#E53935", label="SVM + LIWC")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (Dementia)")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return Path(out_path)


def permutation_test(
    pkl_path: Path,
    df: pd.DataFrame,
    n_permutations: int = 1000,
    out_path: "Path | None" = None,
    random_state: int = 42,
) -> float:
    """
    Run a permutation test and save a null-distribution histogram.

    Returns the empirical p-value.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score as _roc
    from config.config import FIGURES_DIR

    if out_path is None:
        out_path = FIGURES_DIR / "permutation_test.png"

    pipeline, le, feature_cols = _load(pkl_path)
    X        = df[feature_cols].values.astype(np.float32)
    y        = le.transform(df["label"].values)
    true_auc = _roc(y, pipeline.predict_proba(X)[:, 1])

    rng       = np.random.RandomState(random_state)
    null_aucs = [_roc(rng.permutation(y), pipeline.predict_proba(X)[:, 1])
                 for _ in range(n_permutations)]
    p_value = float(np.mean(np.array(null_aucs) >= true_auc))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_aucs, bins=40, color="#B0BEC5", edgecolor="white",
            label="Null distribution")
    ax.axvline(true_auc, color="#E53935", linewidth=2,
               label=f"Observed AUC = {true_auc:.3f}   p = {p_value:.4f}")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test ({n_permutations} permutations)")
    ax.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  AUC = {true_auc:.3f}   p = {p_value:.4f}")
    print(f"  Saved: {out_path}")
    return p_value
