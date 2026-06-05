"""
evaluate.py
-----------
Load saved model pipelines and produce detailed evaluation reports.

For a saved model it computes:
    - ROC-AUC, F1, Accuracy, Precision, Recall
    - Confusion matrix
    - SHAP feature importances (top-20)
    - Classification report

Usage
-----
    from src.models.evaluate import evaluate_all
    evaluate_all(combined_dir, models_dir)

Or via the training script which calls this automatically after training.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_model(model_path: Path) -> dict:
    """Load a saved pipeline dict from disk."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _load_task_df(combined_dir: Path, task: str) -> pd.DataFrame:
    path = combined_dir / f"{task}_features.csv"
    df = pd.read_csv(path)
    df["subject_id"] = df["stem"].apply(lambda s: s.split("-")[0])
    return df


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: Path,
    combined_dir: Path,
    task: str,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a saved model on the full task dataset.

    This is an *in-sample* evaluation (all data used for fitting).
    For generalisation estimates, rely on the LOSO CV metrics saved
    during training.

    Returns a metrics dict.
    """
    artifact = load_model(model_path)
    pipe        = artifact["pipeline"]
    le          = artifact["label_encoder"]
    feature_cols = artifact["feature_cols"]

    df = _load_task_df(combined_dir, task)
    X  = df[feature_cols].values.astype(np.float32)
    y  = le.transform(df["label"].values)

    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]

    metrics = {
        "auc":       float(roc_auc_score(y, y_prob)),
        "f1":        float(f1_score(y, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall":    float(recall_score(y, y_pred, zero_division=0)),
    }

    cm = confusion_matrix(y, y_pred)

    if verbose:
        print(f"\nModel : {model_path.name}")
        print(f"Task  : {task}  ({len(df)} samples)")
        print(f"AUC   : {metrics['auc']:.4f}")
        print(f"F1    : {metrics['f1']:.4f}")
        print(f"Acc   : {metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=le.classes_))
        print("Confusion Matrix:")
        print(f"  {'':12s}  Pred Control  Pred Dementia")
        print(f"  True Control  {cm[0,0]:12d}  {cm[0,1]:13d}")
        print(f"  True Dementia {cm[1,0]:12d}  {cm[1,1]:13d}")

    # SHAP feature importances (top 20)
    shap_values = _shap_importance(pipe, X, feature_cols)
    if shap_values is not None:
        metrics["top20_features"] = shap_values[:20]

    return metrics


# ---------------------------------------------------------------------------
# SHAP importance
# ---------------------------------------------------------------------------

def _shap_importance(pipe, X: np.ndarray, feature_cols: list[str]) -> list[dict] | None:
    """Return list of {feature, importance} sorted descending, or None."""
    try:
        import shap
        clf = pipe[-1]   # last step in pipeline

        # Use TreeExplainer for tree models, LinearExplainer for LR
        model_type = type(clf).__name__
        if model_type in ("RandomForestClassifier", "XGBClassifier"):
            explainer = shap.TreeExplainer(clf)
            # If pipeline has a scaler, transform X first
            X_transformed = pipe[:-1].transform(X) if len(pipe) > 1 else X
            shap_vals = explainer.shap_values(X_transformed)
            # For binary classification shap_values may be list[2] or 3-D
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            elif shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
        elif model_type == "LogisticRegression":
            X_transformed = pipe[:-1].transform(X) if len(pipe) > 1 else X
            explainer = shap.LinearExplainer(clf, X_transformed)
            shap_vals = explainer.shap_values(X_transformed)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            return None

        mean_abs = np.abs(shap_vals).mean(axis=0)
        ranked = sorted(
            zip(feature_cols, mean_abs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": f, "importance": round(v, 6)} for f, v in ranked]

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    combined_dir: Path,
    models_dir: Path,
    tasks: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all saved models found under models_dir.

    Returns a summary DataFrame.
    """
    _ALL_TASKS = ("cookie", "fluency", "recall", "sentence")
    tasks = list(tasks) if tasks else list(_ALL_TASKS)

    rows = []
    for task in tasks:
        task_dir = models_dir / task
        if not task_dir.exists():
            continue

        for pkl_path in sorted(task_dir.glob("*.pkl")):
            # filename pattern: {model}__{feature_group}.pkl
            parts = pkl_path.stem.split("__")
            if len(parts) < 2:
                continue
            model_name, feature_group = parts[0], parts[1]

            if verbose:
                print(f"\n{'─'*60}")
            metrics = evaluate_model(pkl_path, combined_dir, task, verbose=verbose)

            rows.append({
                "task":          task,
                "model":         model_name,
                "feature_group": feature_group,
                **{k: v for k, v in metrics.items() if k != "top20_features"},
            })

    return pd.DataFrame(rows)
