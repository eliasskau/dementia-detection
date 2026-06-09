#!/usr/bin/env python
"""
11_evaluate_best_model.py
-------------------------
Deep evaluation of the best model (SVM + LIWC):
    1. SHAP global feature importance bar chart
    2. Calibration curve (reliability diagram)
    3. Permutation test  -> empirical p-value

Reads  : results/models/cookie/svm__liwc.pkl
         Pitt/processed/combined/cookie_features.csv
Writes : results/figures/shap_summary.png
         results/figures/calibration_curve.png
         results/figures/permutation_test.png

Run
---
    conda run -n dementia-detection python scripts/11_evaluate_best_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import FIGURES_DIR, MODELS_DIR, COMBINED_DIR, TASK, RANDOM_STATE
from src.models.explain import global_shap

PKL_PATH = MODELS_DIR / TASK / "svm__liwc.pkl"
DATA_CSV = COMBINED_DIR / f"{TASK}_features.csv"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_CSV)
    df["subject_id"] = df["stem"].apply(lambda s: s.split("-")[0])
    return df


# ---------------------------------------------------------------------------
# 1. SHAP global summary
# ---------------------------------------------------------------------------

def plot_shap(df: pd.DataFrame, top_n: int = 20) -> None:
    print("Computing SHAP values (KernelExplainer — this takes ~2 min)...")
    shap_df = global_shap(PKL_PATH, df)

    top = shap_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#2196F3")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} LIWC Features — SVM (global importance)")
    plt.tight_layout()
    out = FIGURES_DIR / "shap_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")

    print(f"\n{'Feature':<35} {'Mean |SHAP|':>12}")
    print("─" * 50)
    for _, row in top.iterrows():
        print(f"  {row['feature']:<33} {row['mean_abs_shap']:>10.4f}")


# ---------------------------------------------------------------------------
# 2. Calibration curve
# ---------------------------------------------------------------------------

def plot_calibration(df: pd.DataFrame) -> None:
    import pickle
    with open(PKL_PATH, "rb") as f:
        obj = pickle.load(f)
    pipe, le, feat_cols = obj["pipeline"], obj["label_encoder"], obj["feature_cols"]

    X = df[feat_cols].values.astype(np.float32)
    y = le.transform(df["label"].values)
    y_prob = pipe.predict_proba(X)[:, 1]

    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", color="#E53935", label="SVM + LIWC")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (Dementia)")
    ax.set_title("Calibration Curve — SVM + LIWC")
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# 3. Permutation test
# ---------------------------------------------------------------------------

def permutation_test(df: pd.DataFrame, n_permutations: int = 1000) -> None:
    import pickle
    with open(PKL_PATH, "rb") as f:
        obj = pickle.load(f)
    pipe, le, feat_cols = obj["pipeline"], obj["label_encoder"], obj["feature_cols"]

    X = df[feat_cols].values.astype(np.float32)
    y = le.transform(df["label"].values)
    true_auc = roc_auc_score(y, pipe.predict_proba(X)[:, 1])

    rng = np.random.RandomState(RANDOM_STATE)
    null_aucs = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        null_aucs.append(roc_auc_score(y_perm, pipe.predict_proba(X)[:, 1]))

    p_value = float(np.mean(np.array(null_aucs) >= true_auc))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_aucs, bins=40, color="#B0BEC5", edgecolor="white", label="Null distribution")
    ax.axvline(true_auc, color="#E53935", linewidth=2,
               label=f"Observed AUC = {true_auc:.3f}\np = {p_value:.4f}")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test ({n_permutations} permutations)")
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "permutation_test.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Permutation test: AUC={true_auc:.3f}  p={p_value:.4f}")
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Evaluating SVM + LIWC")
    print("=" * 60)

    if not PKL_PATH.exists():
        print(f"Model not found: {PKL_PATH}")
        print("Run scripts/08_train_models.py first.")
        return

    df = load_data()
    print(f"Loaded {len(df)} samples\n")

    print("── 1. SHAP ──────────────────────────────────────────────────")
    plot_shap(df)

    print("\n── 2. Calibration ───────────────────────────────────────────")
    plot_calibration(df)

    print("\n── 3. Permutation test ──────────────────────────────────────")
    permutation_test(df)

    print("\nDone. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
