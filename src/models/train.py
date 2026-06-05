"""
train.py
--------
Train three classifiers on combined acoustic + syntactic + lexical features.

Evaluation protocol
-------------------
    1. Subject-aware 80/20 train/test split
       - Split is done at SUBJECT level (not session level)
       - All sessions of the same speaker stay on the same side
       - Stratified by diagnosis label
       - The 20% test set is held out and NEVER seen during training

    2. 5-fold cross-validation on the 80% train set
       - Also subject-aware (no subject bleeds across folds)
       - Used to report generalisation variance (CV-AUC +/- std)

    3. Final model fit on full 80% train set -> evaluated on 20% test set
       - test_auc / test_f1 / test_acc are the real reported numbers

Tasks with fewer than 5 minority-class subjects are skipped
(recall/sentence/fluency are ~99% Dementia so they cannot be split meaningfully).

Outputs
-------
    results/models/{task}/{model}__{feature_group}.pkl
    results/models/{task}/{model}__{feature_group}__cv_metrics.json
    results/models/training_summary.csv
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# GiniSelector — fit-on-train-only feature selection via RF Gini importance
# ---------------------------------------------------------------------------

class GiniSelector(BaseEstimator, TransformerMixin):
    """
    Select features whose mean Gini impurity decrease (RF feature importance)
    exceeds `threshold`, computed on the training data only.

    Why Gini selection instead of PCA for single-modality groups:
      - Retains *named* features so SHAP values remain interpretable
      - Drops genuinely uninformative features (e.g. liwc__Emoji, liwc__illness,
        liwc__politic all have Gini ≈ 0 on cookie task — they never appear in
        dementia/cookie speech and only add noise to the covariance matrix)
      - Fitted inside the pipeline so the test set is never seen during selection
        (no label leakage)

    Threshold choice (0.008):
      - Acoustic: keeps 81/88 features (drops 7 near-zero tail)
      - LIWC:     keeps 57/118 features (drops 61, including 30 with Gini < 0.001)
      - Lexical/syntactic: small enough that all features survive; selector is a no-op
    """

    def __init__(self, threshold: float = 0.008, n_estimators: int = 200,
                 random_state: int = 42):
        self.threshold    = threshold
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.selected_mask_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        self.selected_mask_ = rf.feature_importances_ >= self.threshold
        # Always keep at least 1 feature to avoid empty arrays
        if not self.selected_mask_.any():
            self.selected_mask_[np.argmax(rf.feature_importances_)] = True
        # Store full importances so callers can retrieve per-selected-feature scores
        self._full_importances: np.ndarray = rf.feature_importances_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_mask_]

    def importances_selected(self) -> np.ndarray:
        """Return Gini importances for the Gini-selected features only."""
        return self._full_importances[self.selected_mask_]

    def importances_scaled(self) -> np.ndarray:
        """Gini importances for selected features, scaled so max = 100."""
        imp = self.importances_selected()
        return (imp / imp.max() * 100.0) if imp.max() > 0 else imp

    @property
    def n_selected_(self) -> int:
        return int(self.selected_mask_.sum()) if self.selected_mask_ is not None else 0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_TASKS = ("cookie", "fluency", "recall", "sentence")
_MIN_MINORITY_SUBJECTS = 5  # skip task if minority class has fewer subjects

MODELS: dict[str, tuple] = {
    "logistic_regression": (
        LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=42
        ),
        True,   # needs StandardScaler
    ),
    "svm": (
        SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=42,
        ),
        True,   # SVM always needs scaling
    ),
    "random_forest": (
        RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        False,
    ),
    "xgboost": (
        XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
            # class imbalance: scale_pos_weight = n_negative / n_positive
            # computed dynamically in _build_pipeline → set to 1.0 as placeholder,
            # overridden in train_single() after the split is known
            scale_pos_weight=1.0,
        ),
        False,
    ),
}

FEATURE_GROUPS: dict[str, str | None] = {
    "acoustic":        "acoustic__",
    "syntactic":       "syntactic__",
    "lexical":         "lexical__",
    "liwc":            "liwc__",
    "response_length": "response_length__",
    "all":             None,
    # all_pca: same feature set as "all" but PCA(95% variance) applied inside
    # the pipeline.  Only used with linear models (LR, SVM) — tree models are
    # automatically skipped because PCA removes the named-feature structure that
    # makes them both accurate and interpretable via SHAP.
    "all_pca":         None,
}

# Feature groups where PCA dimensionality reduction is applied.
# Justification for NOT applying PCA to single-modality groups:
#   1. Interpretability — single-group SHAP values map to named features
#      (e.g. liwc__nonflu, acoustic__F0mean).  PCA principal components have
#      no semantic meaning and cannot support feature-importance analysis.
#   2. Regularisation suffices — LIWC (118 feats, n/p ≈ 3.8) and acoustic
#      (88 feats, n/p ≈ 5.0) are within ranges where L2 regularisation in
#      LR/SVM and ensemble diversity in RF/XGB already prevent overfitting.
#   3. PCA is only justified for the "all" group (262 feats, n/p ≈ 1.7) where
#      correlated cross-modal features violate the assumptions of a linear
#      decision boundary and exceed the recommended 10 samples-per-feature.
_PCA_GROUPS = {"all_pca"}

# Models for which PCA is meaningful.  Tree-based ensembles split on individual
# features, so PCA rotations destroy interpretability without accuracy benefit
# (empirically, RF/XGB show no AUC improvement with PCA on tabular data).
_PCA_MODELS = {"logistic_regression", "svm"}

# Feature groups where Gini-based feature selection is applied inside the pipeline.
# Justification for selecting only acoustic and LIWC (not lexical/syntactic):
#   - acoustic (88 feats): Gini ranking reveals a clear tail of near-zero importance
#     features (threshold 0.008 keeps 81/88) — marginal but principled cleanup.
#   - liwc (118 feats): 30 features have Gini < 0.001 (e.g. liwc__Emoji,
#     liwc__illness, liwc__politic, liwc__substances — essentially absent from
#     cookie-task speech).  Threshold 0.008 keeps 57/118, halving the LIWC
#     dimensionality while retaining all informative categories.
#   - lexical (33 feats) and syntactic (23 feats) are small enough that all
#     features carry meaningful weight; applying a Gini filter would risk
#     dropping genuinely informative low-variance features in these groups.
#   - response_length (2 feats): too few features to filter.
_GINI_GROUPS = {"acoustic", "liwc"}

# Per-group Gini thresholds (features below threshold are dropped).
# acoustic: threshold 0.012 keeps top 30/88 features — the core eGeMAPS
#   descriptors (jitter, F0 percentiles, MFCCs 2-4, voiced/unvoiced segment
#   stats, F2/F3 formants, shimmer, spectral slope).  Ranks 31-88 are
#   redundant variants of the same measures (different normalisation /
#   frequency bands) with near-identical importance values, so retaining them
#   only inflates the feature:sample ratio without adding signal.
# liwc: threshold 0.008 keeps 57/118 features, dropping 61 categories that
#   are essentially absent from cookie-task speech (liwc__Emoji, liwc__illness,
#   liwc__politic, liwc__substances, etc. — all Gini < 0.001).
_GINI_THRESHOLD: dict[str, float] = {
    "acoustic": 0.012,
    "liwc":     0.008,
}

# ---------------------------------------------------------------------------
# Ablation groups — leave-one-modality-out study
# ---------------------------------------------------------------------------
# Groups are removed in order of weakest individual Test-AUC (weakest first):
#   acoustic (0.55) → response_length (0.56) → syntactic (0.57) → lexical (0.65)
#
# Each entry maps a group name to the set of PREFIXES TO EXCLUDE from "all".
# _feature_cols() handles the exclusion logic.
#
# Reading the ablation table:
#   all                       = baseline (all 5 modalities)
#   ablation_no_acoustic      = drop acoustic       → what does rl+syn+lex+liwc score?
#   ablation_no_ac_rl         = drop acoustic+rl    → what does syn+lex+liwc score?
#   ablation_no_ac_rl_syn     = drop +syntactic     → what does lex+liwc score?
#   ablation_no_ac_rl_syn_lex = drop +lexical       → = liwc only (sanity check)
#
# GiniSelector is applied to acoustic__ and liwc__ prefixes within each ablation
# group exactly as it is in the plain single-group training runs.
ABLATION_GROUPS: dict[str, frozenset[str]] = {
    "ablation_no_acoustic":      frozenset({"acoustic__"}),
    "ablation_no_ac_rl":         frozenset({"acoustic__", "response_length__"}),
    "ablation_no_ac_rl_syn":     frozenset({"acoustic__", "response_length__", "syntactic__"}),
    "ablation_no_ac_rl_syn_lex": frozenset({"acoustic__", "response_length__", "syntactic__", "lexical__"}),
}

# Register ablation groups in FEATURE_GROUPS so the CLI accepts them
# (value = None signals "use all columns minus exclusions", resolved in _feature_cols)
FEATURE_GROUPS.update({k: None for k in ABLATION_GROUPS})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subject_id(stem: str) -> str:
    return stem.split("-")[0]


def _feature_cols(df: pd.DataFrame, group: str) -> list[str]:
    meta = {"stem", "label", "subject_id"}
    if group in ABLATION_GROUPS:
        excluded = ABLATION_GROUPS[group]
        return [c for c in df.columns
                if c not in meta and not any(c.startswith(p) for p in excluded)]
    prefix = FEATURE_GROUPS[group]
    if prefix is None:
        return [c for c in df.columns if c not in meta]
    return [c for c in df.columns if c.startswith(prefix)]


def _load_task(combined_dir: Path, task: str) -> pd.DataFrame:
    path = combined_dir / f"{task}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Combined features not found: {path}")
    df = pd.read_csv(path)
    df["subject_id"] = df["stem"].apply(_subject_id)
    return df


def _build_pipeline(model_name: str, estimator, needs_scaling: bool,
                    use_pca: bool = False, use_gini: bool = False,
                    feature_group: str = "") -> Pipeline:
    # Always impute first (handles liwc__Tone NaN from LIWC-22 when text is too short)
    steps = [("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    if use_gini:
        gini_thresh = _GINI_THRESHOLD.get(feature_group, 0.008)
        steps.append(("gini_selector", GiniSelector(threshold=gini_thresh)))
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    if use_pca:
        # Retain 95% of variance.  whiten=True gives PCA-whitened features,
        # which is beneficial for distance-based models (SVM, LR).
        steps.append(("pca", PCA(n_components=0.95, whiten=True, random_state=42)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Subject-aware 80 / 20 split
# ---------------------------------------------------------------------------

def _subject_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split at subject level (not sample level), stratified by diagnosis.

    Steps:
      1. Compute one majority label per subject.
      2. StratifiedKFold on subjects with n_splits = round(1/test_size).
      3. Take the first split's test fold as the held-out 20%.

    Raises ValueError if the minority class has fewer than
    _MIN_MINORITY_SUBJECTS subjects.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(df["label"].values)

    subject_ids = df["subject_id"].values
    unique_subj = np.unique(subject_ids)

    # Subject-level label (majority vote across sessions)
    subj_label = np.array([
        int(y_enc[subject_ids == s].mean() >= 0.5)
        for s in unique_subj
    ])

    counts = np.bincount(subj_label)
    minority_n = int(counts.min())
    if minority_n < _MIN_MINORITY_SUBJECTS:
        raise ValueError(
            f"Only {minority_n} minority-class subject(s) — need >= "
            f"{_MIN_MINORITY_SUBJECTS} to make a meaningful 80/20 split."
        )

    n_splits = max(2, int(round(1.0 / test_size)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(skf.split(unique_subj, subj_label))

    test_subjects = set(unique_subj[test_idx])
    test_mask = df["subject_id"].isin(test_subjects)

    return df[~test_mask].copy(), df[test_mask].copy()


# ---------------------------------------------------------------------------
# 5-fold CV on the 80% train set
# ---------------------------------------------------------------------------

def cv_on_train(
    df_train: pd.DataFrame,
    feature_group: str,
    model_name: str,
    n_splits: int = 5,
) -> dict:
    """Subject-aware stratified k-fold CV on the training split only."""
    cols = _feature_cols(df_train, feature_group)
    X = df_train[cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df_train["label"].values)

    subject_ids = df_train["subject_id"].values
    unique_subj = np.unique(subject_ids)
    subj_label = np.array([
        int(y[subject_ids == s].mean() >= 0.5)
        for s in unique_subj
    ])

    counts = np.bincount(subj_label)
    actual_splits = min(n_splits, int(counts.min()))
    if actual_splits < 2:
        null = {k: None for k in (
            "cv_auc_mean", "cv_auc_std",
            "cv_f1_mean",  "cv_f1_std",
            "cv_acc_mean", "cv_acc_std",
            "cv_n_folds",
        )}
        return null

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    estimator, needs_scaling = MODELS[model_name]
    use_pca  = (feature_group in _PCA_GROUPS) and (model_name in _PCA_MODELS)
    # Ablation groups retain liwc__ (and sometimes acoustic__) columns, so Gini
    # selection is applied with the liwc threshold (0.008) as the conservative cut.
    use_gini = feature_group in _GINI_GROUPS or feature_group in ABLATION_GROUPS

    fold_auc, fold_f1, fold_acc = [], [], []

    for tr_idx, val_idx in skf.split(unique_subj, subj_label):
        val_subjects = set(unique_subj[val_idx])
        val_mask = np.array([s in val_subjects for s in subject_ids])
        tr_mask  = ~val_mask

        X_tr,  X_val = X[tr_mask],  X[val_mask]
        y_tr,  y_val = y[tr_mask],  y[val_mask]

        if len(np.unique(y_val)) < 2 or len(np.unique(y_tr)) < 2:
            continue

        est_cv = clone(estimator)
        if model_name == "xgboost":
            n_neg = int(np.sum(y_tr == 0))
            n_pos = int(np.sum(y_tr == 1))
            est_cv.set_params(scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0)
        pipe = _build_pipeline(model_name, est_cv, needs_scaling, use_pca=use_pca, use_gini=use_gini, feature_group=feature_group)
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_val)[:, 1]
        y_pred = pipe.predict(X_val)

        fold_auc.append(roc_auc_score(y_val, y_prob))
        fold_f1.append(f1_score(y_val, y_pred, zero_division=0))
        fold_acc.append(accuracy_score(y_val, y_pred))

    if not fold_auc:
        null = {k: None for k in (
            "cv_auc_mean", "cv_auc_std",
            "cv_f1_mean",  "cv_f1_std",
            "cv_acc_mean", "cv_acc_std",
            "cv_n_folds",
        )}
        return null

    return {
        "cv_auc_mean": float(np.mean(fold_auc)),
        "cv_auc_std":  float(np.std(fold_auc)),
        "cv_f1_mean":  float(np.mean(fold_f1)),
        "cv_f1_std":   float(np.std(fold_f1)),
        "cv_acc_mean": float(np.mean(fold_acc)),
        "cv_acc_std":  float(np.std(fold_acc)),
        "cv_n_folds":  len(fold_auc),
    }


# ---------------------------------------------------------------------------
# Train + evaluate one (model, feature_group) combination
# ---------------------------------------------------------------------------

def train_single(
    df: pd.DataFrame,
    feature_group: str,
    model_name: str,
    out_dir: Path,
) -> dict:
    """
    Full pipeline for one model x feature-group combo:
      1. 80/20 subject-aware split
      2. 5-fold CV on the 80% train set  -> CV-AUC +/- std
      3. Fit on full 80% train set
      4. Evaluate on held-out 20% test set  -> test_auc / test_f1 / test_acc
      5. Save .pkl + metrics JSON
    """
    print(f"  [{model_name:22s}] {feature_group:10s} ...", end=" ", flush=True)

    # PCA is only meaningful for linear models; silently skip tree-based models
    # for all_pca (they are already trained under the plain "all" group).
    if feature_group in _PCA_GROUPS and model_name not in _PCA_MODELS:
        print("SKIP -- PCA not applied to tree-based models (see _PCA_MODELS)")
        return {"skipped": True, "reason": "PCA not applicable to tree-based models"}

    use_pca  = feature_group in _PCA_GROUPS
    use_gini = feature_group in _GINI_GROUPS or feature_group in ABLATION_GROUPS

    try:
        df_train, df_test = _subject_train_test_split(df)
    except ValueError as e:
        print(f"SKIP -- {e}")
        return {"skipped": True, "reason": str(e)}

    cols = _feature_cols(df, feature_group)
    le   = LabelEncoder().fit(df["label"].values)

    X_train = df_train[cols].values.astype(np.float32)
    y_train = le.transform(df_train["label"].values)
    X_test  = df_test[cols].values.astype(np.float32)
    y_test  = le.transform(df_test["label"].values)

    # --- CV on train set only ---
    cv = cv_on_train(df_train, feature_group, model_name)

    # --- Final fit on full 80% train ---
    estimator, needs_scaling = MODELS[model_name]
    estimator_cloned = clone(estimator)

    # For XGBoost: set scale_pos_weight = n_negative / n_positive from train set
    # (equivalent to class_weight="balanced" in sklearn models)
    if model_name == "xgboost":
        n_neg = int(np.sum(y_train == 0))   # Control
        n_pos = int(np.sum(y_train == 1))   # Dementia
        spw   = n_neg / n_pos if n_pos > 0 else 1.0
        estimator_cloned.set_params(scale_pos_weight=spw)

    pipe = _build_pipeline(model_name, estimator_cloned, needs_scaling, use_pca=use_pca, use_gini=use_gini, feature_group=feature_group)
    pipe.fit(X_train, y_train)

    # --- Evaluate on held-out 20% test ---
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    test_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else None
    test_f1  = float(f1_score(y_test, y_pred, zero_division=0))
    test_acc = float(accuracy_score(y_test, y_pred))

    test_metrics = {
        "test_auc":       test_auc,
        "test_f1":        test_f1,
        "test_acc":       test_acc,
        "train_samples":  int(len(y_train)),
        "test_samples":   int(len(y_test)),
        "train_subjects": int(df_train["subject_id"].nunique()),
        "test_subjects":  int(df_test["subject_id"].nunique()),
        "n_features_raw": int(len(cols)),
        "n_features_selected": (
            int(pipe.named_steps["gini_selector"].n_selected_)
            if use_gini else int(len(cols))
        ),
    }

    # Pretty print
    cv_str   = (f"{cv['cv_auc_mean']:.3f}+/-{cv['cv_auc_std']:.3f}"
                if cv.get("cv_auc_mean") is not None else "N/A")
    test_str = f"{test_auc:.3f}" if test_auc is not None else "N/A"
    print(
        f"CV-AUC={cv_str}  |  "
        f"Test-AUC={test_str}  Test-F1={test_f1:.3f}  Test-Acc={test_acc:.3f}"
    )

    # --- Persist ---
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"{model_name}__{feature_group}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {"pipeline": pipe, "label_encoder": le, "feature_cols": cols},
            f,
        )

    all_metrics = {**cv, **test_metrics}
    json_path = out_dir / f"{model_name}__{feature_group}__cv_metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


# ---------------------------------------------------------------------------
# Batch training
# ---------------------------------------------------------------------------

def train_all(
    combined_dir: Path,
    output_dir: Path,
    tasks: list[str] | None = None,
    model_names: list[str] | None = None,
    feature_groups: list[str] | None = None,
) -> dict:
    """Train all specified task/model/feature-group combinations."""
    tasks          = list(tasks)          if tasks          else list(_ALL_TASKS)
    model_names    = list(model_names)    if model_names    else list(MODELS.keys())
    feature_groups = list(feature_groups) if feature_groups else list(FEATURE_GROUPS.keys())

    all_results: dict = {}

    for task in tasks:
        print(f"\n{'='*62}\nTask: {task}\n{'='*62}")
        try:
            df = _load_task(combined_dir, task)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue

        label_counts = df["label"].value_counts().to_dict()
        subj_counts  = df.groupby("label")["subject_id"].nunique().to_dict()
        print(f"  Samples  : {len(df)}  {label_counts}")
        print(f"  Subjects : {subj_counts}")

        task_out = output_dir / task
        all_results[task] = {}

        for model_name in model_names:
            all_results[task][model_name] = {}
            for fg in feature_groups:
                all_results[task][model_name][fg] = train_single(
                    df, fg, model_name, task_out
                )

    _write_summary(all_results, output_dir)
    return all_results


def _write_summary(results: dict, output_dir: Path) -> None:
    rows = []
    for task, models in results.items():
        for model_name, fgs in models.items():
            for fg, m in fgs.items():
                if m.get("skipped"):
                    continue
                rows.append({
                    "task":           task,
                    "model":          model_name,
                    "feature_group":  fg,
                    "cv_auc_mean":    m.get("cv_auc_mean"),
                    "cv_auc_std":     m.get("cv_auc_std"),
                    "test_auc":       m.get("test_auc"),
                    "test_f1":        m.get("test_f1"),
                    "test_acc":       m.get("test_acc"),
                    "train_samples":  m.get("train_samples"),
                    "test_samples":   m.get("test_samples"),
                    "train_subjects": m.get("train_subjects"),
                    "test_subjects":  m.get("test_subjects"),
                })

    if not rows:
        print("\n[warning] No completed training runs to summarise.")
        return

    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "training_summary.csv"

    # Merge with existing results so incremental runs accumulate
    if out_path.exists():
        existing = pd.read_csv(out_path)
        # Drop rows that are being overwritten by this run (same task+model+feature_group)
        key = ["task", "model", "feature_group"]
        existing = existing[
            ~existing.set_index(key).index.isin(summary.set_index(key).index)
        ]
        summary = pd.concat([existing, summary], ignore_index=True)

    summary.to_csv(out_path, index=False)

    print(f"\n{'='*62}\nSummary saved -> {out_path}")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 150)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(summary.to_string(index=False))
