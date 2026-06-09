"""
predict.py
----------
Load a saved model pipeline and run inference on new data.

Usage (Python API)
------------------
    from src.models.predict import predict

    predictions = predict(
        model_path="results/models/cookie/xgboost__all.pkl",
        data={"acoustic__F0semitoneFrom27.5Hz_sma3nz_amean": 32.8, ...},
    )
    # → {"label": "Dementia", "probability": 0.87}

    # Or pass a DataFrame / CSV path:
    predictions = predict(
        model_path="results/models/cookie/xgboost__all.pkl",
        data="path/to/new_features.csv",
    )

Usage (CLI)
-----------
    conda run -n dementia-detection python -m src.models.predict \\
        --model results/models/cookie/xgboost__all.pkl \\
        --data  path/to/new_features.csv \\
        --out   predictions.csv
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as _SKPipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config.config import COMBINED_DIR, GINI_THRESHOLD as _GINI_THRESHOLD, MODELS_DIR, TASK
from .train import GiniSelector


# ---------------------------------------------------------------------------
# Core predict function
# ---------------------------------------------------------------------------

def predict(
    model_path: str | Path,
    data: str | Path | pd.DataFrame | dict,
    return_proba: bool = True,
) -> pd.DataFrame:
    """
    Run inference using a saved model pipeline.

    Parameters
    ----------
    model_path : str | Path
        Path to a .pkl file saved by train.py.
    data : str | Path | pd.DataFrame | dict
        Input features.  Can be:
          - A CSV file path (same schema as combined feature CSVs)
          - A pandas DataFrame
          - A single dict {feature_name: value}
    return_proba : bool
        If True, include a `probability` column (P(Dementia)).

    Returns
    -------
    pd.DataFrame with columns: stem (if present), label, probability (if requested)
    """
    # Load artifact
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    pipe         = artifact["pipeline"]
    le           = artifact["label_encoder"]
    feature_cols = artifact["feature_cols"]

    # Coerce input to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Check all required features are present
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} required feature columns are missing from input.\n"
            f"First 5 missing: {missing[:5]}"
        )

    X = df[feature_cols].values.astype(np.float32)

    y_pred  = pipe.predict(X)
    y_label = le.inverse_transform(y_pred)

    out = pd.DataFrame({"label": y_label})

    if return_proba:
        y_prob = pipe.predict_proba(X)[:, 1]
        out["probability"] = y_prob

    if "stem" in df.columns:
        out.insert(0, "stem", df["stem"].values)

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run inference with a saved model.")
    p.add_argument("--model", required=True, help="Path to .pkl model file.")
    p.add_argument("--data",  required=True, help="Path to input CSV with feature columns.")
    p.add_argument("--out",   default=None,  help="Path to save predictions CSV (optional).")
    p.add_argument("--no-proba", action="store_true", help="Do not include probability column.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    predictions = predict(
        model_path=args.model,
        data=args.data,
        return_proba=not args.no_proba,
    )

    print(predictions.to_string(index=False))

    if args.out:
        predictions.to_csv(args.out, index=False)
        print(f"\nPredictions saved → {args.out}")


# ---------------------------------------------------------------------------
# DementiaRiskPredictor — deployment wrapper
# ---------------------------------------------------------------------------

class DementiaRiskPredictor:
    """
    Deployment wrapper around a trained sklearn Pipeline.

    Exposes human-readable outputs:
        predict_risk(df_row)  → float [0, 100]  (50 = decision boundary)
        predict_batch(df)     → list[float]
        top_features(n)       → list[(name, importance_0_100)]
    """

    def __init__(
        self,
        pipeline,
        label_encoder: LabelEncoder,
        feature_cols: list,
        gini_importances: np.ndarray,
        metadata: dict,
    ) -> None:
        self.pipeline      = pipeline
        self.label_encoder = label_encoder
        self.feature_cols  = feature_cols
        self._gini_imp     = gini_importances
        self.metadata      = metadata

        # Strip modality prefix from selected feature names (e.g. "liwc__WC" → "WC")
        mask   = pipeline.named_steps["gini_selector"].selected_mask_
        prefix = (feature_cols[0].rsplit("__", 1)[0] + "__") if feature_cols else ""
        self.feature_names: list = [
            c.replace(prefix, "", 1) for c, keep in zip(feature_cols, mask) if keep
        ]

    def _to_array(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:5]} …")
        return df[self.feature_cols].values.astype(np.float32)

    def predict_risk(self, df_row: pd.DataFrame) -> float:
        """Return a single dementia risk score in [0, 100]."""
        prob = self.pipeline.predict_proba(self._to_array(df_row.iloc[[0]]))[0, 1]
        return round(float(prob) * 100, 2)

    def predict_batch(self, df: pd.DataFrame) -> list:
        """Return dementia risk scores [0, 100] for every row in *df*."""
        probs = self.pipeline.predict_proba(self._to_array(df))[:, 1]
        return [round(float(p) * 100, 2) for p in probs]

    def top_features(self, n: int = 10) -> list:
        """
        Return the *n* most important features, scaled so that the most
        important feature = 100.
        """
        imp = self._gini_imp.copy()
        if imp.max() == 0:
            return [(name, 0.0) for name in self.feature_names[:n]]
        scaled = (imp / imp.max()) * 100.0
        pairs  = sorted(zip(self.feature_names, scaled), key=lambda x: -x[1])
        return [(name, round(float(s), 2)) for name, s in pairs[:n]]

    def __repr__(self) -> str:
        m = self.metadata
        return (
            f"DementiaRiskPredictor(\n"
            f"  model          = {m.get('model_name')}\n"
            f"  feature_group  = {m.get('feature_group')}\n"
            f"  trained_on     = {m.get('n_samples')} samples "
            f"({m.get('n_subjects')} subjects)\n"
            f"  n_features     = {m.get('n_features_raw')} raw → "
            f"{m.get('n_features_selected')} Gini-selected\n"
            f"  rskf_auc_mean  = {m.get('rskf_auc_mean')}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# export_best_model — train best config on full dataset, save predictor
# ---------------------------------------------------------------------------

def export_best_model(
    task: str = TASK,
    feature_group: str = "liwc",
    out_path: "Path | None" = None,
) -> DementiaRiskPredictor:
    """
    Train SVM + LIWC (Gini-selected) on the **full** dataset and save a
    :class:`DementiaRiskPredictor` to disk.

    Parameters
    ----------
    task          : Pitt task to use (default: ``config.TASK``)
    feature_group : feature modality prefix (default: ``"liwc"``)
    out_path      : destination .pkl path
                    (default: ``results/models/best_model.pkl``)

    Returns
    -------
    :class:`DementiaRiskPredictor` instance.
    """
    if out_path is None:
        out_path = MODELS_DIR / "best_model.pkl"
    out_path = Path(out_path)

    print("=" * 60)
    print(f"Exporting best model — SVM + {feature_group.upper()} (Gini-selected)")
    print("=" * 60)

    csv_path     = COMBINED_DIR / f"{task}_features.csv"
    print(f"\nLoading : {csv_path}")
    df           = pd.read_csv(csv_path)
    df["subject_id"] = df["stem"].apply(lambda s: s.split("-")[0])
    prefix       = f"{feature_group}__"
    feature_cols = [c for c in df.columns if c.startswith(prefix)]
    X            = df[feature_cols].values.astype(np.float32)
    le           = LabelEncoder()
    y            = le.fit_transform(df["label"].values)

    print(f"  Samples  : {len(df)}   {dict(df['label'].value_counts())}")
    print(f"  Subjects : {df['subject_id'].nunique()}")
    print(f"  Features : {len(feature_cols)} raw {feature_group.upper()} columns")

    gini_thresh = _GINI_THRESHOLD.get(feature_group, 0.008)
    pipe = _SKPipeline([
        ("imputer",       SimpleImputer(strategy="constant", fill_value=0.0)),
        ("gini_selector", GiniSelector(threshold=gini_thresh)),
        ("scaler",        StandardScaler()),
        ("clf",           SVC(kernel="rbf", C=1.0, gamma="scale",
                              class_weight="balanced", probability=True,
                              random_state=42)),
    ])

    print(f"\nTraining SVM+{feature_group.upper()} on all {len(df)} samples …")
    pipe.fit(X, y)

    selector   = pipe.named_steps["gini_selector"]
    n_selected = selector.n_selected_
    gini_imp   = selector.importances_selected()
    print(f"  Gini selection: {len(feature_cols)} → {n_selected} features retained")

    metadata = {
        "model_name":          "svm",
        "feature_group":       feature_group,
        "task":                task,
        "n_samples":           len(df),
        "n_subjects":          int(df["subject_id"].nunique()),
        "n_features_raw":      len(feature_cols),
        "n_features_selected": n_selected,
        "gini_threshold":      gini_thresh,
        "rskf_auc_mean":       None,   # fill after `make train`
        "label_classes":       list(le.classes_),
        "trained_on":          "full dataset",
    }

    predictor = DementiaRiskPredictor(
        pipeline=pipe, label_encoder=le,
        feature_cols=feature_cols, gini_importances=gini_imp,
        metadata=metadata,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(predictor, f)
    print(f"\nSaved  : {out_path}")

    ctrl = df[df["label"] == "Control"].iloc[[0]]
    dem  = df[df["label"] == "Dementia"].iloc[[0]]
    print(f"\n  Control sample  → risk {predictor.predict_risk(ctrl):.1f}/100")
    print(f"  Dementia sample → risk {predictor.predict_risk(dem):.1f}/100")

    print("\n── Top 10 features (Gini importance 0–100) ──")
    for name, score in predictor.top_features(10):
        print(f"  {name:<35} {score:>6.1f}  {'█' * int(score / 5)}")

    print(f"\n{predictor}\nDone.")
    return predictor
