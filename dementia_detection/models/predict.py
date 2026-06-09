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
