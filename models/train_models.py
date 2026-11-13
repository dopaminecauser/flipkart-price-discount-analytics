"""
models/train_models.py — Train the two-stage ML engine on the SQLite data.

Stage 1 (classification): XGBoost — will_drop_next_7_days (binary).
Stage 2 (regression): CatBoost — the price-floor *ratio*
    (predicted_floor / current_price). Regression on a ratio instead of raw
    rupees keeps the loss scale-free across a catalogue priced from ₹1 to
    ₹80,000, so cheap products no longer get a floor larger than their own
    price. The API multiplies the ratio back by the current price.

Validation: TimeSeriesSplit walk-forward (no random splits, prevents target
leakage). After cross-validating, each stage is refit on the full training set
and that final model is what gets saved — a fold model trained on ~2/3 of the
data is not the model you want serving requests.

Both artifacts are saved as a dict carrying the exact feature column order and
the regression target mode, so inference can never silently score a model with
mismatched columns or a stale target definition.
"""

from __future__ import annotations

import pickle
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    f1_score,
    precision_score,
    mean_absolute_error,
    mean_squared_error,
)

import xgboost as xgb
from catboost import CatBoostRegressor

# Allow this script to import db/ (parent of models/) regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.features import compute_feature_vectors  # noqa: E402

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "trained"
DB_PATH = BASE_DIR.parent / "data" / "flipkart.db"

FEATURES = [
    "day_of_week", "month", "is_weekend", "days_until_mega_sale",
    "current_price", "mrp", "discount_percentage", "rating_count_velocity",
    "brand", "category", "price_segment",
]
CATEGORICALS = ["brand", "category", "price_segment"]
TARGET_MODE = "floor_ratio"


def build_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical block once, for both stages."""
    X = frame[FEATURES].copy()
    X = pd.get_dummies(X, columns=CATEGORICALS, drop_first=True)
    return X.astype(float)


def load_feature_dataframe() -> pd.DataFrame:
    """Connect to SQLite and return every product's current feature vector."""
    conn = sqlite3.connect(str(DB_PATH))
    vecs = compute_feature_vectors(conn)
    conn.close()
    return pd.DataFrame(vecs)


def save_artifact(name: str, model, features, extra: dict | None = None) -> Path:
    """Persist a model with the metadata inference needs to match it."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / name
    payload = {
        "model": model,
        "features": list(features),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def train_stage1(df: pd.DataFrame, save_path: str | None = None) -> dict:
    X = build_matrix(df)
    y = df["will_drop_next_7_days"].astype(int)

    cv_scores = {"f1": [], "precision": [], "accuracy": []}
    tscv = TimeSeriesSplit(n_splits=4)

    for train_idx, valid_idx in tscv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False, eval_set=[(X_valid, y_valid)])
        pred = model.predict(X_valid)
        cv_scores["f1"].append(f1_score(y_valid, pred))
        cv_scores["precision"].append(precision_score(y_valid, pred, zero_division=0))

    # Final model for serving: trained on every row, not just the last fold.
    final = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    final.fit(X, y, verbose=False)

    if save_path:
        save_artifact(f"{save_path}.pkl", final, X.columns)

    summary = {
        "fold_f1": [round(v, 4) for v in cv_scores["f1"]],
        "fold_precision": [round(v, 4) for v in cv_scores["precision"]],
        "mean_f1": float(np.mean(cv_scores["f1"])),
        "mean_precision": float(np.mean(cv_scores["precision"])),
        "positive_rate": float(y.mean()),
    }
    print("[Stage 1] Classification CV:", summary)
    return summary


def train_stage2(df: pd.DataFrame, save_path: str | None = None) -> dict:
    drop_mask = df["will_drop_next_7_days"] == 1
    sub = df[drop_mask].copy()
    if len(sub) < 10:
        print("[Stage 2] Too few drop products to train:", len(sub))
        return {}

    X = build_matrix(sub)
    price = sub["current_price"].clip(lower=1e-9)
    # Scale-free target: how far below today's price the floor sits.
    y = (sub["predicted_price_floor"] / price).clip(0.05, 1.0)

    tscv = TimeSeriesSplit(n_splits=4)
    fold_mae, fold_rmse = [], []

    for train_idx, valid_idx in tscv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="MAE",
            random_seed=SEED,
            verbose=False,
        )
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
        pred_ratio = np.clip(model.predict(X_valid), 0.05, 1.0)

        # Report in INR: the brief asks for MAE/RMSE in rupees.
        actual_inr = (y_valid * price.iloc[valid_idx]).to_numpy()
        pred_inr = pred_ratio * price.iloc[valid_idx].to_numpy()
        fold_mae.append(float(mean_absolute_error(actual_inr, pred_inr)))
        fold_rmse.append(float(np.sqrt(mean_squared_error(actual_inr, pred_inr))))

    final = CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        random_seed=SEED,
        verbose=False,
    )
    final.fit(X, y, verbose=False)

    if save_path:
        save_artifact(f"{save_path}.pkl", final, X.columns, {"target": TARGET_MODE})

    summary = {
        "fold_mae_inr": [round(v, 2) for v in fold_mae],
        "fold_rmse_inr": [round(v, 2) for v in fold_rmse],
        "mean_mae_inr": float(np.mean(fold_mae)),
        "mean_rmse_inr": float(np.mean(fold_rmse)),
        "train_rows": int(len(sub)),
    }
    print("[Stage 2] Regression CV (INR):", summary)
    return summary


def main() -> None:
    print("Loading feature dataframe from SQLite...")
    df = load_feature_dataframe()
    print(f"  Products: {len(df)}")

    print("\n=== Training Stage 1 (XGBoost classifier) ===")
    train_stage1(df, save_path="classifier")

    print("\n=== Training Stage 2 (CatBoost regressor) ===")
    train_stage2(df, save_path="regressor")

    print("\nDone. Models written to", MODELS_DIR)


if __name__ == "__main__":
    main()
