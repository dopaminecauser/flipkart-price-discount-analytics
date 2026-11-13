"""
models/engine.py — Phase 3 Two-Stage Machine Learning Engine (inference)

Sequentially chained models:

    [feature vector] -> Stage 1  XGBoost classifier: will_drop_next_7_days?
                          |-- class 0 (stable) -> terminate, "Buy Now"
                          `-- class 1 (drop)   -> Stage 2 CatBoost regressor
                                                   -> price floor

Training lives in models/train_models.py (single source of truth). This module
only loads saved artifacts and scores records, so the serving path can never
drift from the training path.

Artifacts are dicts:
    {"model": <estimator>, "features": [<column order>], "target": "floor_ratio"}

The feature list travels with the model, so inference reindexes onto exactly the
columns seen in training. The `target` of the regressor is a *ratio*
(predicted_floor / current_price), keeping the loss scale-free from ₹1 items to
₹80,000 items; the floor is reconstructed by multiplying by the current price.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR
TRAINED_DIR = BASE_DIR / "trained"

# Features fed to both stages (categoricals are one-hot encoded below).
FEATURES = [
    "day_of_week", "month", "is_weekend", "days_until_mega_sale",
    "current_price", "mrp", "discount_percentage", "rating_count_velocity",
    "brand", "category", "price_segment",
]
CATEGORICALS = ["brand", "category", "price_segment"]

# Verdict thresholds. Falling back to a bare 0.5 cut-off collapses every
# prediction into the same badge, so there is a genuine "watch" band between
# confident stable and confident drop.
DROP_THRESHOLD = 0.5
WATCH_THRESHOLD = 0.3


class TwoStageMLEngine:
    """Sequentially chained classifier -> regressor."""

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.classifier = None
        self.regressor = None
        self.feature_columns: List[str] | None = None
        self.target_mode: str = "floor_ratio"

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self, classifier_path: Path, regressor_path: Path) -> "TwoStageMLEngine":
        """Load both saved artifacts (dict payload or bare estimator)."""
        with open(classifier_path, "rb") as f:
            clf_payload = pickle.load(f)
        with open(regressor_path, "rb") as f:
            reg_payload = pickle.load(f)

        self.classifier, clf_features = self._unpack(clf_payload)
        self.regressor, reg_features = self._unpack(reg_payload)
        self.feature_columns = reg_features or clf_features
        if isinstance(reg_payload, dict) and reg_payload.get("target"):
            self.target_mode = str(reg_payload["target"])
        return self

    @staticmethod
    def _unpack(payload: Any):
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"], payload.get("features")
        return payload, None

    # ------------------------------------------------------------------ #
    # Feature matrix
    # ------------------------------------------------------------------ #
    def training_columns(self) -> List[str]:
        """Column order recorded with the artifact, else derived from self.df."""
        if self.feature_columns:
            return list(self.feature_columns)
        if self.df.empty:
            raise RuntimeError("Engine has no dataframe and no saved feature columns.")
        X = self.df[FEATURES].copy()
        return list(pd.get_dummies(X, columns=CATEGORICALS, drop_first=True).columns)

    def _matrix(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Build the model matrix for arbitrary records.

        `drop_first=True` cannot be used on a handful of rows: it drops that
        row's only category and silently zeroes brand/category/price_segment.
        Expand every category instead, then reindex onto the training columns so
        the training baseline stays all-zeros and the rest get their learned
        indicators.
        """
        frame = pd.DataFrame(records)
        frame = pd.get_dummies(frame, columns=CATEGORICALS)
        frame = frame.reindex(columns=self.training_columns(), fill_value=0)
        return frame.fillna(0).astype(float)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Score one feature record through the chained models."""
        return self.predict_many([record])[0]

    def predict_many(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a batch, sharing one vectorised pass through both models."""
        if not records:
            return []

        X = self._matrix(records)
        classes = list(getattr(self.classifier, "classes_", [0, 1]))
        proba = self.classifier.predict_proba(X)
        positive = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(records))
        probabilities = [float(p) for p in positive]

        floors: Dict[int, float] = {}
        drop_idx = [i for i, p in enumerate(probabilities) if p >= DROP_THRESHOLD]
        if drop_idx and self.regressor is not None:
            raw = np.asarray(self.regressor.predict(X.iloc[drop_idx]), dtype=float)
            if self.target_mode == "floor_ratio":
                ratios = np.clip(raw, 0.05, 1.0)
                prices = np.array([
                    float(records[i].get("current_price") or 0.0) for i in drop_idx
                ])
                values = ratios * prices
            else:
                values = np.clip(raw, 0.0, None)
            for i, value in zip(drop_idx, values):
                floors[i] = round(float(value), 2)

        results: List[Dict[str, Any]] = []
        for i, (record, prob) in enumerate(zip(records, probabilities)):
            will_drop = prob >= DROP_THRESHOLD
            current_price = float(record.get("current_price") or 0.0)
            floor = floors.get(i) if will_drop else None
            expected_drop_pct = None
            if will_drop and floor is not None and current_price > 0:
                expected_drop_pct = round(max(0.0, (1.0 - floor / current_price)) * 100.0, 2)

            result: Dict[str, Any] = {
                "classification": "drop" if will_drop else "stable",
                "drop_probability": round(prob, 4),
                "classification_state": int(will_drop),
                "verdict": self._verdict(will_drop, prob),
                "status_badge": self._badge_text(will_drop, prob, expected_drop_pct),
                "predicted_price_floor": floor,
                "expected_drop_pct": expected_drop_pct,
                "current_price": round(current_price, 2),
                "flag": (
                    "Wait: High Probability of Drop" if will_drop
                    else "Buy Now: Stable Price"
                ),
            }
            results.append(result)

        return results

    # ------------------------------------------------------------------ #
    # Verdict text
    # ------------------------------------------------------------------ #
    @staticmethod
    def _verdict(will_drop: bool, prob: float) -> str:
        if will_drop:
            return "wait" if prob >= 0.65 else "lean_wait"
        return "watch" if prob >= WATCH_THRESHOLD else "buy"

    @staticmethod
    def _badge_text(will_drop: bool, prob: float, expected_drop_pct: float | None) -> str:
        """Human badge: [Buy Now: Stable Price] / [Wait: 71.2% Chance of a 9.4% Drop in 7 Days]."""
        pct = prob * 100.0
        if will_drop:
            if expected_drop_pct is not None:
                return f"Wait: {pct:.1f}% Chance of a {expected_drop_pct:.1f}% Drop in 7 Days"
            return f"Wait: {pct:.1f}% Probability of Drop"
        if prob >= WATCH_THRESHOLD:
            return f"Watch: {pct:.1f}% Drop Risk — Price Could Slip"
        return f"Buy Now: Stable Price ({pct:.1f}% Drop Risk)"
