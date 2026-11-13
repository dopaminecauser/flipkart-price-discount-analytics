"""
db/features.py — Phase 1.2 Feature Engineering Store

Reads raw rows from `price_history_logs` and enriches each product row with the
feature vector specified in the project brief:

  Temporal features
    - day_of_week          (0-6)
    - month                (1-12)
    - is_weekend           (0/1)
    - days_until_mega_sale (distance to known sale window, in days)

  Product dynamics
    - current_price
    - mrp
    - discount_percentage
    - rating_count_velocity (delta in rating_count over a rolling 7-day window)

  Target variables
    - will_drop_next_7_days   (1 if price will drop >= 5% in next 7 days, else 0)
    - predicted_price_floor   (continuous target for Stage-2 regression)

Ground-truth targets come from `ground_truth` (written by the synthetic
generator). In a real pipeline these would be computed from future prices;
here they are used directly so the end-to-end ML pipeline is verifiable in a lab.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Tuple

FEATURE_COLUMNS = [
    "product_id", "title", "brand", "category", "base_mrp", "product_url",
    "price_segment",
    "day_of_week", "month", "is_weekend", "days_until_mega_sale",
    "current_price", "mrp", "discount_percentage", "rating_count_velocity",
    "will_drop_next_7_days",    "predicted_price_floor",
]

MEGA_SALE_DATES = [
    "2026-10-04", "2026-12-01", "2027-01-15", "2027-10-20",
    "2027-11-25", "2028-01-05",
]


def _mega_sale_distance(as_of: datetime, ts: str | datetime) -> float:
    """Days between `ts` and the nearest upcoming mega-sale date."""
    if isinstance(ts, str):
        ts = datetime.strptime(ts, "%Y-%m-%d")
    if as_of >= ts:
        return 0.0
    upcoming = [d for d in MEGA_SALE_DATES if d >= ts.strftime("%Y-%m-%d")]
    if not upcoming:
        return 0.0
    nearest = min(upcoming, key=lambda d: d >= ts.strftime("%Y-%m-%d"))
    return (datetime.strptime(nearest, "%Y-%m-%d") - ts).days


def _rating_count_velocity(history: List[Dict[str, Any]]) -> float:
    """
    Delta in rating_count over the available rolling window.
    """
    if len(history) < 2:
        return 0.0
    sorted_rows = sorted(history, key=lambda r: r["timestamp"])
    oldest = sorted_rows[0]["rating_count"]
    newest = sorted_rows[-1]["rating_count"]
    span = (datetime.strptime(sorted_rows[-1]["timestamp"], "%Y-%m-%d")
            - datetime.strptime(sorted_rows[0]["timestamp"], "%Y-%m-%d")).days or 1
    return round((newest - oldest) / span, 4)


def _price_segment(base_mrp: float) -> str:
    """Assign a simple price segment from the base MRP."""
    if base_mrp < 10:
        return "budget"
    if base_mrp < 100:
        return "mid"
    if base_mrp < 1000:
        return "premium"
    return "luxury"


def compute_feature_vectors(conn: sqlite3.Connection, as_of: str | None = None) -> List[Dict[str, Any]]:
    """
    Build the current feature vector for every product in the DB.

    `as_of` accepts a 'YYYY-MM-DD' date (defaults to today) so the pipeline can
    be run deterministically for validation.
    """
    if as_of:
        as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    else:
        as_of_dt = datetime.utcnow()

    # Most recent price row per product (sqlite3 returns tuples unless Row is used).
    cur = conn.execute(
        """
        SELECT product_id, timestamp, current_price, discount_percentage,
               rating_count, stock_status
        FROM price_history_logs ph
        WHERE timestamp = (
            SELECT MAX(timestamp) FROM price_history_logs ph2
            WHERE ph2.product_id = ph.product_id
        )
        """
    )
    cols = ["product_id", "timestamp", "current_price", "discount_percentage", "rating_count", "stock_status"]
    latest = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}

    # Chronological history per product.
    hist = conn.execute(
        """
        SELECT product_id, timestamp, rating_count
        FROM price_history_logs
        ORDER BY product_id, timestamp
        """
    )
    history: Dict[int, List[Dict[str, Any]]] = {}
    for pid, ts, rating in hist.fetchall():
        history.setdefault(pid, []).append({"timestamp": ts, "rating_count": rating})

    # Map product_id -> (will_drop_next_7_days, predicted_price_floor)
    # from the ground_truth table (written by the synthetic generator).
    truth_map: Dict[int, Tuple[int, float]] = {}
    for pid, drop, floor in conn.execute(
        "SELECT product_id, will_drop_next_7_days, predicted_price_floor FROM ground_truth"
    ):
        truth_map[pid] = (drop, floor)

    rows: List[Dict[str, Any]] = []
    for pid, latest_row in latest.items():
        history_rows = history.get(pid, [])
        if not history_rows:
            continue

        # Pull metadata fields.
        meta = conn.execute(
            "SELECT product_id, title, brand, category, base_mrp, product_url "
            "FROM product_metadata WHERE product_id = ?",
            (pid,),
        ).fetchone()
        if not meta:
            continue

        doc = dict(latest_row)
        meta_cols = ["product_id", "title", "brand", "category", "base_mrp", "product_url"]
        meta = dict(zip(meta_cols, meta))
        row = {
            "product_id": pid,
            "title": meta["title"],
            "brand": meta["brand"],
            "category": meta["category"],
            "base_mrp": meta["base_mrp"],
            "product_url": meta["product_url"],
            "price_segment": _price_segment(meta["base_mrp"]),
            "day_of_week": as_of_dt.weekday(),
            "month": as_of_dt.month,
            "is_weekend": 1 if as_of_dt.weekday() >= 5 else 0,
            "days_until_mega_sale": _mega_sale_distance(as_of_dt, doc["timestamp"]),
            "current_price": doc["current_price"],
            "mrp": meta["base_mrp"],
            "discount_percentage": doc["discount_percentage"],
            "rating_count_velocity": _rating_count_velocity(history_rows),
            "will_drop_next_7_days": truth_map.get(pid, (0, 0.0))[0],
            "predicted_price_floor": truth_map.get(pid, (0, 0.0))[1],
        }
        rows.append(row)

    return rows


def write_feature_vectors(rows: List[Dict[str, Any]], conn: sqlite3.Connection) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO feature_engineering (
            product_id, title, brand, category, base_mrp, price_segment,
            day_of_week, month, is_weekend, days_until_mega_sale,
            current_price, mrp, discount_percentage, rating_count_velocity,
            will_drop_next_7_days, predicted_price_floor
        ) VALUES (
            :product_id, :title, :brand, :category, :base_mrp, :price_segment,
            :day_of_week, :month, :is_weekend, :days_until_mega_sale,
            :current_price, :mrp, :discount_percentage, :rating_count_velocity,
            :will_drop_next_7_days, :predicted_price_floor
        )
        """,
        rows,
    )
    conn.commit()


def load_feature_vectors(path: str = "data/flipkart.db") -> List[Dict[str, Any]]:
    """Convenience: connect, compute, write, and return the vectors."""
    conn = sqlite3.connect(path)
    vecs = compute_feature_vectors(conn)
    write_feature_vectors(vecs, conn)
    conn.close()
    return vecs


if __name__ == "__main__":
    vecs = load_feature_vectors()
    print(f"Feature vectors computed for {len(vecs)} products")
    if vecs:
        print("Sample:", vecs[0])
