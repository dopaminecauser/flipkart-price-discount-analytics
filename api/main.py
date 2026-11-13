"""
api/main.py — Phase 4 FastAPI Backend

Endpoints:
  GET /health              -> service health
  GET /predict?url=...     -> two-stage ML prediction for a Flipkart product URL
  GET /api/products        -> product summaries for the dashboard search box
  GET /api/products/{id}   -> full feature vector + history + prediction
  GET /api/overview        -> catalogue-wide verdict distribution (market pulse)

The /predict endpoint:
  1. Resolves the Flipkart URL to a product_id
  2. Extracts the current feature vector from SQLite
  3. Feeds it into the saved Stage-1 classifier + Stage-2 regressor
  4. Returns structured JSON: classification state, predicted price floor, probabilities
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from db.features import compute_feature_vectors, write_feature_vectors
from db.schema import init_db
from models.engine import TRAINED_DIR, TwoStageMLEngine

app = FastAPI(title="Flipkart Price & Discount Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded ML engine (built on first request that needs it).
engine: Optional[TwoStageMLEngine] = None

# SQLite connections are thread-bound: FastAPI runs sync endpoints in a
# threadpool, so each worker thread gets its own connection.
_local = threading.local()

# Catalogue-wide inference is expensive relative to a single prediction; cache
# it briefly so the dashboard's "market pulse" panel does not re-score 500
# products on every page load.
_OVERVIEW_TTL_SECONDS = 60.0
_overview_cache: Dict[str, Any] = {"at": 0.0, "data": None}


def _get_conn() -> sqlite3.Connection:
    """Return this thread's SQLite connection, creating it on demand."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = init_db()
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def get_engine() -> TwoStageMLEngine:
    """Load the saved model artifacts once and reuse them."""
    global engine
    if engine is None:
        clf_path = TRAINED_DIR / "classifier.pkl"
        reg_path = TRAINED_DIR / "regressor.pkl"
        if not clf_path.exists() or not reg_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Trained models not found. Run: python models/train_models.py",
            )
        engine = TwoStageMLEngine().load(clf_path, reg_path)
    return engine


def _price_segment_from_mrp(base_mrp: float) -> str:
    """Assign a price segment from the base MRP (mirrors db/features.py)."""
    if base_mrp < 10:
        return "budget"
    if base_mrp < 100:
        return "mid"
    if base_mrp < 1000:
        return "premium"
    return "luxury"


def sqlite3_date_to_dt(value: str) -> datetime:
    """Convert a SQLite 'YYYY-MM-DD' string to a datetime object."""
    return datetime.strptime(str(value), "%Y-%m-%d")


def _days_until_mega_sale(latest_ts: datetime) -> float:
    """Days until the next known mega-sale (0 if a sale is already in progress)."""
    for sale_date in ["2026-12-01", "2027-01-15", "2027-10-20", "2027-11-25", "2028-01-05"]:
        sale_dt = datetime.strptime(sale_date, "%Y-%m-%d")
        if sale_dt >= latest_ts:
            return (sale_dt - latest_ts).days
    return 0.0


def _feature_record(conn: sqlite3.Connection, product_id: int) -> Dict[str, Any]:
    """Build the current feature vector for one product, with metadata merged in."""
    meta = conn.execute(
        "SELECT title, brand, category, base_mrp, product_url "
        "FROM product_metadata WHERE product_id = ?",
        (product_id,),
    ).fetchone()
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Metadata missing for product_id={product_id}")

    latest = conn.execute(
        "SELECT timestamp, current_price, discount_percentage, rating_count "
        "FROM price_history_logs WHERE product_id = ? ORDER BY timestamp DESC LIMIT 1",
        (product_id,),
    ).fetchone()
    if latest is None:
        raise HTTPException(status_code=404, detail=f"No price history for product_id={product_id}")

    ratings = [
        r["rating_count"]
        for r in conn.execute(
            "SELECT rating_count FROM price_history_logs WHERE product_id = ? ORDER BY timestamp",
            (product_id,),
        ).fetchall()
    ]
    rating_count_velocity = (
        round((ratings[-1] - ratings[0]) / max(len(ratings) - 1, 1), 4)
        if len(ratings) >= 2
        else 0.0
    )

    base_mrp = float(meta["base_mrp"])
    latest_ts = sqlite3_date_to_dt(latest["timestamp"])

    return {
        "product_id": product_id,
        "title": meta["title"],
        "brand": meta["brand"],
        "category": meta["category"],
        "product_url": meta["product_url"],
        "base_mrp": base_mrp,
        "price_segment": _price_segment_from_mrp(base_mrp),
        "day_of_week": latest_ts.weekday(),
        "month": latest_ts.month,
        "is_weekend": 1 if latest_ts.weekday() >= 5 else 0,
        "days_until_mega_sale": _days_until_mega_sale(latest_ts),
        "current_price": float(latest["current_price"]),
        "mrp": base_mrp,
        "discount_percentage": float(latest["discount_percentage"]),
        "rating_count": int(latest["rating_count"]),
        "rating_count_velocity": rating_count_velocity,
        "will_drop_next_7_days": _get_ground_truth(conn, product_id)[0],
        "predicted_price_floor": _get_ground_truth(conn, product_id)[1],
    }


# ---------------------------------------------------------------------- #
# Health
# ---------------------------------------------------------------------- #
@app.get("/health")
def health():
    conn = _get_conn()
    products = conn.execute("SELECT COUNT(*) FROM product_metadata").fetchone()[0]
    history = conn.execute("SELECT COUNT(*) FROM price_history_logs").fetchone()[0]
    models_ready = (TRAINED_DIR / "classifier.pkl").exists() and (TRAINED_DIR / "regressor.pkl").exists()
    return {
        "status": "ok",
        "message": "Flipkart Price & Discount Analytics API is running",
        "products": products,
        "price_history_rows": history,
        "models_ready": models_ready,
    }


# ---------------------------------------------------------------------- #
# Prediction
# ---------------------------------------------------------------------- #
@app.get("/predict")
def predict(url: str = Query(..., description="Flipkart product URL")) -> Dict[str, Any]:
    """Resolve a Flipkart URL to product_id, run the two-stage engine, return a prediction."""
    conn = _get_conn()
    ml = get_engine()

    product_id = resolve_url_to_id(conn, url)
    if product_id is None:
        raise HTTPException(status_code=404, detail=f"No product found for URL: {url}")

    record = _feature_record(conn, product_id)
    result = ml.predict(record)

    history = conn.execute(
        "SELECT timestamp, current_price, discount_percentage, rating_count "
        "FROM price_history_logs WHERE product_id = ? ORDER BY timestamp",
        (product_id,),
    ).fetchall()

    return {
        **result,
        "product_id": product_id,
        "product_title": record["title"],
        "product_brand": record["brand"],
        "product_category": record["category"],
        "product_url": record["product_url"],
        "base_mrp": record["base_mrp"],
        "price_segment": record["price_segment"],
        "discount_percentage": record["discount_percentage"],
        "rating_count": record["rating_count"],
        "rating_count_velocity": record["rating_count_velocity"],
        "days_until_mega_sale": record["days_until_mega_sale"],
        "history": [
            {
                "timestamp": h["timestamp"],
                "price": h["current_price"],
                "discount": h["discount_percentage"],
                "ratings": h["rating_count"],
            }
            for h in history
        ],
    }


def _get_ground_truth(conn: sqlite3.Connection, product_id: int) -> Tuple[int, float]:
    """Return (will_drop_next_7_days, predicted_price_floor) from ground_truth."""
    row = conn.execute(
        "SELECT will_drop_next_7_days, predicted_price_floor FROM ground_truth WHERE product_id = ?",
        (product_id,),
    ).fetchone()
    if row is not None:
        return int(row["will_drop_next_7_days"]), float(row["predicted_price_floor"])
    return 0, 0.0


def resolve_url_to_id(conn: sqlite3.Connection, url: str) -> Optional[int]:
    """Match a Flipkart URL to a product_id. Falls back to a partial path match."""
    url = url.strip()
    row = conn.execute(
        "SELECT product_id FROM product_metadata WHERE product_url = ?", (url,)
    ).fetchone()
    if row:
        return row["product_id"]

    from urllib.parse import unquote, urlparse

    try:
        path = unquote(urlparse(url).path.rstrip("/").split("/")[-1])
    except Exception:
        path = None
    if path:
        row = conn.execute(
            "SELECT product_id FROM product_metadata WHERE product_url LIKE ?",
            (f"%{path}%",),
        ).fetchone()
        if row:
            return row["product_id"]
    return None


# ---------------------------------------------------------------------- #
# Dashboard data endpoints
# ---------------------------------------------------------------------- #
@app.get("/api/products")
def list_products(
    limit: int = Query(20, ge=1, le=500),
    q: Optional[str] = Query(None, description="Search across title, brand and category"),
):
    """Product summaries for the dashboard search box (optionally filtered)."""
    conn = _get_conn()
    sql = (
        "SELECT m.product_id, m.title, m.brand, m.category, m.base_mrp, m.product_url, "
        "       p.current_price, p.discount_percentage "
        "FROM product_metadata m "
        "LEFT JOIN price_history_logs p ON p.product_id = m.product_id "
        "  AND p.timestamp = (SELECT MAX(timestamp) FROM price_history_logs "
        "                     WHERE product_id = m.product_id) "
    )
    params: List[Any] = []
    if q:
        sql += "WHERE m.title LIKE ? OR m.brand LIKE ? OR m.category LIKE ? "
        like = f"%{q.strip()}%"
        params = [like, like, like]
    sql += "ORDER BY m.product_id LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "product_id": r["product_id"],
            "title": r["title"],
            "brand": r["brand"],
            "category": r["category"],
            "base_mrp": r["base_mrp"],
            "product_url": r["product_url"],
            "current_price": r["current_price"],
            "discount_percentage": r["discount_percentage"],
        }
        for r in rows
    ]


@app.get("/api/overview")
def overview(force: bool = Query(False, description="Bypass the short-lived cache")):
    """Catalogue-wide verdict distribution powering the dashboard's market pulse."""
    now = time.time()
    if not force and _overview_cache["data"] and now - _overview_cache["at"] < _OVERVIEW_TTL_SECONDS:
        return _overview_cache["data"]

    conn = _get_conn()
    ml = get_engine()
    records = compute_feature_vectors(conn)
    if not records:
        raise HTTPException(status_code=503, detail="No feature rows available")

    # Keep the on-disk feature store in sync with what the models just scored.
    write_feature_vectors(records, conn)

    predictions = ml.predict_many(records)

    total = len(records)
    dropping = sum(1 for p in predictions if p["classification"] == "drop")
    # Genuinely uncertain middle band (0.30-0.65), not a second count of drops.
    watch_band = sum(1 for p in predictions if 0.30 <= p["drop_probability"] < 0.65)
    buy = total - dropping
    floors = [p for p in predictions if p["predicted_price_floor"] is not None]
    avg_expected_drop = (
        sum(p["expected_drop_pct"] or 0.0 for p in floors) / len(floors) if floors else 0.0
    )

    by_category: Dict[str, Dict[str, Any]] = {}
    for record, pred in zip(records, predictions):
        bucket = by_category.setdefault(
            record["category"],
            {"category": record["category"], "count": 0, "dropping": 0,
             "discount_sum": 0.0, "prob_sum": 0.0},
        )
        bucket["count"] += 1
        bucket["dropping"] += 1 if pred["classification"] == "drop" else 0
        bucket["discount_sum"] += float(record["discount_percentage"])
        bucket["prob_sum"] += pred["drop_probability"]

    categories = [
        {
            "category": b["category"],
            "count": b["count"],
            "dropping": b["dropping"],
            "drop_share": round(b["dropping"] / b["count"] * 100.0, 1),
            "avg_discount": round(b["discount_sum"] / b["count"], 1),
            "avg_drop_probability": round(b["prob_sum"] / b["count"], 4),
        }
        for b in by_category.values()
    ]
    categories.sort(key=lambda c: c["drop_share"], reverse=True)

    def brief(record: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "product_id": record["product_id"],
            "title": record["title"],
            "brand": record["brand"],
            "category": record["category"],
            "product_url": record["product_url"],
            "current_price": round(float(record["current_price"]), 2),
            "discount_percentage": round(float(record["discount_percentage"]), 1),
            "drop_probability": pred["drop_probability"],
            "predicted_price_floor": pred["predicted_price_floor"],
            "expected_drop_pct": pred["expected_drop_pct"],
            "verdict": pred["verdict"],
        }

    drops = [(r, p) for r, p in zip(records, predictions) if p["classification"] == "drop"]
    buys = [(r, p) for r, p in zip(records, predictions) if p["classification"] != "drop"]
    # Deepest predicted cuts to wait for, and the best deals available today.
    # (Sorting purely by probability returns a wall of 1.000 / 0.000 rows.)
    top_watch = sorted(drops, key=lambda t: t[1]["expected_drop_pct"] or 0.0, reverse=True)[:4]
    # Biggest absolute saving vs MRP — sorting by discount % surfaces sub-rupee
    # products whose percentage is large but whose saving is a few paise.
    top_buy = sorted(
        buys,
        key=lambda t: float(t[0]["base_mrp"]) - float(t[0]["current_price"]),
        reverse=True,
    )[:4]

    payload = {
        "total_products": total,
        "dropping": dropping,
        "stable": buy,
        "watch_band": watch_band,
        "avg_discount": round(
            sum(float(r["discount_percentage"]) for r in records) / total, 1
        ),
        "avg_drop_probability": round(
            sum(p["drop_probability"] for p in predictions) / total, 4
        ),
        "avg_expected_drop_pct": round(avg_expected_drop, 2),
        "by_category": categories,
        "top_watch": [brief(r, p) for r, p in top_watch],
        "top_buy": [brief(r, p) for r, p in top_buy],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    _overview_cache["data"] = payload
    _overview_cache["at"] = now
    return payload


@app.get("/api/products/{product_id}")
def get_product(product_id: int):
    """Full feature vector + history + prediction for one product."""
    conn = _get_conn()
    ml = get_engine()

    record = _feature_record(conn, product_id)
    prediction = ml.predict(record)
    history = conn.execute(
        "SELECT timestamp, current_price, discount_percentage, rating_count "
        "FROM price_history_logs WHERE product_id = ? ORDER BY timestamp",
        (product_id,),
    ).fetchall()

    return {
        "product_id": product_id,
        "title": record["title"],
        "brand": record["brand"],
        "category": record["category"],
        "base_mrp": record["base_mrp"],
        "product_url": record["product_url"],
        "latest_price": record["current_price"],
        "latest_discount": record["discount_percentage"],
        "price_segment": record["price_segment"],
        "prediction": prediction,
        "history": [
            {
                "timestamp": h["timestamp"],
                "price": h["current_price"],
                "discount": h["discount_percentage"],
                "ratings": h["rating_count"],
            }
            for h in history
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
