"""
db/schema.py — Phase 1.1 Storage Layer Schema (SQLite)

Relational schema for the Flipkart Price & Discount Analytics Tool.
Designed for an academic college-lab project. SQLite is used locally so the
pipeline runs without any external service, while the table/column names map
directly to the requested PostgreSQL model.
"""

import sqlite3
from pathlib import Path

SCHEMA = """
-- Product metadata registry (Product_Metadata)
CREATE TABLE IF NOT EXISTS product_metadata (
    product_id        INTEGER PRIMARY KEY,  -- reset on DELETE for idempotent re-runs

    title             TEXT NOT NULL,
    brand             TEXT NOT NULL DEFAULT 'Unknown',
    category          TEXT NOT NULL DEFAULT 'Uncategorized',
    base_mrp          REAL NOT NULL CHECK (base_mrp >= 0),
    product_url       TEXT NOT NULL UNIQUE
);

-- Historical price / listing logs (Price_History_Logs)
CREATE TABLE IF NOT EXISTS price_history_logs (
    log_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id       INTEGER NOT NULL REFERENCES product_metadata(product_id) ON DELETE CASCADE,
    timestamp        TEXT NOT NULL,
    current_price    REAL NOT NULL CHECK (current_price >= 0),
    discount_percentage REAL NOT NULL CHECK (discount_percentage >= 0),
    rating_count     INTEGER NOT NULL DEFAULT 0 CHECK (rating_count >= 0),
    stock_status     TEXT NOT NULL DEFAULT 'in_stock' CHECK (stock_status IN ('in_stock','out_of_stock','low_stock','back_order'))
);

-- Feature store: current vectors for each product (Feature_Engineering)
CREATE TABLE IF NOT EXISTS feature_engineering (
    product_id             INTEGER PRIMARY KEY REFERENCES product_metadata(product_id) ON DELETE CASCADE,
    title                  TEXT NOT NULL,
    brand                  TEXT NOT NULL,
    category               TEXT NOT NULL,
    base_mrp               REAL NOT NULL,

    -- Temporal features
    day_of_week            INTEGER NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    month                  INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    is_weekend             INTEGER NOT NULL CHECK (is_weekend IN (0, 1)),
    days_until_mega_sale   REAL NOT NULL DEFAULT 0,   -- 0 = not in a known sale window

    -- Price segmentation for model-level features
    price_segment          TEXT NOT NULL DEFAULT 'mid',   -- budget / mid / premium / luxury

    -- Product dynamics
    current_price          REAL NOT NULL,
    mrp                    REAL NOT NULL,
    discount_percentage    REAL NOT NULL,
    rating_count_velocity  REAL NOT NULL DEFAULT 0.0,  -- +/-(reviews/minute) over last 7 days

    -- Target variables
    will_drop_next_7_days  INTEGER NOT NULL CHECK (will_drop_next_7_days IN (0, 1)),
    predicted_price_floor  REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_price_history_product ON price_history_logs(product_id, timestamp);
"""

def init_db(db_path: str = "data/flipkart.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn

if __name__ == "__main__":
    conn = init_db()
    print(f"Schema ready at {conn}")
    conn.close()
