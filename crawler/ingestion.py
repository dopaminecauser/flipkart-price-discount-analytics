"""
crawler/ingestion.py — Phase 2 Ingestion Engine

Entry point for the data ingestion layer.
  * On first run: builds the schema + inserts the synthetic catalogue.
  * Every subsequent run (local scheduler, or CI daily job): regenerates the
    catalogue and upsert the rows.

The generator is deterministic (fixed numpy seed), so a daily run reproduces
the same catalogue; historical rows accumulate so the series grows, which is
the only real change day-over-day (exactly like a live feed). No external
network, no browser, no API keys, no safety or terms-of-service concerns.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Allow this script to import db/ (parent of crawler/) regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.schema import init_db
from crawler import synthetic_data

DB_PATH = Path("data/flipkart.db")


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def build_or_update(args: argparse.Namespace) -> None:
    conn = init_db(args.db)
    existing_products = _count_rows(conn, "product_metadata")

    if existing_products == 0:
        print("First run: generating full synthetic catalogue...")
    else:
        print("Scheduler run: regenerating catalogue + appending new price rows...")

    catalog = synthetic_data.generate_catalogue()
    synthetic_data.insert_into_db(catalog, args.db)

    # Print a compact summary so the scheduler log is readable.
    conn = sqlite3.connect(args.db)
    current = _count_rows(conn, "price_history_logs")
    products = _count_rows(conn, "product_metadata")
    print(f"Ingestion complete: {products} products, {current} price rows "
          f"(as of {datetime.now().strftime('%Y-%m-%d %H:%M')})")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Flipkart analytics ingestion engine")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to the SQLite database")
    args = parser.parse_args()
    build_or_update(args)


if __name__ == "__main__":
    main()
