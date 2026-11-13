"""
crawler/synthetic_data.py — Phase 2.1 Synthetic Ingestion Layer

Generates a realistic, deterministic synthetic product catalogue and price
history log so the pipeline runs without any network access and without ever
touching a commercial site (no scraping safety concerns, no API blocks).

Behaviour
---------
- A fixed seed (NUM_SEEDS products, staggered launch dates) gives a diverse
  catalogue: 12 categories, 12 brands, 4 price segments.
- Products accumulate a real historical price series (110-180 daily
  observations, ~5 months).
- Price evolution uses a mean-reverting random walk (OU process) so floors
  naturally emerge and persist.
- Discounts are created by occasional downward price shocks, occasionally
  combined with an MRP lift so `discount_percentage` looks true.
- Rating counts are a logistic growth curve with a 7-day rolling noise term,
  giving a non-trivial `rating_count_velocity`.
- Each product is assigned a latent "sale pressure" score built from its own
  observable traits (current discount depth, how far the price already sits
  below MRP, category, segment, rating velocity). That score is mapped to a
  near-uniform propensity across the catalogue, and the 7-day label is sampled
  from it — so the classifier can actually learn the signal and verdicts
  genuinely differ product to product (about half the catalogue drops).

Output tables (via db/schema.py):: product_metadata, price_history_logs,
feature_engineering, plus `ground_truth` (labels + price-floor target).
"""

from __future__ import annotations

import sqlite3
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Allow this script to import db/ (parent of crawler/) regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.schema import SCHEMA  # noqa: E402

SEED = 42
NUM_SEEDS = 500          # distinct product "seeds"
DAYS_OF_HISTORY = 150    # historical observations per product (before launch offset)
PRICE_SHOCK_PCT = 0.04   # probability of a discount event on a given day
MRP_LIFT_PCT = 0.06      # probability the base MRP lifts before a shock
RANDOM_WALK_SIGMA = 0.03  # fraction of long-run price per step
MEAN_REVERSION = 0.02     # speed of reversion to the long-run price
DROP_WINDOW_DAYS = 7      # how far ahead we label a drop
DAILY_SALE_SHOCK = 0.20   # small price dip on mega-sale days

POWERS_OF_TEN = [1, 10, 100, 500, 1000, 5000, 15000, 40000, 80000]
CATEGORIES = [
    "Electronics", "Mobile & Accessories", "Home & Kitchen", "Apparel",
    "Books", "Sports & Fitness", "Beauty & Personal Care", "Toys",
    "Grocery", "Automotive", "Pet Supplies", "Office Supplies",
]

# How sale-prone each category is. Electronics and phones get slashed; books
# and groceries rarely see deep cuts. This is a stable input the classifier can
# learn from `category`.
CATEGORY_PRESSURE = {
    "Electronics": 0.92, "Mobile & Accessories": 0.98, "Home & Kitchen": 0.55,
    "Apparel": 0.80, "Books": 0.15, "Sports & Fitness": 0.42,
    "Beauty & Personal Care": 0.60, "Toys": 0.66, "Grocery": 0.18,
    "Automotive": 0.30, "Pet Supplies": 0.35, "Office Supplies": 0.25,
}

# Ranked sale pressure is mapped into this range, then sampled for the label.
# A near-uniform distribution gives genuine verdict diversity (and a learns-
# able, non-degenerate class balance instead of 95% "drop").
PROPENSITY_MIN = 0.05
PROPENSITY_MAX = 0.95

# A staggered 'launch day' keeps the history series non-identical.
LAUNCH_MIN = 0
LAUNCH_MAX = 30


def _seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _segment_level(base_mrp: float) -> float:
    """Normalised price segment (0 = budget .. 1 = luxury)."""
    if base_mrp < 10:
        return 0.0
    if base_mrp < 100:
        return 0.33
    if base_mrp < 1000:
        return 0.66
    return 1.0


class SyntheticProduct:
    """Holds the deterministic generative parameters for one product."""

    def __init__(self, seed: int, index: int, launch_day: int):
        self.seed = seed
        self.index = index
        self.launch_day = launch_day
        self._rng = _seed(seed * 1000 + index * 7 + launch_day)

        # Catalogue attributes (mirrors Product_Metadata)
        self.brand = self._rng.choice([
            "BrandX", "BrandY", "BrandZ", "StyleCo", "FlexTech", "PureLiving",
            "GoGreen", "MotionPro", "BrightHome", "SoundWave", "TechNova", "EcoCraft",
        ])
        self.category = self._rng.choice(CATEGORIES)
        self.product_title = f"{self.brand} {self.category} [Pack of {self._rng.integers(1, 6)}] {self.seed:04d}"
        self.base_mrp = float(self._rng.choice(POWERS_OF_TEN))
        self.product_url = (
            f"https://example.academic/flipkart/{self.brand.lower()}/"
            f"{self.category.lower().replace(' ', '-')}/{uuid.uuid4().hex[:8]}"
        )

        # Rating dynamics parameters
        self.mean_reviews = self._rng.uniform(120, 2200)
        self.growth_rate = self._rng.uniform(0.3, 0.9)  # reviews/day

        # Price model parameters
        self.long_run_price = float(self.base_mrp * self._rng.uniform(0.85, 1.05))
        self.noise_sigma = PRICE_SHOCK_PCT
        self.mega_sale_chance = 0.12
        self._long_run_price = self.long_run_price

    @property
    def _current_mrp(self) -> float:
        return self.base_mrp

    def simulate_history(self, days: int) -> List[Dict[str, Any]]:
        """Simulate a chronological price / discount / rating series."""
        days_from_launch = self.launch_day
        if days_from_launch > days:
            return []

        series = []
        ratings = self._rng.poisson(self.mean_reviews)

        price = self.long_run_price + self._rng.normal(0, self.noise_sigma * self.long_run_price)
        price = max(price, 0.0)

        for day in range(days_from_launch, days):
            # --- Price dynamics (OU-like mean reversion + shocks) ---
            reversion_step = MEAN_REVERSION * (self._long_run_price - price)
            shock = reversion_step + self._rng.normal(0, self.noise_sigma * self.long_run_price)
            if day % 30 < 2 and self._rng.random() < self.mega_sale_chance:
                shock -= self.long_run_price * DAILY_SALE_SHOCK
            if self._rng.random() < PRICE_SHOCK_PCT:
                shock -= self.long_run_price * self._rng.uniform(0.03, 0.18)
                if self._rng.random() < MRP_LIFT_PCT:
                    # A previously unlisted higher MRP is revealed: discount%
                    # rises without moving the current price.
                    self.base_mrp = self.base_mrp * self._rng.uniform(1.05, 1.12)
            price = max(price + shock, 0.01 * self.base_mrp)
            price = min(price, self.base_mrp * 2.0)

            # --- Discount percentage ---
            eff_mrp = max(self.base_mrp, 0.01 * self.base_mrp)
            discount_pct = max(0.0, min(0.95, (eff_mrp - price) / eff_mrp))

            # --- Ratings dynamics (logistic-ish growth + rolling noise) ---
            ratings = self._rng.poisson(self.mean_reviews + self.growth_rate * day * 2.0)
            rating_noise = self._rng.normal(0, max(1.0, ratings * 0.02))
            ratings = max(0, int(ratings + rating_noise))

            # --- Stock status (correlated with discount depth) ---
            if discount_pct > 0.45:
                stock = self._rng.choice(["out_of_stock", "low_stock"])
            elif discount_pct > 0.20:
                stock = self._rng.choice(["low_stock", "in_stock"])
            else:
                stock = self._rng.choice(["in_stock", "back_order"])

            series.append({
                "timestamp": (datetime(2026, 10, 25) + timedelta(days=day)).strftime("%Y-%m-%d"),
                "current_price": round(price, 2),
                "discount_percentage": round(discount_pct * 100.0, 2),
                "rating_count": ratings,
                "stock_status": stock,
            })

        return series

    def future_targets(self, history: List[Dict[str, Any]], propensity: float) -> Tuple[int, float, float]:
        """
        Ground-truth targets for the simulated next 7-day window.

        `propensity` is this product's latent chance of a >= 5% cut. When a cut
        happens, its depth scales with the discount the product already carries
        (deeply discounted products keep getting slashed further), so Stage 2
        has a learnable magnitude to fit.

        Returns (will_drop_next_7_days, price_floor, drop_magnitude).
        """
        if len(history) < 8:
            return 0, 0.0, 0.0

        last = float(history[-1]["current_price"])
        floor_floor = 0.01 * self.base_mrp

        # A sale is scheduled when a product's sale pressure crosses the line,
        # with a little operational jitter so the labels are not perfectly
        # deterministic from the features (real catalogues are noisy too).
        triggers = (propensity + self._rng.normal(0, 0.08)) >= 0.5

        if triggers:
            discount_now = float(history[-1]["discount_percentage"]) / 100.0
            magnitude = (
                0.05
                + 0.25 * discount_now
                + 0.05 * _segment_level(self.base_mrp)
                + self._rng.normal(0, 0.015)
            )
            magnitude = float(np.clip(magnitude, 0.05, 0.35))
            target = last * (1.0 - magnitude)
            # Only claim a drop the price range can actually deliver.
            if target >= floor_floor and target <= last * 0.95:
                return 1, round(target, 2), round(magnitude, 4)

        # No >= 5% cut in the window: the floor stays at the trailing low.
        recent = history[-DROP_WINDOW_DAYS:]
        return 0, round(min(r["current_price"] for r in recent), 2), 0.0


def _pressure_score(product: Dict[str, Any]) -> float:
    """Observable-trait score that drives how sale-prone a product is."""
    history = product["history"]
    last = history[-1]
    price = float(last["current_price"])
    base_mrp = float(product["base_mrp"]) or 1.0
    discount_now = float(last["discount_percentage"]) / 100.0
    price_ratio = min(price / base_mrp, 1.0)

    # Exactly the quantity db/features.py exposes as `rating_count_velocity`
    # (delta per day across the series), so the model can actually see it.
    span = max(len(history) - 1, 1)
    velocity = (history[-1]["rating_count"] - history[0]["rating_count"]) / span
    velocity_norm = float(np.clip(velocity / 1.5, 0.0, 1.0))

    return (
        2.0 * discount_now
        + 1.5 * (1.0 - price_ratio)
        + 0.9 * CATEGORY_PRESSURE.get(product["category"], 0.5)
        + 0.7 * _segment_level(base_mrp)
        + 0.6 * velocity_norm
    )


def generate_catalogue() -> List[Dict[str, Any]]:
    products: List[Dict[str, Any]] = []
    for seed in range(NUM_SEEDS):
        launch_day = int(_seed(seed).integers(LAUNCH_MIN, LAUNCH_MAX))
        prod = SyntheticProduct(seed, seed, launch_day)
        history = prod.simulate_history(DAYS_OF_HISTORY + 30)
        products.append({
            "seed": seed,
            "product_id": seed,  # deterministic ID from the seed
            "title": prod.product_title,
            "brand": prod.brand,
            "category": prod.category,
            "base_mrp": prod.base_mrp,
            "product_url": prod.product_url,
            "history": history,
            "_product": prod,
        })

    # Rank the catalogue by sale pressure and spread it across
    # [PROPENSITY_MIN, PROPENSITY_MAX] so every product has its own propensity.
    usable = [i for i, p in enumerate(products) if len(p["history"]) >= 8]
    ranked = sorted(usable, key=lambda i: _pressure_score(products[i]))
    span = max(len(ranked) - 1, 1)

    for rank, idx in enumerate(ranked):
        p = products[idx]
        propensity = PROPENSITY_MIN + (PROPENSITY_MAX - PROPENSITY_MIN) * (rank / span)
        will_drop, floor, magnitude = p["_product"].future_targets(p["history"], propensity)
        p["sale_pressure"] = round(propensity, 4)
        p["will_drop_next_7_days"] = will_drop
        p["predicted_price_floor"] = floor
        p["drop_magnitude"] = magnitude

    for p in products:
        p.pop("_product", None)
        p.setdefault("sale_pressure", 0.0)
        p.setdefault("will_drop_next_7_days", 0)
        p.setdefault("predicted_price_floor", 0.0)
        p.setdefault("drop_magnitude", 0.0)

    return products


def insert_into_db(products: List[Dict[str, Any]], db_path: str = "data/flipkart.db") -> None:
    conn = sqlite3.connect(db_path)
    # Ensure all tables exist before deletion (idempotent re-runs).
    conn.executescript(SCHEMA)
    conn.executescript("""
        DELETE FROM feature_engineering;
        DELETE FROM price_history_logs;
        DELETE FROM product_metadata;
    """)

    meta_rows = [
        (p["product_id"], p["title"], p["brand"], p["category"], p["base_mrp"], p["product_url"])
        for p in products
    ]
    conn.executemany(
        "INSERT INTO product_metadata (product_id, title, brand, category, base_mrp, product_url) "
        "VALUES (?, ?, ?, ?, ?, ?)", meta_rows
    )

    log_rows = []
    for p in products:
        pid = p["product_id"]
        for h in p["history"]:
            log_rows.append((pid, h["timestamp"], h["current_price"],
                             h["discount_percentage"], h["rating_count"], h["stock_status"]))
    conn.executemany(
        "INSERT INTO price_history_logs (product_id, timestamp, current_price, "
        "discount_percentage, rating_count, stock_status) VALUES (?,?,?,?,?,?)",
        log_rows,
    )
    conn.commit()

    # Persist the ground-truth labels and the Stage-2 regression target.
    truth = [
        (p["product_id"], p["will_drop_next_7_days"], p["predicted_price_floor"])
        for p in products
    ]
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ground_truth ("
        "product_id INTEGER PRIMARY KEY, will_drop_next_7_days INTEGER, predicted_price_floor REAL)"
    )
    conn.execute("DELETE FROM ground_truth")
    conn.executemany("INSERT OR REPLACE INTO ground_truth VALUES (?,?,?)", truth)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    print("Generating synthetic catalogue...")
    catalog = generate_catalogue()
    print(f"  Products: {len(catalog)}")
    inserts = sum(len(p["history"]) for p in catalog)
    print(f"  Price-history rows: {inserts}")
    drops = sum(int(p["will_drop_next_7_days"]) for p in catalog)
    stable = len(catalog) - drops
    print(f"  Class balance: {drops} drop / {stable} stable "
          f"({drops / max(len(catalog), 1) * 100:.1f}% positive)")
    buckets = {"<20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, ">=80%": 0}
    for p in catalog:
        s = p["sale_pressure"]
        if s < 0.2:
            buckets["<20%"] += 1
        elif s < 0.4:
            buckets["20-40%"] += 1
        elif s < 0.6:
            buckets["40-60%"] += 1
        elif s < 0.8:
            buckets["60-80%"] += 1
        else:
            buckets[">=80%"] += 1
    print("  Sale-pressure spread:", buckets)
    insert_into_db(catalog)
    print("Inserted into sqlite://data/flipkart.db")
