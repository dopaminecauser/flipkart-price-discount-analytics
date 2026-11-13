# Flipkart Price & Discount Analytics Tool

A modular, academic-grade analytics pipeline for studying Flipkart price
dynamics, discount behaviour, and price-floor prediction. The project is built
in four phases (Phase 1: database schema + feature engineering, Phase 2:
synthetic data ingestion, Phase 3: two-stage ML engine, Phase 4: FastAPI +
Next.js dashboard).

> **Important (academic integrity + safety):** Flipkart does not publish an
> academic-friendly product API and its Terms of Service prohibit automated
> access, so this project ships with a **deterministic synthetic data generator**
> that produces realistic price fluctuations, discount dips, and rolling review
> counts. This lets the entire pipeline run offline with zero legal, safety, or
> terms-of-service concerns. The ingestion layer (`crawler/ingestion.py`) is
> structured so a real licensed dataset or public API can be plugged in later.

---

## Project Structure

```
flipkart-analytics/
├── crawler/          # Ingestion & synthetic data
│   ├── synthetic_data.py   # Deterministic synthetic generator
│   ├── ingestion.py        # Ingestion engine entry point
│   ├── scheduler_windows.ps1  # Windows Task Scheduler script
│   └── __init__.py
├── db/               # Storage & feature engineering
│   ├── schema.py         # SQLite relational schema
│   ├── features.py       # Feature engineering store
│   └── __init__.py
├── models/           # Two-stage ML engine
│   ├── engine.py         # Inference-only chained engine
│   ├── train_models.py   # Single training entry point (walk-forward CV)
│   ├── trained/
│   │   ├── classifier.pkl   # Stage-1 model + feature manifest
│   │   └── regressor.pkl    # Stage-2 model + target mode
│   └── __init__.py
├── api/              # FastAPI backend
│   ├── main.py           # FastAPI app + /predict endpoint
│   └── __init__.py
├── app/              # Next.js/React frontend
│   ├── pages/            # index.js dashboard, _app.js
│   ├── components/       # PriceChart, MarketPulse
│   ├── lib/format.js     # Number/date/verdict formatting
│   ├── styles/globals.css
│   └── package.json
├── data/             # SQLite database (created at runtime)
│   └── flipkart.db
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data + build the schema (Phase 2 + 1)

```bash
python crawler/synthetic_data.py
python crawler/ingestion.py
```

This creates `data/flipkart.db` with 500 products and ~82,000 price-history rows.

### 3. Compute feature vectors (Phase 1.2)

```bash
python -c "import db.features as F; F.load_feature_vectors()"
```

### 4. Train the two-stage ML engine (Phase 3)

```bash
python models/train_models.py
```

This walk-forward validates both stages, refits each on the full dataset and writes
`models/trained/classifier.pkl` and `models/trained/regressor.pkl`. Re-run it whenever
the data changes — the API caches loaded models in memory, so restart uvicorn after
training to pick up new binaries.

### 5. Run the FastAPI backend (Phase 4)

```bash
uvicorn api.main:app --reload
```

### 6. Run the Next.js dashboard (Phase 4)

```bash
cd app
npm install
npm run dev
```

---

## Phase 1 — Database Schema & Feature Architecture

### Storage Layer Schema (SQLite ↔ PostgreSQL mapping)

| Table | Columns | Notes |
|---|---|---|
| `product_metadata` | `product_id PK`, `title`, `brand`, `category`, `base_mrp`, `product_url` | Mirrors `Product_Metadata` |
| `price_history_logs` | `log_id PK`, `product_id FK`, `timestamp`, `current_price`, `discount_percentage`, `rating_count`, `stock_status` | Mirrors `Price_History_Logs` |
| `feature_engineering` | All engineered features + 2 targets | Mirrors `Feature_Engineering` |

### Feature Vector Formula

| Category | Features |
|---|---|
| Temporal | `day_of_week` (0-6), `month` (1-12), `is_weekend` (0/1), `days_until_mega_sale` |
| Product Dynamics | `current_price`, `mrp`, `discount_percentage`, `rating_count_velocity` (Δ reviews / 7-day rolling window) |
| Target A (Classification) | `will_drop_next_7_days` (1 if price drops ≥ 5% in next 7 days) |
| Target B (Regression) | `predicted_price_floor` (continuous target for Stage-2) |

---

## Phase 2 — Data Pipeline & Processing Layer

### Synthetic Ingestion Engine

The `crawler/synthetic_data.py` generator is **fully deterministic** (fixed numpy
seed) and produces:

- A catalogue of 500 products across 12 brands and 10 categories
- Historical price series (110–200 daily observations per product) using a
  mean-reverting random walk (OU process) so price floors naturally emerge
- Discount events (3–18% downward shocks) with occasional MRPlift reveals
- Rating counts growing via a logistic curve with rolling noise → `rating_count_velocity`
- A hidden 7-day "drop window" injected for a subset of products so
  `will_drop_next_7_days` and `predicted_price_floor` are learnable

### Scheduling (Windows)

The native equivalent of a local Cron job on Windows is **Task Scheduler**.
`crawler/scheduler_windows.ps1` holds the registration command:

```powershell
$action = New-ScheduledTaskAction -Execute "python.exe" -Argument "-u crawler/ingestion.py" -WorkingDirectory "C:\path\to\project"
$trigger = New-ScheduledTaskTrigger -Daily -At 23:30
Register-ScheduledTask -TaskName "FlipkartAnalytics-Ingestion" -Action $action -Trigger $trigger -Description "Daily Flipkart analytics ingestion"
```

Run it once to register the job, then verify:

```powershell
Get-ScheduledTask -TaskName "FlipkartAnalytics-Ingestion" | Format-List
```

---

## Phase 3 — Two-Stage ML Engine

```
[Input Feature Vector] ──► [Stage 1: XGBoost Classifier] ──► Class 0 (Stable) ──► Stop
                          │                                        │
                          │ Class 1 (Drop)                         │
                          └──► [Stage 2: CatBoost Regressor] ──► Predicted Price Floor
```

- **Stage 1:** `XGBClassifier` — `mean F1 = 0.869`, `mean Precision = 0.873` (class balance 50/50)
- **Stage 2:** `CatBoostRegressor` on the **floor ratio** (`predicted_floor / current_price`) — `mean MAE ≈ 168 INR`, `mean RMSE ≈ 316 INR`
- **Validation:** `TimeSeriesSplit(n_splits=4)` walk-forward — **no random splits**, so no temporal leakage

Stage 2 regresses a ratio rather than raw rupees: the catalogue spans ₹1 to ₹80,000, so a
rupee loss is dominated by a handful of expensive items and can predict a floor above the
current price for cheap ones. The API multiplies the ratio back by the current price, which
keeps every floor below the price it was predicted from.

---

## Phase 4 — Backend API & Frontend

### FastAPI Backend

| Endpoint | Purpose |
|---|---|
| `GET /health` | service + dataset status |
| `GET /predict?url=...` | two-stage prediction for one product URL |
| `GET /api/products?q=&limit=` | searchable product list (dashboard autocomplete) |
| `GET /api/products/{id}` | full feature vector, history and prediction |
| `GET /api/overview` | catalogue-wide verdict distribution (cached 60s) |

`GET /predict?url=[FLIPKART_URL]` returns:

```json
{
  "classification": "drop",
  "drop_probability": 0.9997,
  "classification_state": 1,
  "verdict": "wait",
  "status_badge": "Wait: 100.0% Chance of a 34.6% Drop in 7 Days",
  "predicted_price_floor": 1111.0,
  "expected_drop_pct": 34.6,
  "current_price": 1700.0,
  "flag": "Wait: High Probability of Drop"
}
```

- Resolves the URL to a `product_id` (exact match, then partial path match)
- Extracts the current feature vector from SQLite
- Feeds the vector into the saved `.pkl` models
- Returns structured JSON with classification state, predicted floor, and probabilities

### Next.js/React Frontend

- Live API status pill, with an inline hint when the backend is unreachable
- Search accepts a product URL **or** a name, with debounced autocomplete and keyboard nav
- Verdict banner with a drop-probability meter and a plain-language badge
- KPI tiles: current price, MRP, discount, predicted floor, expected cut, ratings
- Interactive history chart: hover crosshair + tooltip, price/discount toggle, 30d/90d/all range, predicted-floor line
- Model trace showing the Stage 1 → Stage 2 chain and which stages actually ran
- Market pulse: verdict split, per-category cut share, deepest predicted cuts and best current deals (clickable)
- Responsive down to phone widths

---

## Development Requirements

- **PEP 8** compliant Python
- Modular directories: `crawler/`, `db/`, `models/`, `api/`, `app/`
- `requirements.txt` for environment reproduction
- `README.md` with local setup steps

---

## Notes

- **No external network access required.** The pipeline is fully offline once
  the dependencies are installed. The synthetic generator is deterministic, so
  re-runs produce identical catalogues.
- **To swap in real data:** replace `crawler/synthetic_data.py` with a reader
  against your licensed source or public API, and the rest of the pipeline
  (`ingestion.py`, `features.py`, `engine.py`) already consumes the standard
  SQLite schema.
