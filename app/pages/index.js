/**
 * app/pages/index.js — Phase 4 Frontend
 *
 * Next.js dashboard for the Flipkart Price & Discount Analytics Tool.
 *
 * Sections:
 *   - Live API status + headline search with autocomplete
 *   - Catalogue stat strip (from /api/overview)
 *   - Prediction result: verdict badge, KPI tiles, pipeline trace
 *   - Interactive historical price / discount chart
 *   - Market pulse: verdict split, category cut share, live picks
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Head from "next/head";

import MarketPulse from "../components/MarketPulse";
import PriceChart from "../components/PriceChart";
import {
  formatINR,
  formatPercent,
  verdictTheme,
} from "../lib/format";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Index() {
  const [health, setHealth] = useState(null);
  const [overview, setOverview] = useState(null);
  const [apiDown, setApiDown] = useState(false);

  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [suggestOpen, setSuggestOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const shellRef = useRef(null);

  /* ---------------------------------------------------------------- */
  /* Bootstrap: status + catalogue overview                            */
  /* ---------------------------------------------------------------- */
  const loadOverview = useCallback(async () => {
    try {
      const [healthRes, overviewRes] = await Promise.all([
        fetch(`${API}/health`),
        fetch(`${API}/api/overview`),
      ]);
      if (!healthRes.ok || !overviewRes.ok) throw new Error("backend error");
      setHealth(await healthRes.json());
      setOverview(await overviewRes.json());
      setApiDown(false);
    } catch {
      setApiDown(true);
    }
  }, []);

  useEffect(() => {
    loadOverview();
  }, [loadOverview]);

  /* ---------------------------------------------------------------- */
  /* Autocomplete (debounced)                                          */
  /* ---------------------------------------------------------------- */
  useEffect(() => {
    const term = query.trim();
    if (term.length < 2 || /^https?:\/\//i.test(term)) {
      setSuggestions([]);
      return undefined;
    }
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(
          `${API}/api/products?q=${encodeURIComponent(term)}&limit=6`,
          { signal: controller.signal }
        );
        if (!res.ok) return;
        setSuggestions(await res.json());
        setActiveIndex(-1);
      } catch {
        /* aborted or offline — the status pill already says so */
      }
    }, 180);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  /* Close the dropdown when clicking elsewhere */
  useEffect(() => {
    const onClick = (event) => {
      if (shellRef.current && !shellRef.current.contains(event.target)) {
        setSuggestOpen(false);
      }
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  /* ---------------------------------------------------------------- */
  /* Prediction                                                        */
  /* ---------------------------------------------------------------- */
  const analyze = useCallback(async (target) => {
    setLoading(true);
    setError("");
    setResult(null);
    setSuggestOpen(false);
    try {
      const resp = await fetch(`${API}/predict?url=${encodeURIComponent(target)}`);
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `Prediction failed (${resp.status})`);
      setResult(data);
      setApiDown(false);
    } catch (err) {
      setError(
        err.message ||
          `Could not reach the API at ${API}. Make sure the backend is running.`
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const resolveTarget = async (term) => {
    const value = term.trim();
    if (!value) return null;
    if (/^https?:\/\//i.test(value)) return value;

    const res = await fetch(`${API}/api/products?q=${encodeURIComponent(value)}&limit=1`);
    const matches = res.ok ? await res.json() : [];
    if (!matches.length) {
      throw new Error(
        `No product matched “${value}”. Paste a product URL or pick one from the list.`
      );
    }
    return matches[0].product_url;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim() || loading) return;
    try {
      const target = await resolveTarget(query);
      if (target) await analyze(target);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleKeyDown = (event) => {
    if (!suggestOpen || suggestions.length === 0) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((i) => (i + 1) % suggestions.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((i) => (i <= 0 ? suggestions.length - 1 : i - 1));
    } else if (event.key === "Enter" && activeIndex >= 0) {
      event.preventDefault();
      pickSuggestion(suggestions[activeIndex]);
    } else if (event.key === "Escape") {
      setSuggestOpen(false);
    }
  };

  const pickSuggestion = (product) => {
    setQuery(product.title);
    setSuggestOpen(false);
    analyze(product.product_url);
  };

  const pickFromPulse = (url) => {
    setQuery(url);
    analyze(url);
  };

  const shortcuts = useMemo(() => {
    if (!overview) return [];
    return [...(overview.top_watch || []).slice(0, 3), ...(overview.top_buy || []).slice(0, 1)];
  }, [overview]);

  /* ---------------------------------------------------------------- */
  /* Render                                                            */
  /* ---------------------------------------------------------------- */
  return (
    <>
      <Head>
        <title>Flipkart Price &amp; Discount Analytics</title>
        <meta
          name="description"
          content="Two-stage machine learning dashboard for Flipkart price dynamics, discounts and predicted price floors."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="shell" ref={shellRef}>
        <header className="topbar">
          <div className="brand">
            <div className="brand-mark">FP</div>
            <div className="brand-text">
              <strong>Price &amp; Discount Analytics</strong>
              <span>Phase 1–4 pipeline</span>
            </div>
          </div>
          <div className={`status ${apiDown ? "down" : health ? "live" : ""}`}>
            <span className="status-dot" />
            {apiDown
              ? "API offline"
              : health
              ? `API live · ${health.products} products · ${health.price_history_rows.toLocaleString("en-IN")} price rows`
              : "Connecting…"}
          </div>
        </header>

        <section className="hero">
          <span className="eyebrow">XGBoost trigger · CatBoost magnitude · walk-forward CV</span>
          <h1>
            Should you buy now,
            <br /> or <em>wait for the drop?</em>
          </h1>
          <p className="hero-sub">
            Paste a product link or search the catalogue. Stage 1 decides whether a cut of
            5% or more is coming in the next 7 days; Stage 2 prices the floor it will land on.
          </p>

          <div className="search-shell">
            <form className="search-form" onSubmit={handleSubmit}>
              <label className="search-field">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
                  <path d="m16.5 16.5 4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
                <input
                  className="search-input"
                  type="text"
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    setSuggestOpen(true);
                  }}
                  onFocus={() => setSuggestOpen(true)}
                  onKeyDown={handleKeyDown}
                  placeholder="Search a product, or paste a Flipkart URL…"
                  disabled={loading}
                  aria-label="Product search"
                  autoComplete="off"
                />
              </label>
              <button className="search-btn" type="submit" disabled={loading}>
                {loading ? (
                  <>
                    <Spinner />
                    Analyzing
                  </>
                ) : (
                  "Predict"
                )}
              </button>
            </form>

            {suggestOpen && suggestions.length > 0 && (
              <ul className="suggestions">
                {suggestions.map((p, i) => (
                  <li
                    key={p.product_id}
                    className={`suggestion ${i === activeIndex ? "active" : ""}`}
                    onMouseEnter={() => setActiveIndex(i)}
                    onClick={() => pickSuggestion(p)}
                  >
                    <span className="suggestion-body">
                      <span className="suggestion-title">{p.title}</span>
                      <span className="suggestion-meta">
                        {p.brand} · {p.category}
                        {p.discount_percentage != null
                          ? ` · ${formatPercent(p.discount_percentage, 0)} off`
                          : ""}
                      </span>
                    </span>
                    <span className="suggestion-price">
                      {formatINR(p.current_price ?? p.base_mrp)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {shortcuts.length > 0 && (
            <div className="quick-row">
              <span className="quick-label">Try</span>
              {shortcuts.map((p) => (
                <button
                  key={p.product_id}
                  type="button"
                  className="chip"
                  onClick={() => pickFromPulse(p.product_url)}
                >
                  {p.title}
                </button>
              ))}
            </div>
          )}

          {apiDown && (
            <div className="notice">
              <span>⚠</span>
              <span>
                Can’t reach the backend at <code>{API}</code>. Start it from the project root with{" "}
                <code>python -m uvicorn api.main:app --reload</code>, then reload this page.
              </span>
            </div>
          )}
        </section>

        <section className="stat-grid">
          <Stat
            label="Catalogue"
            value={overview ? overview.total_products.toLocaleString("en-IN") : null}
            hint="products modelled"
          />
          <Stat
            label="Flagged for a cut"
            value={overview ? formatPercent((overview.dropping / overview.total_products) * 100, 0) : null}
            hint={overview ? `${overview.dropping} wait · ${overview.stable} buy` : ""}
            tone="accent"
          />
          <Stat
            label="Avg discount today"
            value={overview ? formatPercent(overview.avg_discount, 0) : null}
            hint="across the catalogue"
            tone="info"
          />
          <Stat
            label="Avg predicted cut"
            value={overview ? formatPercent(overview.avg_expected_drop_pct, 0) : null}
            hint="where a drop is flagged"
            tone="good"
          />
        </section>

        {error && (
          <div className="notice">
            <span>⚠</span>
            <span>{error}</span>
          </div>
        )}

        {loading && <ResultSkeleton />}

        {result && <ResultPanel result={result} />}

        <section className="panel">
          <div className="panel-head">
            <div>
              <h2 className="panel-title">Market pulse</h2>
              <p className="panel-sub">
                Both stages run across the whole catalogue
                {overview?.generated_at ? ` · scored ${overview.generated_at.replace("T", " ")}` : ""}
              </p>
            </div>
          </div>
          <MarketPulse overview={overview} onPick={pickFromPulse} />
        </section>

        <footer className="footer">
          <span>
            <strong>Data:</strong> deterministic synthetic generator ({health?.products ?? 500} products,
            {" "}{health ? health.price_history_rows.toLocaleString("en-IN") : "82,666"} daily price rows).
            No external API or crawler involved.
          </span>
          <span>
            <strong>Validation:</strong> TimeSeriesSplit walk-forward — no random splits, so no future
            price leaks into training.
          </span>
        </footer>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Result                                                             */
/* ------------------------------------------------------------------ */
function ResultPanel({ result }) {
  const theme = verdictTheme(result.verdict, result.classification);
  const probability = result.drop_probability ?? 0;
  const price = result.current_price ?? 0;

  // Recompute from the raw numbers so the banner never disagrees with the tiles.
  const saving =
    result.predicted_price_floor != null ? Math.max(0, price - result.predicted_price_floor) : null;

  return (
    <section className="panel" aria-live="polite">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">{result.product_title}</h2>
          <p className="panel-sub">
            {result.product_brand} · {result.product_category} · {result.price_segment} segment
          </p>
        </div>
      </div>

      <div className={`verdict ${theme.tone}`}>
        <div className="verdict-left">
          <span className="verdict-tag">
            <i />
            {theme.label}
          </span>
          <h3 className="verdict-title">{result.status_badge}</h3>
          <p className="verdict-note">{theme.note}</p>
        </div>
        <div className="meter-wrap">
          <div className="meter-top">
            <span>Drop probability</span>
            <b>{formatPercent(probability * 100, 1)}</b>
          </div>
          <div className="meter">
            <div className="meter-fill" style={{ width: `${probability * 100}%` }} />
          </div>
          <div className="meter-scale">
            <span>Stable</span>
            <span>50%</span>
            <span>Certain</span>
          </div>
        </div>
      </div>

      <div className="kpi-grid">
        <Kpi label="Current price" value={formatINR(price)} hint="latest observation" />
        <Kpi label="MRP" value={formatINR(result.base_mrp)} hint="list price" />
        <Kpi
          label="Discount"
          value={formatPercent(result.discount_percentage, 1)}
          hint="already off MRP"
          tone="sky"
        />
        <Kpi
          label="Predicted floor"
          value={result.predicted_price_floor != null ? formatINR(result.predicted_price_floor) : "—"}
          hint={result.predicted_price_floor != null ? "Stage 2 estimate" : "no cut expected"}
          tone={result.predicted_price_floor != null ? "amber" : undefined}
        />
        <Kpi
          label="Expected cut"
          value={result.expected_drop_pct != null ? formatPercent(result.expected_drop_pct, 1) : "—"}
          hint={saving != null ? `${formatINR(saving)} below today` : "—"}
          tone={result.expected_drop_pct != null ? "rose" : undefined}
        />
        <Kpi
          label="Ratings"
          value={Number(result.rating_count ?? 0).toLocaleString("en-IN")}
          hint={`+${Number(result.rating_count_velocity ?? 0).toFixed(1)}/day velocity`}
          tone="emerald"
        />
      </div>

      <div className="panel-head" style={{ marginTop: 30, marginBottom: 14 }}>
        <div>
          <h3 className="panel-title">Historical trend</h3>
          <p className="panel-sub">
            {result.history?.length ?? 0} daily observations from price_history_logs
          </p>
        </div>
      </div>
      <PriceChart
        history={result.history}
        floor={result.predicted_price_floor}
        currentPrice={price}
      />

      <div className="panel-head" style={{ marginTop: 30, marginBottom: 14 }}>
        <div>
          <h3 className="panel-title">Model trace</h3>
          <p className="panel-sub">Sequentially chained — Stage 2 only runs when Stage 1 fires</p>
        </div>
      </div>
      <div className="pipeline">
        <div className="stage">
          <div className="stage-label">
            <span className="stage-num">1</span> Trigger model · XGBoost
          </div>
          <div className="stage-title">
            {result.classification === "drop" ? "Cut expected" : "No cut expected"}
          </div>
          <div className="stage-meta">
            P(drop ≥ 5% in 7 days) = {probability.toFixed(4)} · decision{" "}
            {result.classification_state === 1 ? "1" : "0"}
          </div>
        </div>
        <div className="stage-arrow" aria-hidden="true">
          →
        </div>
        <div className={`stage ${result.predicted_price_floor == null ? "disabled" : ""}`}>
          <div className="stage-label">
            <span className="stage-num">2</span> Magnitude model · CatBoost
          </div>
          <div className="stage-title">
            {result.predicted_price_floor != null
              ? `${formatINR(result.predicted_price_floor)} price floor`
              : "Not executed"}
          </div>
          <div className="stage-meta">
            {result.predicted_price_floor != null
              ? `${formatPercent(result.expected_drop_pct, 1)} below today · ${formatINR(saving)} potential saving`
              : "Stage 1 returned class 0, so no floor was priced"}
          </div>
        </div>
      </div>
    </section>
  );
}

function Kpi({ label, value, hint, tone }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value ${tone || ""}`}>{value}</div>
      <div className="kpi-hint">{hint}</div>
    </div>
  );
}

function Stat({ label, value, hint, tone }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      {value === null || value === undefined ? (
        <div className="skeleton" style={{ height: 30, marginTop: 10 }} />
      ) : (
        <div className={`stat-value ${tone || ""}`}>{value}</div>
      )}
      <div className="stat-hint">{hint}</div>
    </div>
  );
}

function ResultSkeleton() {
  return (
    <section className="panel" aria-busy="true">
      <div className="skeleton" style={{ height: 24, width: "48%", marginBottom: 20 }} />
      <div className="skeleton" style={{ height: 96, marginBottom: 16 }} />
      <div className="kpi-grid">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="skeleton" style={{ height: 84 }} />
        ))}
      </div>
      <div className="skeleton" style={{ height: 260, marginTop: 22 }} />
    </section>
  );
}

function Spinner({ size = 15 }) {
  return (
    <svg
      className="spinner"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.28" />
      <path
        d="M21 12a9 9 0 0 0-9-9"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}
