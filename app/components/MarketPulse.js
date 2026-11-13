import React from "react";
import { formatINR, formatPercent } from "../lib/format";

/**
 * Catalogue-wide view: how the two-stage engine is classifying all products
 * right now, which categories are being cut hardest, and a short list of
 * recommendations on both sides of the verdict.
 */
export default function MarketPulse({ overview, onPick }) {
  if (!overview) {
    return (
      <div className="pulse-grid">
        <div>
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="skeleton" style={{ height: 26, marginBottom: 12 }} />
          ))}
        </div>
        <div>
          {[0, 1, 2].map((i) => (
            <div key={i} className="skeleton" style={{ height: 46, marginBottom: 10 }} />
          ))}
        </div>
      </div>
    );
  }

  const {
    total_products: total,
    dropping,
    stable,
    avg_discount: avgDiscount,
    by_category: categories = [],
    top_watch: topWatch = [],
    top_buy: topBuy = [],
  } = overview;

  const dropShare = total ? (dropping / total) * 100 : 0;
  const maxShare = Math.max(...categories.map((c) => c.drop_share), 1);

  return (
    <div className="pulse-grid">
      <div>
        <div className="pick-head">
          <span>Verdict split across {total} products</span>
          <span>{formatPercent(dropShare, 0)} flagged</span>
        </div>

        <div className="meter" style={{ height: 12, marginBottom: 10 }}>
          <div
            className="meter-fill"
            style={{
              width: `${dropShare}%`,
              background: "linear-gradient(90deg, #fbbf24, #fb7185)",
            }}
          />
        </div>
        <p className="panel-sub" style={{ marginBottom: 22 }}>
          {dropping} wait · {stable} buy · {formatPercent(avgDiscount, 0)} average discount today
        </p>

        <div className="pick-head">
          <span>Which categories are being cut</span>
          <span>drop share</span>
        </div>
        <div className="cat-list">
          {categories.map((c) => (
            <div className="cat-row" key={c.category}>
              <span className="cat-name">{c.category}</span>
              <span className="cat-pct">
                {formatPercent(c.drop_share, 0)}{" "}
                <span style={{ color: "var(--dim)" }}>
                  · {formatPercent(c.avg_discount, 0)} off
                </span>
              </span>
              <span className="cat-track">
                <span
                  className="cat-bar"
                  style={{ width: `${(c.drop_share / maxShare) * 100}%`, display: "block" }}
                />
              </span>
            </div>
          ))}
        </div>
      </div>

      <div>
        <div className="pick-group">
          <div className="pick-head">
            <span>Deepest predicted cuts</span>
            <span>wait</span>
          </div>
          {topWatch.map((p) => (
            <button
              key={p.product_id}
              type="button"
              className="pick"
              onClick={() => onPick(p.product_url)}
              title="Analyse this product"
            >
              <span className="pick-body">
                <span className="pick-title">{p.title}</span>
                <span className="pick-meta">
                  {p.brand} · {formatINR(p.current_price)}
                </span>
              </span>
              <span className="pick-value amber">
                −{formatPercent(p.expected_drop_pct, 0)}
                <small>
                  {formatINR(p.predicted_price_floor)} floor
                </small>
              </span>
            </button>
          ))}
        </div>

        <div className="pick-group">
          <div className="pick-head">
            <span>Best deals right now</span>
            <span>buy</span>
          </div>
          {topBuy.map((p) => (
            <button
              key={p.product_id}
              type="button"
              className="pick"
              onClick={() => onPick(p.product_url)}
              title="Analyse this product"
            >
              <span className="pick-body">
                <span className="pick-title">{p.title}</span>
                <span className="pick-meta">
                  {p.brand} · {formatINR(p.current_price)}
                </span>
              </span>
              <span className="pick-value emerald">
                {formatPercent(p.discount_percentage, 0)}
                <small>{formatPercent(p.drop_probability * 100, 0)} drop risk</small>
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
