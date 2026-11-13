import React, { useMemo, useState } from "react";
import { formatDate, formatINR, formatINRCompact } from "../lib/format";

/**
 * Interactive price / discount history chart.
 *
 * Rendered as a single responsive SVG (viewBox + width:100%), so it reflows
 * with the layout without a resize listener. Hovering anywhere on the plot
 * snaps a crosshair to the nearest observation and shows a tooltip.
 */
const W = 760;
const H = 280;
const PAD = { l: 60, r: 20, t: 20, b: 32 };

const RANGES = [
  { key: "30", label: "30d" },
  { key: "90", label: "90d" },
  { key: "all", label: "All" },
];

export default function PriceChart({ history, floor, currentPrice }) {
  const [range, setRange] = useState("90");
  const [metric, setMetric] = useState("price");
  const [hoverIndex, setHoverIndex] = useState(null);

  const data = useMemo(() => {
    const all = Array.isArray(history) ? history : [];
    if (all.length === 0) return [];
    if (range === "all") return all;
    return all.slice(Math.max(0, all.length - Number(range)));
  }, [history, range]);

  const isPrice = metric === "price";
  const plotW = W - PAD.l - PAD.r;
  const plotH = H - PAD.t - PAD.b;

  const scale = useMemo(() => {
    if (data.length === 0) return null;
    const values = data.map((d) => (isPrice ? Number(d.price) : Number(d.discount)));
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (isPrice && floor != null) {
      min = Math.min(min, floor);
      max = Math.max(max, floor);
    }
    if (!isPrice) min = 0;
    if (max === min) {
      max = min + Math.max(min * 0.05, 1);
    }
    const pad = (max - min) * 0.12;
    return { min: Math.max(min - pad, isPrice ? 0 : 0), max: max + pad };
  }, [data, isPrice, floor]);

  if (data.length === 0 || !scale) {
    return (
      <div className="chart-wrap">
        <div className="skeleton" style={{ height: 220 }} />
      </div>
    );
  }

  const xOf = (i) =>
    PAD.l + (data.length === 1 ? plotW / 2 : (i / (data.length - 1)) * plotW);
  const yOf = (v) =>
    PAD.t + plotH - ((Number(v) - scale.min) / (scale.max - scale.min)) * plotH;

  const points = data.map((d, i) => ({
    x: xOf(i),
    y: yOf(isPrice ? d.price : d.discount),
    raw: d,
  }));

  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y.toFixed(2)}`)
    .join(" ");
  const areaPath = `${linePath} L${points[points.length - 1].x.toFixed(2)},${(
    PAD.t + plotH
  ).toFixed(2)} L${points[0].x.toFixed(2)},${(PAD.t + plotH).toFixed(2)} Z`;

  // Four horizontal gridlines with axis labels.
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((frac) => {
    const value = scale.max - frac * (scale.max - scale.min);
    return { value, y: PAD.t + frac * plotH };
  });

  // Five date ticks, de-duplicated so short ranges do not print twice.
  const dateTicks = [];
  const step = Math.max(1, Math.floor((data.length - 1) / 4));
  for (let i = 0; i < data.length; i += step) {
    if (dateTicks.length === 4 && i < data.length - 1) continue;
    dateTicks.push({ i, label: formatDate(data[i].timestamp) });
  }
  if (dateTicks[dateTicks.length - 1]?.i !== data.length - 1) {
    dateTicks.push({ i: data.length - 1, label: formatDate(data[data.length - 1].timestamp) });
  }

  const floorY = isPrice && floor != null ? yOf(floor) : null;
  const hovered = hoverIndex != null ? points[hoverIndex] : null;

  const handleMove = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const xInView = ((event.clientX - rect.left) / rect.width) * W;
    const ratio = (xInView - PAD.l) / plotW;
    const idx = Math.round(ratio * (data.length - 1));
    setHoverIndex(Math.min(Math.max(idx, 0), data.length - 1));
  };

  return (
    <div>
      <div className="chart-head">
        <div className="segmented" role="group" aria-label="Metric">
          <button
            type="button"
            className="seg-btn"
            aria-pressed={isPrice}
            onClick={() => setMetric("price")}
          >
            Price
          </button>
          <button
            type="button"
            className="seg-btn"
            aria-pressed={!isPrice}
            onClick={() => setMetric("discount")}
          >
            Discount %
          </button>
        </div>
        <div className="segmented" role="group" aria-label="Range">
          {RANGES.map((r) => (
            <button
              key={r.key}
              type="button"
              className="seg-btn"
              aria-pressed={range === r.key}
              onClick={() => {
                setRange(r.key);
                setHoverIndex(null);
              }}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      <div className="chart-wrap">
        <svg
          className="chart-svg"
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`${isPrice ? "Price" : "Discount"} history, ${data.length} observations`}
          onMouseMove={handleMove}
          onMouseLeave={() => setHoverIndex(null)}
        >
          <defs>
            <linearGradient id="chartArea" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.34" />
              <stop offset="70%" stopColor="#38bdf8" stopOpacity="0.05" />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="chartLine" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#6ee7b7" />
              <stop offset="55%" stopColor="#38bdf8" />
              <stop offset="100%" stopColor="#a78bfa" />
            </linearGradient>
          </defs>

          {ticks.map((t, i) => (
            <g key={i}>
              <line
                x1={PAD.l}
                x2={W - PAD.r}
                y1={t.y}
                y2={t.y}
                stroke="rgba(255,255,255,0.07)"
                strokeDasharray={i === ticks.length - 1 ? "0" : "3 5"}
              />
              <text
                x={PAD.l - 10}
                y={t.y + 4}
                textAnchor="end"
                fontSize="10.5"
                fill="#5f6d80"
              >
                {isPrice ? formatINRCompact(t.value) : `${t.value.toFixed(0)}%`}
              </text>
            </g>
          ))}

          <path d={areaPath} fill="url(#chartArea)" />
          <path
            d={linePath}
            fill="none"
            stroke="url(#chartLine)"
            strokeWidth="2.4"
            strokeLinejoin="round"
            strokeLinecap="round"
          />

          {floorY != null && (
            <g>
              <line
                x1={PAD.l}
                x2={W - PAD.r}
                y1={floorY}
                y2={floorY}
                stroke="#fbbf24"
                strokeWidth="1.6"
                strokeDasharray="6 5"
                opacity="0.9"
              />
              <text
                x={PAD.l + 6}
                y={floorY - 7}
                textAnchor="start"
                fontSize="10.5"
                fill="#fbbf24"
              >
                floor {formatINR(floor)}
              </text>
            </g>
          )}

          {currentPrice != null && isPrice && (
            <text
              x={points[points.length - 1].x - 6}
              y={points[points.length - 1].y - 12}
              textAnchor="end"
              fontSize="10.5"
              fill="#9fe8c8"
            >
              now {formatINR(currentPrice)}
            </text>
          )}

          {dateTicks.map((t) => (
            <text
              key={`d${t.i}`}
              x={xOf(t.i)}
              y={H - 8}
              textAnchor={t.i === 0 ? "start" : t.i === data.length - 1 ? "end" : "middle"}
              fontSize="10.5"
              fill="#5f6d80"
            >
              {t.label}
            </text>
          ))}

          {/* Hover crosshair */}
          {hovered && (
            <g>
              <line
                x1={hovered.x}
                x2={hovered.x}
                y1={PAD.t}
                y2={PAD.t + plotH}
                stroke="rgba(255,255,255,0.28)"
              />
              <circle cx={hovered.x} cy={hovered.y} r="4.5" fill="#38bdf8" stroke="#04121b" strokeWidth="2" />
            </g>
          )}

          {/* Transparent capture surface keeps hover smooth over the whole plot */}
          <rect
            x={PAD.l}
            y={PAD.t}
            width={plotW}
            height={plotH}
            fill="transparent"
          />
        </svg>

        {hovered && (
          <div
            className="chart-tooltip"
            style={{
              left: `${(hovered.x / W) * 100}%`,
              top: `${(hovered.y / H) * 100}%`,
            }}
          >
            <b>{formatINR(hovered.raw.price)}</b>
            <span>
              {formatDate(hovered.raw.timestamp)} · {Number(hovered.raw.discount).toFixed(1)}% off
            </span>
          </div>
        )}
      </div>

      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-swatch" /> {isPrice ? "Daily price" : "Discount %"}
        </span>
        {isPrice && floorY != null && (
          <span className="legend-item">
            <span className="legend-swatch dashed" /> Predicted floor
          </span>
        )}
        <span className="legend-item">Hover the plot for exact values</span>
      </div>
    </div>
  );
}
