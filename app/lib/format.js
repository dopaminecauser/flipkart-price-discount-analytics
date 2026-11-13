/** Formatting helpers shared by the dashboard components. */

const inrFormatter = (digits) =>
  new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });

/** ₹ with Indian digit grouping; keeps paise for cheap items. */
export function formatINR(value, decimals) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const num = Number(value);
  const digits =
    decimals !== undefined ? decimals : Math.abs(num) < 1000 ? 2 : 0;
  return `₹${inrFormatter(digits).format(num)}`;
}

/** Compact axis label: ₹845, ₹12.4k, ₹1.2L. */
export function formatINRCompact(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "";
  if (Math.abs(num) >= 100000) return `₹${(num / 100000).toFixed(1)}L`;
  if (Math.abs(num) >= 1000) return `₹${(num / 1000).toFixed(num >= 10000 ? 0 : 1)}k`;
  if (Math.abs(num) >= 100) return `₹${num.toFixed(0)}`;
  return `₹${num.toFixed(1)}`;
}

export function formatPercent(value, decimals = 1) {
  if (value === null || value === undefined) return "—";
  return `${Number(value).toFixed(decimals)}%`;
}

/** "2027-04-22" -> "22 Apr" */
export function formatDate(timestamp) {
  if (!timestamp) return "";
  const date = new Date(`${timestamp}T00:00:00`);
  if (Number.isNaN(date.getTime())) return String(timestamp);
  return date.toLocaleDateString("en-IN", { day: "2-digit", month: "short" });
}

/** Verdict key -> presentation metadata. */
export function verdictTheme(verdict, classification) {
  if (verdict === "watch") {
    return { tone: "watch", label: "Watch closely", note: "Early warning signal" };
  }
  if (classification === "drop") {
    return {
      tone: "wait",
      label: "Wait recommended",
      note: "Stage 1 triggered — Stage 2 priced the upcoming floor",
    };
  }
  return {
    tone: "buy",
    label: "Buy now",
    note: "No meaningful cut expected in the next 7 days",
  };
}
