import type { Case } from "./types";

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (!Number.isFinite(value ?? Number.NaN)) return "—";
  const numeric = value as number;
  const abs = Math.abs(numeric);
  if (abs >= 1e9) return `${(numeric / 1e9).toFixed(digits)}B`;
  if (abs >= 1e6) return `${(numeric / 1e6).toFixed(digits)}M`;
  if (abs >= 1e3) return `${(numeric / 1e3).toFixed(digits)}K`;
  return numeric.toFixed(abs >= 10 ? 1 : digits);
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? "—"
    : parsed.toLocaleString("zh-CN", { hour12: false });
}

export function metric(entry: Case | undefined, key: string): number | null {
  const value = entry?.result?.metrics?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function concurrency(entry: Case | undefined): number | null {
  const requests = entry?.matrix?.requests;
  if (typeof requests === "number") return requests;
  return typeof requests?.concurrency === "number" ? requests.concurrency : null;
}

export function finiteMean(values: (number | null | undefined)[]): number | null {
  const items = values.filter((value): value is number => Number.isFinite(value as number));
  return items.length
    ? items.reduce((sum, value) => sum + value, 0) / items.length
    : null;
}
