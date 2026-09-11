import { formatBytes, formatDate, formatDuration } from "./format";
import type { Case, Report } from "./types";

export type MetricDirection = "higher" | "lower" | "neutral";

export interface MetricOption {
  key: string;
  label: string;
  unit: string;
  direction: MetricDirection;
}

export interface MetricGroup {
  name: string;
  options: MetricOption[];
}

export const metricGroups: MetricGroup[] = [
  {
    name: "时延",
    options: [
      { key: "p50_ttft", label: "TTFT P50", unit: "s", direction: "lower" },
      { key: "p90_ttft", label: "TTFT P90", unit: "s", direction: "lower" },
      { key: "p99_ttft", label: "TTFT P99", unit: "s", direction: "lower" },
      { key: "p50_tpot", label: "TPOT P50", unit: "s", direction: "lower" },
      { key: "p90_tpot", label: "TPOT P90", unit: "s", direction: "lower" },
      { key: "p99_tpot", label: "TPOT P99", unit: "s", direction: "lower" },
      { key: "p50_estimated_itl", label: "ITL P50", unit: "s", direction: "lower" },
      { key: "p90_estimated_itl", label: "ITL P90", unit: "s", direction: "lower" },
      { key: "p99_estimated_itl", label: "ITL P99", unit: "s", direction: "lower" },
      { key: "p50_e2e", label: "E2E P50", unit: "s", direction: "lower" },
      { key: "p90_e2e", label: "E2E P90", unit: "s", direction: "lower" },
      { key: "p99_e2e", label: "E2E P99", unit: "s", direction: "lower" },
    ],
  },
  {
    name: "速率",
    options: [
      { key: "overall_throughput", label: "整体吞吐", unit: "tok/s", direction: "higher" },
      { key: "decode_throughput", label: "解码吞吐", unit: "tok/s", direction: "higher" },
      { key: "prefill_throughput", label: "预填充吞吐", unit: "tok/s", direction: "higher" },
      { key: "prompt_throughput", label: "Prompt 吞吐", unit: "tok/s", direction: "higher" },
      { key: "goodput_pct", label: "Goodput", unit: "%", direction: "higher" },
      { key: "goodput_qps", label: "Goodput QPS", unit: "req/s", direction: "higher" },
      { key: "qps", label: "QPS", unit: "req/s", direction: "higher" },
    ],
  },
  {
    name: "质量与规模",
    options: [
      { key: "failure_rate", label: "失败率", unit: "%", direction: "lower" },
      { key: "total_prompt_tokens", label: "Prompt Tokens", unit: "", direction: "neutral" },
      { key: "total_generated_tokens", label: "Decode Tokens", unit: "", direction: "neutral" },
      { key: "successful", label: "成功请求", unit: "", direction: "neutral" },
      { key: "failed", label: "失败请求", unit: "", direction: "lower" },
      { key: "wall_time", label: "用例耗时", unit: "s", direction: "lower" },
    ],
  },
];

export const metricOptions = metricGroups.flatMap((group) => group.options);

export interface SummaryCard {
  label: string;
  value: string;
  detail: string;
  tone: string;
}

export interface StatusSegment {
  key: string;
  label: string;
  value: number;
}

export type MetricDelta = {
  option: MetricOption;
  valueA: number | null;
  valueB: number | null;
  delta: number | null;
  deltaPct: number | null;
  tone: "positive" | "negative" | "neutral";
};

export type CasePair = {
  key: string;
  label: string;
  entryA?: Case;
  entryB?: Case;
  statusA?: string;
  statusB?: string;
  matched: boolean;
};

const statusOrder = [
  "passed",
  "failed",
  "interrupted",
  "running",
  "pending",
  "skipped",
] as const;

export function caseMetric(caseEntry: Case | undefined, key: string): number | null {
  if (!caseEntry?.result?.metrics) return null;
  const value = caseEntry.result.metrics[key];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value.replace("%", ""));
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function caseMetricMean(cases: Case[] | undefined, key: string): number | null {
  const values = (cases ?? []).map((caseEntry) => caseMetric(caseEntry, key)).filter((value): value is number => value !== null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

export function caseConcurrency(caseEntry: Case | undefined): number | null {
  const requests = caseEntry?.matrix?.requests;
  if (typeof requests === "number") return requests;
  return typeof requests?.concurrency === "number" ? requests.concurrency : null;
}

export function caseKey(caseEntry: Case | undefined): string {
  return caseEntry?.case_key || caseEntry?.id || caseEntry?.name || "";
}

export function caseLabel(caseEntry: Case | undefined, index: number): string {
  const concurrency = caseConcurrency(caseEntry);
  if (concurrency !== null) return `C${concurrency}`;
  return caseEntry?.name ? `${index + 1}` : `用例 ${index + 1}`;
}

export function caseSortIndex(caseEntry: Case | undefined): number {
  return caseConcurrency(caseEntry) ?? 0;
}

function summarizeStatus(report: Report): Record<string, number> {
  const cases = report?.cases ?? [];
  const counts: Record<string, number> = {};
  const summary = report?.summary ?? {};

  for (const status of statusOrder) {
    const numeric = summary[status];
    counts[status] = typeof numeric === "number" && Number.isFinite(numeric)
      ? numeric
      : cases.filter((caseEntry) => caseEntry.status === status).length;
  }

  return counts;
}

export function statusSegments(report: Report): StatusSegment[] {
  const counts = summarizeStatus(report);
  const labels: Record<string, string> = {
    passed: "通过",
    failed: "失败",
    interrupted: "中断",
    running: "运行中",
    pending: "待运行",
    skipped: "跳过",
  };
  return statusOrder.map((key) => ({ key, label: labels[key], value: counts[key] ?? 0 }));
}

export function statusBadgeClass(status: string | undefined): string {
  if (!status) return "unknown";
  return statusOrder.includes(status as (typeof statusOrder)[number]) ? status : "unknown";
}

export function getSummaryCards(report: Report): SummaryCard[] {
  const cases = report?.cases ?? [];
  const counts = summarizeStatus(report);
  const total = counts.passed + counts.failed + counts.interrupted + counts.running + counts.pending + counts.skipped;
  const averageTTFT = caseMetricMean(cases, "p50_ttft");
  const averageTPOT = caseMetricMean(cases, "p50_tpot");
  const averageThroughput = caseMetricMean(cases, "overall_throughput");
  const averageGoodput = caseMetricMean(cases, "goodput_pct");
  const averageQPS = caseMetricMean(cases, "qps");
  const model = caseStringParam(cases[0], "model");
  const host = report?.environment?.host_inventory;
  const accelerator = host?.accelerators?.devices?.[0]?.name;
  const environment = accelerator ?? `${host?.cpu?.logical_cores ?? "—"}C`;

  return [
    {
      label: "通过 / 总数",
      value: `${counts.passed} / ${total}`,
      detail: `${counts.failed} failed · ${counts.skipped} skipped`,
      tone: "blue",
    },
    {
      label: "TTFT P50",
      value: averageTTFT === null ? "—" : `${formatDuration(averageTTFT)}s`,
      detail: "均值",
      tone: "purple",
    },
    {
      label: "TPOT P50",
      value: averageTPOT === null ? "—" : `${formatDuration(averageTPOT)}s`,
      detail: "均值",
      tone: "green",
    },
    {
      label: "吞吐量",
      value: averageThroughput === null ? "—" : `${formatNumber(averageThroughput)} tok/s`,
      detail: "均值",
      tone: "blue",
    },
    {
      label: "Goodput",
      value: averageGoodput === null ? "—" : `${formatNumber(averageGoodput, 1)}%`,
      detail: "均值",
      tone: "green",
    },
    {
      label: "请求数",
      value: averageQPS === null ? "—" : `${formatNumber(averageQPS, 1)} req/s`,
      detail: counts.pending > 0 ? `${counts.pending} pending` : "均值",
      tone: "amber",
    },
    {
      label: "核心配置",
      value: environment,
      detail: model || report?.suite?.name || "—",
      tone: "purple",
    },
  ];
}

export function environmentFacts(report: Report): { label: string; value: string }[] {
  const environment = report?.environment;
  const host = environment?.host_inventory;
  const accelerator = host?.accelerators?.devices?.[0]?.name;
  return [
    { label: "Python", value: environment?.python ?? "—" },
    { label: "主机", value: environment?.hostname ?? "—" },
    { label: "CPU", value: host?.cpu?.model ? `${host.cpu.model} · ${host.cpu.logical_cores ?? "—"}C` : "—" },
    { label: "内存", value: host?.memory?.total_bytes === undefined ? "—" : formatBytes(host.memory.total_bytes) },
    { label: "加速器", value: accelerator ?? "未发现" },
    { label: "开始", value: formatDate(report?.suite?.started_at) },
    { label: "结束", value: formatDate(report?.suite?.finished_at) },
  ];
}

export function caseStringParam(caseEntry: Case | undefined, key: string): string {
  const value = caseEntry?.params?.[key];
  return typeof value === "string" ? value : "";
}

export function caseParamNumber(caseEntry: Case | undefined, key: string): number | null {
  if (!caseEntry?.params) return null;
  const value = caseEntry.params[key];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function pairCases(reportA: Report | null, reportB: Report | null): CasePair[] {
  const casesA = reportA?.cases ?? [];
  const casesB = reportB?.cases ?? [];
  const byKeyA = new Map(casesA.map((entry, index) => [caseKey(entry) || `case-${index}`, entry]));
  const byKeyB = new Map(casesB.map((entry, index) => [caseKey(entry) || `case-${index}`, entry]));
  const matchedKeys = [...byKeyA.keys()].filter((key) => byKeyB.has(key));
  const onlyAKeys = [...byKeyA.keys()].filter((key) => !byKeyB.has(key));
  const onlyBKeys = [...byKeyB.keys()].filter((key) => !byKeyA.has(key));

  const matched = matchedKeys.map((key) => {
    const entryA = byKeyA.get(key);
    const entryB = byKeyB.get(key);
    return {
      key,
      label: caseLabel(entryA, 0),
      entryA,
      entryB,
      statusA: entryA?.status ?? "",
      statusB: entryB?.status ?? "",
      matched: true,
    };
  });

  const onlyA = onlyAKeys.map((key) => {
    const entryA = byKeyA.get(key);
    return { key, label: caseLabel(entryA, 0), entryA, entryB: undefined, statusA: entryA?.status ?? "", statusB: "", matched: false };
  });
  const onlyB = onlyBKeys.map((key) => {
    const entryB = byKeyB.get(key);
    return { key, label: caseLabel(undefined, 0), entryA: undefined, entryB, statusA: "", statusB: entryB?.status ?? "", matched: false };
  });

  return [...matched, ...onlyA, ...onlyB].sort((left, right) => caseSortIndex(left.entryA ?? left.entryB) - caseSortIndex(right.entryA ?? right.entryB));
}

export function metricComparisonRows(reportA: Report | null, reportB: Report | null): MetricDelta[] {
  const casesA = reportA?.cases ?? [];
  const casesB = reportB?.cases ?? [];

  return metricOptions.map((option) => {
    const valueA = caseMetricMean(casesA, option.key);
    const valueB = caseMetricMean(casesB, option.key);
    const delta = valueA !== null && valueB !== null ? valueB - valueA : null;
    const deltaPct = valueA !== null && valueB !== null && valueA !== 0
      ? ((valueB - valueA) / Math.abs(valueA)) * 100
      : null;
    let tone: MetricDelta["tone"] = "neutral";
    if (delta !== 0 && option.direction !== "neutral" && delta !== null) {
      tone = option.direction === "higher" ? (delta > 0 ? "positive" : "negative") : (delta < 0 ? "positive" : "negative");
    } else if (delta !== null && delta !== 0) {
      tone = "neutral";
    }
    return { option, valueA, valueB, delta, deltaPct, tone };
  });
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}
