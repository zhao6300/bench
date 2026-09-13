import { caseConcurrency, caseKey, caseMetric, caseParamNumber, caseStringParam } from "../metrics";
import { formatNumber } from "../format";
import StatusBadge from "./StatusBadge";
import type { Case } from "../types";

export interface CaseDetailPanelProps {
  caseEntry: Case | undefined;
  onClose: () => void;
}

interface CaseMetricRow {
  key: string;
  label: string;
  unit: string;
}

interface CaseParamRow {
  key: string;
  label: string;
}

const coreMetrics: CaseMetricRow[] = [
  { key: "p50_ttft", label: "TTFT P50", unit: "s" },
  { key: "p50_tpot", label: "TPOT P50", unit: "s" },
  { key: "overall_throughput", label: "整体吞吐", unit: "tok/s" },
  { key: "goodput_pct", label: "Goodput", unit: "%" },
  { key: "wall_time", label: "用例耗时", unit: "s" },
  { key: "concurrency", label: "并发", unit: "" },
];

const metricSections: { name: string; metrics: CaseMetricRow[] }[] = [
  {
    name: "延迟分位",
    metrics: [
      { key: "p50_ttft", label: "TTFT P50", unit: "s" },
      { key: "p90_ttft", label: "TTFT P90", unit: "s" },
      { key: "p99_ttft", label: "TTFT P99", unit: "s" },
      { key: "p50_tpot", label: "TPOT P50", unit: "s" },
      { key: "p90_tpot", label: "TPOT P90", unit: "s" },
      { key: "p99_tpot", label: "TPOT P99", unit: "s" },
    ],
  },
  {
    name: "吞吐与 QPS",
    metrics: [
      { key: "overall_throughput", label: "整体吞吐", unit: "tok/s" },
      { key: "prefill_throughput", label: "预填充吞吐", unit: "tok/s" },
      { key: "decode_throughput", label: "解码吞吐", unit: "tok/s" },
      { key: "qps", label: "QPS", unit: "req/s" },
      { key: "goodput_qps", label: "Goodput QPS", unit: "req/s" },
      { key: "goodput_pct", label: "Goodput", unit: "%" },
    ],
  },
];

const parameterRows: CaseParamRow[] = [
  { key: "id", label: "ID" },
  { key: "name", label: "名称" },
  { key: "model", label: "模型" },
  { key: "dataset", label: "数据集" },
  { key: "concurrency", label: "并发" },
  { key: "num_prompts", label: "请求数" },
  { key: "max_tokens", label: "最大 tokens" },
  { key: "tp_size", label: "TP size" },
  { key: "gpu_mem_util", label: "GPU 内存" },
  { key: "slo_ttft", label: "SLO TTFT" },
  { key: "slo_tpot", label: "SLO TPOT" },
];

function formatMetric(row: CaseMetricRow, caseEntry: Case | undefined): string {
  const value = caseMetric(caseEntry, row.key);
  if (value === null) {
    if (row.key === "concurrency") return caseConcurrency(caseEntry)?.toString() ?? "—";
    return "—";
  }
  return formatNumber(value, row.unit ? undefined : 0) + (row.unit ? ` ${row.unit}` : "");
}

function parameterValue(row: CaseParamRow, caseEntry: Case | undefined): string {
  if (!caseEntry) return "—";
  switch (row.key) {
    case "id":
      return caseKey(caseEntry) || "—";
    case "name":
      return caseEntry.name ?? "—";
    case "concurrency":
      return caseConcurrency(caseEntry)?.toString() ?? "—";
    default:
      return caseStringParam(caseEntry, row.key) || caseParamNumber(caseEntry, row.key)?.toString() || "—";
  }
}

function CaseMetricItem({ header, value }: { header: string; value: string }) {
  return (
    <div key={header}>
      <span>{header}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function CaseDetailPanel({ caseEntry, onClose }: CaseDetailPanelProps) {
  return (
    <main className="case-detail">
      <article className="case-detail-panel">
        <header className="case-detail-head">
          <div>
            <h1>{caseEntry?.name ?? "用例"}</h1>
            <span className="case-detail-id">{caseKey(caseEntry)}</span>
          </div>
          <div className="case-detail-head-actions">
            <StatusBadge status={caseEntry?.status} />
            <button type="button" className="case-detail-close" onClick={onClose}>
              返回
            </button>
          </div>
        </header>

        <div className="case-detail-grid">
          <section className="case-detail-card">
            <h2>核心指标</h2>
            <div className="case-detail-items">
              {coreMetrics.map((metric) => (
                <CaseMetricItem key={metric.key} header={metric.label} value={formatMetric(metric, caseEntry)} />
              ))}
            </div>
          </section>

          <section className="case-detail-card">
            <h2>配置</h2>
            <div className="case-detail-items">
              {parameterRows.map((row) => (
                <CaseMetricItem key={row.key} header={row.label} value={parameterValue(row, caseEntry)} />
              ))}
            </div>
          </section>
        </div>

        <div className="case-detail-sections">
          {metricSections.map((section) => (
            <section key={section.name} className="case-detail-card">
              <h2>{section.name}</h2>
              <div className="case-detail-items">
                {section.metrics.map((metric) => (
                  <CaseMetricItem key={metric.key} header={metric.label} value={formatMetric(metric, caseEntry)} />
                ))}
              </div>
            </section>
          ))}
        </div>
      </article>
    </main>
  );
}
