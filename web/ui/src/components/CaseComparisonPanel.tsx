import { caseMetric, type CasePair } from "../metrics";
import StatusBadge from "./StatusBadge";

type MetricDirection = "higher" | "lower";

interface ComparisonMetric {
  key: string;
  label: string;
  unit: string;
  direction: MetricDirection;
}

const comparisonMetrics: ComparisonMetric[] = [
  { key: "p50_ttft", label: "TTFT P50", unit: "s", direction: "lower" },
  { key: "p50_tpot", label: "TPOT P50", unit: "s", direction: "lower" },
  { key: "overall_throughput", label: "整体吞吐", unit: "tok/s", direction: "higher" },
  { key: "goodput_pct", label: "Goodput", unit: "%", direction: "higher" },
];

const sideKeys = ["entryA", "entryB"] as const;

function formatNumber(value: number): string {
  const digits = Math.abs(value) >= 100 ? 1 : Math.abs(value) >= 10 ? 2 : 4;
  return value.toFixed(digits).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

function formatMetricValue(value: number, unit: string): string {
  return `${formatNumber(value)} ${unit}`.trim();
}

function formatPercent(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${Math.abs(value).toFixed(1)}%`;
}

function relativeChange(valueA: number, valueB: number): number {
  if (valueA === 0) return valueB === 0 ? 0 : Number.POSITIVE_INFINITY;
  return ((valueB - valueA) / Math.abs(valueA)) * 100;
}

function metricMax(pairs: CasePair[], metricKey: string): number {
  const values = pairs.flatMap((pair) =>
    sideKeys.map((entryKey) => caseMetric(pair[entryKey], metricKey)),
  ).filter((value): value is number => value !== null && Number.isFinite(value) && value >= 0);
  return values.length ? Math.max(...values) : 1;
}

function comparisonWinner(
  valueA: number | null,
  valueB: number | null,
  direction: MetricDirection,
): "a" | "b" | null {
  if (valueA === null || valueB === null || valueA === valueB) return null;
  if (direction === "higher") return valueA > valueB ? "a" : "b";
  return valueA < valueB ? "a" : "b";
}

function MetricComparison({
  pair,
  metric,
  max,
}: {
  pair: CasePair;
  metric: ComparisonMetric;
  max: number;
}) {
  const valueA = caseMetric(pair.entryA, metric.key);
  const valueB = caseMetric(pair.entryB, metric.key);
  const winner = comparisonWinner(valueA, valueB, metric.direction);
  const hasBoth = valueA !== null && valueB !== null;
  const deltaText = !hasBoth
    ? valueA === null && valueB === null ? "无数据" : "缺一侧"
    : winner === null
      ? "持平"
      : `${winner.toUpperCase()} 更优 ${formatPercent(relativeChange(valueA, valueB))}`;

  return (
    <div className="case-comparison-metric">
      <div className="case-comparison-metric-head">
        <span className="case-comparison-metric-label">{metric.label}</span>
        <span className={winner ? `case-comparison-winner ${winner}` : "case-comparison-winner"}>{deltaText}</span>
      </div>
      <div className="case-comparison-bars">
        {sideKeys.map((entryKey, index) => {
          const value = index === 0 ? valueA : valueB;
          const side = index === 0 ? "a" : "b";
          const ratio = value === null || max <= 0 || value < 0 ? 0 : Math.min((value / max) * 100, 100);
          return (
            <div className="case-comparison-bar" key={entryKey}>
              <span className="case-comparison-side" aria-hidden="true">{side.toUpperCase()}</span>
              <div className="case-comparison-track">
                <span className={`case-comparison-fill ${side}`} style={{ width: `${ratio}%` }} />
              </div>
              <span className="case-comparison-value">
                {value === null ? "—" : formatMetricValue(value, metric.unit)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function CaseComparisonPanel({ pairs }: { pairs: CasePair[] }) {
  const maxima = comparisonMetrics.map((metric) => ({
    ...metric,
    max: metricMax(pairs, metric.key),
  }));

  return (
    <section className="panel case-comparison" aria-label="用例对照">
      <div className="case-comparison-header">
        <div>
          <h3>用例对照</h3>
          <p className="case-comparison-subtitle">
            按用例对齐 A/B 两份报告，较优方向以角标和条形边界标识
          </p>
        </div>
      </div>
      {pairs.length === 0 ? (
        <div className="case-comparison-empty">暂无可对齐的用例</div>
      ) : (
        <div className="case-comparison-grid">
          {pairs.map((pair) => {
            const title = pair.entryA?.name ?? pair.entryB?.name ?? pair.label;
            const detail = pair.entryA?.id ?? pair.entryB?.id ?? "";
            return (
              <article className="case-comparison-card" key={pair.key}>
                <header className="case-comparison-case-head">
                  <div className="case-comparison-title-block">
                    <span className="case-comparison-title">{title}</span>
                    {detail ? <span className="case-comparison-detail">{detail}</span> : null}
                  </div>
                  <div className="case-comparison-status">
                    <StatusBadge status={pair.statusA} />
                    <StatusBadge status={pair.statusB} />
                  </div>
                </header>
                <div className="case-comparison-metrics">
                  {maxima.map((metric) => (
                    <MetricComparison key={metric.key} pair={pair} metric={metric} max={metric.max} />
                  ))}
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
