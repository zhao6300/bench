import { caseMetric, metricOptions, type CasePair } from "../metrics";
import StatusBadge from "./StatusBadge";
import { useState } from "react";

type MetricDirection = "higher" | "lower" | "neutral";

interface ComparisonMetric {
  key: string;
  label: string;
  unit: string;
  direction: MetricDirection;
}

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

const detailMetricKeys = [
  "p50_ttft",
  "p90_ttft",
  "p99_ttft",
  "p50_tpot",
  "p90_tpot",
  "p99_tpot",
  "overall_throughput",
  "goodput_pct",
  "qps",
  "failure_rate",
] as const;

const detailMetrics = metricOptions.filter((metric) =>
  (detailMetricKeys as readonly string[]).includes(metric.key),
);

function pairMetricMax(pair: CasePair, metricKey: string): number {
  const values = sideKeys
    .map((sideKey) => caseMetric(pair[sideKey], metricKey))
    .filter((value): value is number => value !== null && Number.isFinite(value) && value >= 0);
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
        <span
          className={winner ? `case-comparison-winner ${winner}` : "case-comparison-winner"}
          title={deltaText}
        >
          {winner ? (
            <svg className="case-comparison-winner-icon" viewBox="0 0 12 12" aria-hidden="true">
              <path d="M1 7 4.35 10.35 11 3.7" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
            </svg>
          ) : null}
          <span className="case-comparison-winner-text">{deltaText}</span>
        </span>
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
  const [selectedPair, setSelectedPair] = useState<CasePair | undefined>();

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
        <>
        {selectedPair ? (
          <section className="case-pair-detail" aria-label="用例详细对比">
            <header className="case-pair-detail-head">
              <div>
                <h4>用例详细对比</h4>
                <span className="case-pair-detail-title">
                  {selectedPair.entryA?.name ?? selectedPair.entryB?.name ?? selectedPair.label}
                </span>
              </div>
              <div className="case-pair-detail-status">
                <StatusBadge status={selectedPair.statusA} />
                <StatusBadge status={selectedPair.statusB} />
              </div>
              <button type="button" onClick={() => setSelectedPair(undefined)}>返回</button>
            </header>
            <div className="case-pair-detail-metrics">
              {detailMetrics.map((metric) => (
                <MetricComparison
                  key={metric.key}
                  pair={selectedPair}
                  metric={metric}
                  max={pairMetricMax(selectedPair, metric.key)}
                />
              ))}
            </div>
          </section>
        ) : null}

        <div className="case-comparison-grid">
          {pairs.map((pair) => {
            const title = pair.entryA?.name ?? pair.entryB?.name ?? pair.label;
            const detail = pair.entryA?.id ?? pair.entryB?.id ?? "";
            return (
              <article
                className="case-comparison-card"
                key={pair.key}
                tabIndex={0}
                role="button"
                aria-label={`查看 ${title} 详细对比`}
                onClick={() => setSelectedPair(pair)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setSelectedPair(pair);
                  }
                }}
              >
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
                  {detailMetrics.map((metric) => (
                    <MetricComparison
                      key={metric.key}
                      pair={pair}
                      metric={metric}
                      max={pairMetricMax(pair, metric.key)}
                    />
                  ))}
                </div>
              </article>
            );
          })}
        </div>
        </>
      )}
    </section>
  );
}
