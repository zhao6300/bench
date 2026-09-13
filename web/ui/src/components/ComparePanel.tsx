import { formatDate, formatDuration } from "../format";
import { caseMetric, caseLabel, pairCases } from "../metrics";
import { LineChart } from "./charts";
import type { Report } from "../types";
import type { Run } from "../types";
import StatusBadge from "./StatusBadge";
import { CaseComparisonPanel } from "./CaseComparisonPanel";
import { ConfigComparePanel } from "./ConfigComparePanel";

const comparisonMetrics = [
  { key: "p50_ttft", label: "TTFT P50", unit: "s", color: "chart-color-purple" },
  { key: "p50_tpot", label: "TPOT P50", unit: "s", color: "chart-color-green" },
  { key: "overall_throughput", label: "吞吐量", unit: "tok/s", color: "chart-color-blue" },
  { key: "goodput_pct", label: "Goodput", unit: "%", color: "chart-color-amber" },
] as const;

export function ComparePanel({ reportA, reportB, labelA, labelB, items, onChangeA, onChangeB }: {
  reportA: Report | null;
  reportB: Report | null;
  labelA: string;
  labelB: string;
  items?: Run[];
  onChangeA: (filename: string) => void;
  onChangeB: (filename: string) => void;
}) {
  const pairs = pairCases(reportA, reportB);
  const matched = pairs.filter((pair) => pair.matched);
  const labels = matched.map((pair) => caseLabel(pair.entryA, 0) || caseLabel(pair.entryB, 0));

  return (
    <section className="compare">
      <article className="panel pair-panel">
        <div className="panel-head">
          <div>
            <h2>报告对比</h2>
            <p className="hero-subtitle">
              {formatDate(reportA?.suite?.started_at)} → {formatDate(reportB?.suite?.finished_at)}
            </p>
          </div>
          <div className="hero-actions">
            <StatusBadge status={reportA?.suite?.run_state} />
            <StatusBadge status={reportB?.suite?.run_state} />
          </div>
        </div>

        <div className="pair-header">
          <div className="pair-item">
            <span className="pair-label">报告 A</span>
            <strong>{labelA}</strong>
            <p className="pair-detail">{formatDuration(reportA?.suite?.duration_seconds)}</p>
          </div>
          <div className="pair-item">
            <span className="pair-label">报告 B</span>
            <strong>{labelB}</strong>
            <p className="pair-detail">{formatDuration(reportB?.suite?.duration_seconds)}</p>
          </div>
        </div>

        <div className="compare-picker-grid">
          <label className="compare-picker" htmlFor="compare-report-a">
            <span>基准报告 A</span>
          <select id="compare-report-a" value={labelA} onChange={(event) => onChangeA(event.target.value)}>
            {items?.map((run) => (
              <option key={run.filename} value={run.filename}>
                {run.filename}
              </option>
            ))}
          </select>
          </label>
          <label className="compare-picker" htmlFor="compare-report-b">
            <span>对比报告 B</span>
          <select id="compare-report-b" value={labelB} onChange={(event) => onChangeB(event.target.value)}>
            {items?.map((run) => (
              <option key={run.filename} value={run.filename}>
                {run.filename}
              </option>
            ))}
          </select>
          </label>
        </div>
      </article>

      <ConfigComparePanel pairs={matched} reportA={reportA} reportB={reportB} />

                  <div className="chart-grid">
        {comparisonMetrics.map((metric) => {
          const valuesA = matched.map((pair) => caseMetric(pair.entryA, metric.key));
          const valuesB = matched.map((pair) => caseMetric(pair.entryB, metric.key));
          return (
            <article key={metric.key} className="panel chart-panel">
              <div className="panel-head">
                <h3>
                  {metric.label}
                  <span className="unit">{metric.unit}</span>
                </h3>
              </div>
              <LineChart
                labels={labels}
                series={[
                  { name: "A", color: "chart-color-blue", values: valuesA },
                  { name: "B", color: "chart-color-teal", values: valuesB },
                ]}
                unit=""
              />
            </article>
          );
        })}
      </div>

      <div className="compare-cases">
        <CaseComparisonPanel pairs={pairs} />
      </div>
    </section>
  );
}

export default ComparePanel;
