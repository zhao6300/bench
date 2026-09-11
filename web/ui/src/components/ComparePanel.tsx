import { useState } from "react";
import { formatDate, formatDuration, formatNumber } from "../format";
import {
  caseMetric,
  caseLabel,
  metricComparisonRows,
  metricGroups,
  metricOptions,
  pairCases,
} from "../metrics";
import { LineChart, MetricDeltaBars } from "./charts";
import StatusBadge from "./StatusBadge";
import type { Report } from "../types";
import type { Run } from "../types";

const comparisonMetrics = [
  { key: "p50_ttft", label: "TTFT P50", unit: "s", color: "chart-color-purple" },
  { key: "p50_tpot", label: "TPOT P50", unit: "s", color: "chart-color-green" },
  { key: "overall_throughput", label: "吞吐量", unit: "tok/s", color: "chart-color-blue" },
  { key: "goodput_pct", label: "Goodput", unit: "%", color: "chart-color-amber" },
] as const;

export function ComparePanel({ reportA, reportB, labelA, labelB, items }: {
  reportA: Report | null;
  reportB: Report | null;
  labelA: string;
  labelB: string;
  items?: Run[];
}) {
  const [compareReportA, setCompareReportA] = useState(labelA);
  const [compareReportB, setCompareReportB] = useState(labelB);
  const pairs = pairCases(reportA, reportB);
  const matched = pairs.filter((pair) => pair.matched);
  const deltaRows = metricComparisonRows(reportA, reportB);
  const metricDeltaRows = deltaRows.map((row) => ({
    label: row.option.label,
    unit: row.option.unit,
    valueA: row.valueA,
    valueB: row.valueB,
    delta: row.delta,
    deltaPct: row.deltaPct,
    tone: row.tone,
  }));
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
          <select id="compare-report-a" value={compareReportA} onChange={(event) => setCompareReportA(event.target.value)}>
            {items?.map((run) => (
              <option key={run.filename} value={run.filename}>
                {run.filename}
              </option>
            ))}
          </select>
          </label>
          <label className="compare-picker" htmlFor="compare-report-b">
            <span>对比报告 B</span>
          <select id="compare-report-b" value={compareReportB} onChange={(event) => setCompareReportB(event.target.value)}>
            {items?.map((run) => (
              <option key={run.filename} value={run.filename}>
                {run.filename}
              </option>
            ))}
          </select>
          </label>
        </div>
      </article>

      <div className="metric-delta">
        <div className="panel-head">
          <h3>指标差异一览</h3>
          <span>{matched.length} / {pairs.length} 个用例对齐</span>
        </div>
        <MetricDeltaBars rows={metricDeltaRows} />
      </div>

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

      <div className="panel-grid">
        <section className="panel metadata-panel">
          <div className="panel-head">
            <h3>指标总结</h3>
            <span>{metricOptions.length} 项</span>
          </div>
          <div className="metric-group-list">
            {metricGroups.map((group) => (
              <div key={group.name} className="metric-group-block">
                <h4>{group.name}</h4>
                <ul>
            {deltaRows.filter((row) => group.options.some((option) => option.key === row.option.key))
                    .map((row) => (
                      <li key={row.option.key}>
                        <div>{row.option.label}</div>
                        <div>
                          {formatNumber(row.valueA)}
                          {" → "}
                          {formatNumber(row.valueB)}
                        </div>
                      </li>
                    ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        <section className="table-panel">
          <div className="panel-head">
            <h3>用例对照</h3>
            <span>{pairs.length} 对</span>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>用例</th>
                  <th>状态</th>
                  <th>TTFT P50</th>
                  <th>TPOT P50</th>
                  <th>吞吐量</th>
                </tr>
              </thead>
              <tbody>
                {pairs.map((pair) => (
                  <tr key={pair.key}>
                    <td>{pair.label}</td>
                    <td>
                      <StatusBadge status={pair.statusA} />
                      <StatusBadge status={pair.statusB} />
                    </td>
                    <td>
                      {formatNumber(caseMetric(pair.entryA, "p50_ttft"))}
                      {" → "}
                      {formatNumber(caseMetric(pair.entryB, "p50_ttft"))}
                    </td>
                    <td>
                      {formatNumber(caseMetric(pair.entryA, "p50_tpot"))}
                      {" → "}
                      {formatNumber(caseMetric(pair.entryB, "p50_tpot"))}
                    </td>
                    <td>
                      {formatNumber(caseMetric(pair.entryA, "overall_throughput"))}
                      {" → "}
                      {formatNumber(caseMetric(pair.entryB, "overall_throughput"))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </section>
  );
}

export default ComparePanel;
