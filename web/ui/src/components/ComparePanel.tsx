import { formatDate, formatDuration, formatNumber } from "../format";
import {
  caseMetric,
  caseLabel,
  metricComparisonRows,
  metricGroups,
  pairCases,
  type CasePair,
} from "../metrics";
import { LineChart } from "./charts";
import StatusBadge from "./StatusBadge";
import type { Report } from "../types";
import type { Run } from "../types";
import type { Case } from "../types";

const comparisonMetrics = [
  { key: "p50_ttft", label: "TTFT P50", unit: "s", color: "chart-color-purple" },
  { key: "p50_tpot", label: "TPOT P50", unit: "s", color: "chart-color-green" },
  { key: "overall_throughput", label: "吞吐量", unit: "tok/s", color: "chart-color-blue" },
  { key: "goodput_pct", label: "Goodput", unit: "%", color: "chart-color-amber" },
] as const;

function formatMetric(value: number | null, unit: string): string {
  if (value === null) return "—";
  return [formatNumber(value, 3), unit].filter(Boolean).join(" ");
}

function formatSignedDelta(value: number | null, unit: string): string {
  if (value === null) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${formatNumber(Math.abs(value), 3)}${unit ? ` ${unit}` : ""}`;
}

function formatSignedPercent(value: number | null): string {
  if (value === null) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${formatNumber(Math.abs(value), 1)}%`;
}

type PairEntries = { entryA?: Case; entryB?: Case };

type CasePairKey = keyof Pick<CasePair, "entryA" | "entryB">;

function metricBarWidth(value: number | null, max: number): string {
  const ratio = value === null || max <= 0 ? 0 : (value / max) * 100;
  return `${Math.min(Math.max(ratio, 0), 100)}%`;
}

function CaseMetricCell({ casePair, entryKey, tone, title, metricKey, unit, max }: {
  casePair: PairEntries;
  entryKey: CasePairKey;
  tone: "a" | "b";
  title: string;
  metricKey: string;
  unit: string;
  max: number;
}) {
  const value = caseMetric(casePair[entryKey], metricKey);
  return (
    <div className="case-pair-row">
      <span className={`case-pair-dot ${tone}`} aria-hidden="true">{title}</span>
      <span className="case-pair-value" title={`${title}: ${formatMetric(value, unit)}`}>
        <span className="case-pair-track">
          <span
            className={`case-pair-bar ${tone}`}
            style={{ width: metricBarWidth(value, max) }}
            aria-hidden="true"
          />
        </span>
        <span>{formatMetric(value, unit)}</span>
      </span>
    </div>
  );
}

function maxMetricValue(pairs: PairEntries[], key: string): number {
  const values = pairs.flatMap((pair) => [
    caseMetric(pair.entryA, key),
    caseMetric(pair.entryB, key),
  ]).filter((value): value is number => value !== null && Number.isFinite(value) && value >= 0);
  return values.length ? Math.max(...values, 0) : 1;
}

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
  const deltaRows = metricComparisonRows(matched);
  const labels = matched.map((pair) => caseLabel(pair.entryA, 0) || caseLabel(pair.entryB, 0));
  const ttftMax = maxMetricValue(pairs, "p50_ttft");
  const tpotMax = maxMetricValue(pairs, "p50_tpot");
  const throughputMax = maxMetricValue(pairs, "overall_throughput");

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

      <div className="metric-delta">
        <div className="panel-head">
          <h3>指标差异一览</h3>
          <span>{matched.length} / {pairs.length} 个用例对齐</span>
        </div>
            <div className="table-wrap">
          <table className="data-table compare-metric-table">
            <thead>
              <tr>
                <th>指标</th>
                <th>报告 A</th>
                <th>报告 B</th>
                <th>变化（B − A）</th>
                <th>相对变化</th>
              </tr>
            </thead>
            <tbody>
              {metricGroups.map((group) => (
                <>
                  <tr key={group.name} className="metric-group-header">
                    <th colSpan={5}>{group.name}</th>
                  </tr>
                  {deltaRows
                    .filter((row) => group.options.some((option) => option.key === row.option.key))
                    .map((row) => (
                      <tr key={row.option.key}>
                        <td>{row.option.label}</td>
                        <td>{formatMetric(row.valueA, row.option.unit)}</td>
                        <td>{formatMetric(row.valueB, row.option.unit)}</td>
                        <td>{formatSignedDelta(row.delta, row.option.unit)}</td>
                        <td>{formatSignedPercent(row.deltaPct)}</td>
                      </tr>
                    ))}
                </>
              ))}
            </tbody>
          </table>
        </div>
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
        <section className="panel table-panel">
          <div className="panel-head">
            <h3>用例对照</h3>
            <span>{pairs.length} 对</span>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>用例</th>
                  <th>状态 A</th>
                  <th>状态 B</th>
                  <th>TTFT P50 A</th>
                  <th>TTFT P50 B</th>
                  <th>TPOT P50 A</th>
                  <th>TPOT P50 B</th>
                  <th>吞吐量 A</th>
                  <th>吞吐量 B</th>
                </tr>
              </thead>
              <tbody>
                {pairs.map((pair) => (
                  <tr key={pair.key}>
                    <td>{pair.label}</td>
                    <td><StatusBadge status={pair.statusA} /></td>
                    <td><StatusBadge status={pair.statusB} /></td>
                    <td className="case-pair-cell">
                      <CaseMetricCell casePair={pair} entryKey="entryA" tone="a" title="A" metricKey="p50_ttft" unit="s" max={ttftMax} />
                    </td>
                    <td className="case-pair-cell">
                      <CaseMetricCell casePair={pair} entryKey="entryB" tone="b" title="B" metricKey="p50_ttft" unit="s" max={ttftMax} />
                    </td>
                    <td className="case-pair-cell">
                      <CaseMetricCell casePair={pair} entryKey="entryA" tone="a" title="A" metricKey="p50_tpot" unit="s" max={tpotMax} />
                    </td>
                    <td className="case-pair-cell">
                      <CaseMetricCell casePair={pair} entryKey="entryB" tone="b" title="B" metricKey="p50_tpot" unit="s" max={tpotMax} />
                    </td>
                    <td className="case-pair-cell">
                      <CaseMetricCell casePair={pair} entryKey="entryA" tone="a" title="A" metricKey="overall_throughput" unit="tok/s" max={throughputMax} />
                    </td>
                    <td className="case-pair-cell">
                      <CaseMetricCell casePair={pair} entryKey="entryB" tone="b" title="B" metricKey="overall_throughput" unit="tok/s" max={throughputMax} />
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
