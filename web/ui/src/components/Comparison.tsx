import { useMemo, useState } from "react";
import { finiteMean, formatNumber, metric } from "../format";
import type { Report } from "../types";

type MetricDefinition = {
  key: string;
  label: string;
  unit?: string;
  direction: "higher" | "lower";
};

type Change = "positive" | "negative" | "neutral";

const metrics = [
  { key: "p50_ttft", label: "TTFT P50", unit: "s", direction: "lower" },
  { key: "p99_ttft", label: "TTFT P99", unit: "s", direction: "lower" },
  { key: "p50_tpot", label: "TPOT P50", unit: "s", direction: "lower" },
  { key: "p99_tpot", label: "TPOT P99", unit: "s", direction: "lower" },
  { key: "p50_e2e", label: "E2E P50", unit: "s", direction: "lower" },
  { key: "p99_e2e", label: "E2E P99", unit: "s", direction: "lower" },
  { key: "overall_throughput", label: "整体吞吐", unit: "tok/s", direction: "higher" },
  { key: "goodput_pct", label: "Goodput", unit: "%", direction: "higher" },
] as const;

type MetricChoice = (typeof metrics)[number]["key"];

interface ReportComparisonProps {
  labelA: string;
  labelB: string;
  reportA: Report | null;
  reportB: Report | null;
}

export default function ReportComparison({ labelA, labelB, reportA, reportB }: ReportComparisonProps) {
  const [metricChoice, setMetricChoice] = useState<MetricChoice>("p50_ttft");

  const rows = useMemo(() => {
    return metrics.map((row) => {
      const casesA = reportA?.cases ?? [];
      const casesB = reportB?.cases ?? [];
      return {
        ...row,
        valueA: finiteMean(casesA.map((entry) => metric(entry, row.key))),
        valueB: finiteMean(casesB.map((entry) => metric(entry, row.key))),
      };
    });
  }, [reportA, reportB]);

  const selected = useMemo(() => {
    const casesA = reportA?.cases ?? [];
    const casesB = reportB?.cases ?? [];
    const mapA = new Map(casesA.map((entry) => [entry.name, entry]));
    const mapB = new Map(casesB.map((entry) => [entry.name, entry]));
    return [...mapA.keys()].map((name) => {
      const left = mapA.get(name);
      const right = mapB.get(name);
      return {
        name,
        left,
        right,
        leftValue: metric(left, metricChoice),
        rightValue: metric(right, metricChoice),
      };
    }).filter((item) => item.left && item.right);
  }, [reportA, reportB, metricChoice]);

  return (
    <section className="panel compare-panel">
      <div className="panel-head">
        <h2>对比</h2>
        <small>{rows.filter((row) => row.valueA !== null && row.valueB !== null).length} metrics</small>
      </div>
      <div className="compare-grid">
        <div className="compare-block">
          <label htmlFor="metricChoice">基准指标</label>
          <select id="metricChoice" onChange={(event) => setMetricChoice(event.target.value as MetricChoice)} value={metricChoice}>
            {metrics.map((item) => (
              <option key={item.key} value={item.key}>{item.label}</option>
            ))}
          </select>
        </div>
        <div className="compare-block">
          <label htmlFor="reportA">报告 A</label>
          <code>{labelA}</code>
        </div>
        <div className="compare-block">
          <label htmlFor="reportB">报告 B</label>
          <code>{labelB}</code>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>指标</th>
              <th>A</th>
              <th>B</th>
              <th>差异</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const delta = row.valueA !== null && row.valueB !== null ? row.valueB - row.valueA : null;
              const change = getChange(row, delta);
              return (
                <tr key={row.key}>
                  <td>{row.label}</td>
                  <td>{formatNumber(row.valueA)}</td>
                  <td>{formatNumber(row.valueB)}</td>
                  <td className={`change ${change}`}>{formatChange(delta, row.unit)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="case-table-wrap table-wrap">
        <table>
          <thead>
            <tr>
              <th>用例</th>
              <th>A</th>
              <th>B</th>
              <th>变化</th>
            </tr>
          </thead>
          <tbody>
            {selected.map((item) => (
              <tr key={item.name}>
                <td><span className="case-name">{item.name}</span></td>
                <td>{formatNumber(item.leftValue)}</td>
                <td>{formatNumber(item.rightValue ?? 0)}</td>
                <td>{formatChange((item.rightValue ?? 0) - (item.leftValue ?? 0), metrics.find((row) => row.key === metricChoice)?.unit)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function getChange(row: MetricDefinition, delta: number | null): Change {
  if (delta === null) return "neutral";
  return row.direction === "higher" ? (delta > 0 ? "positive" : "negative") : (delta < 0 ? "positive" : "negative");
}

function formatChange(delta: number | null, unit?: string): string {
  if (!Number.isFinite(delta)) return "—";
  if (delta === null) return "—";
  if ((delta ?? 0) === 0) return "持平";
  return `${delta > 0 ? "+" : ""}${formatNumber(delta, 2)}${unit ? ` ${unit}` : ""}`;
}
