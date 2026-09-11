import { useState } from "react";
import { finiteMean, formatDate, formatNumber } from "../format";
import type { Case, Report } from "../types";

type MetricDirection = "higher" | "lower" | "neutral";

interface MetricOption {
  key: string;
  label: string;
  unit: string;
  direction: MetricDirection;
}

const metricGroups: { name: string; options: MetricOption[] }[] = [
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

const flatMetricOptions = metricGroups.flatMap((group) => group.options);

interface ComparisonProps {
  reportA: Report | null;
  reportB: Report | null;
  labelA: string;
  labelB: string;
}

export default function Comparison({ reportA, reportB, labelA, labelB }: ComparisonProps) {
  const [choice, setChoice] = useState(flatMetricOptions[0].key);
  const selectedMetric =
    flatMetricOptions.find((option) => option.key === choice) ?? flatMetricOptions[0];
  const casesA = reportA?.cases ?? [];
  const casesB = reportB?.cases ?? [];
  const aggregationA = casesByName(casesA);
  const aggregationB = casesByName(casesB);
  const matchedCases = [...aggregationA.values()].filter((caseA) => {
    const name = caseA.name || caseA.id;
    return name ? aggregationB.has(name) : false;
  });
  const onlyA = [...aggregationA.entries()].filter(([name]) => !aggregationB.has(name));
  const onlyB = [...aggregationB.entries()].filter(([name]) => !aggregationA.has(name));

  const summaryRows = metricGroups.map((group) => ({
    name: group.name,
    rows: group.options.map((option) => ({
      option,
      valueA: candidateMean(casesA.map((entry) => metricValue(entry, option.key))),
      valueB: candidateMean(casesB.map((entry) => metricValue(entry, option.key))),
    })),
  }));

  const caseRows = [...aggregationA.entries()]
    .filter(([name]) => aggregationB.has(name))
    .map(([name, entryA]) => {
      const entryB = aggregationB.get(name)!;
      return {
        name,
        statusA: entryA.status ?? "",
        statusB: entryB.status ?? "",
        valueA: metricValue(entryA, selectedMetric.key),
        valueB: metricValue(entryB, selectedMetric.key),
      };
    });
  const totalCases = casesA.length + casesB.length;
  const matchRate = totalCases ? (matchedCases.length * 2 * 100) / totalCases : 0;

  return (
    <section className="panel compare-panel" aria-label="指标对比">
      <div className="panel-head">
        <h2>指标对比</h2>
        <small>{matchedCases.length} / {Math.max(casesA.length, casesB.length)} 用例自动对齐</small>
      </div>

      <div className="compare-grid">
        <label>
          <span>单用例指标</span>
          <select value={choice} onChange={(event) => setChoice(event.target.value)}>
            {metricGroups.map((group) => (
              <optgroup key={group.name} label={group.name}>
                {group.options.map((option) => (
                  <option key={option.key} value={option.key}>
                    {option.label}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>
        <div className="tag">
          <span>报告 A · {formatDate(reportA?.suite?.started_at)}</span>
          <strong>{truncateLabel(labelA)}</strong>
        </div>
        <div className="tag">
          <span>报告 B · {formatDate(reportB?.suite?.started_at)}</span>
          <strong>{truncateLabel(labelB)}</strong>
        </div>
      </div>

      <div className="match-note">
        对齐率 {formatNumber(matchRate, 1)}%；{onlyA.length + onlyB.length === 0 ? "两份报告全部用例可配对。" : "缺失或新增用例列在下方的对齐明细中。"}
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>指标</th>
              <th>A 均值</th>
              <th>B 均值</th>
              <th>变化</th>
              <th>变化率</th>
            </tr>
          </thead>
          {summaryRows.map((group) => (
            <tbody key={group.name}>
              <tr className="group-row">
                <td colSpan={5}>{group.name}</td>
              </tr>
              {group.rows.map((row) => {
                const delta = numericDelta(row.valueA, row.valueB);
                return (
                  <tr key={row.option.key}>
                    <td>{row.option.label}</td>
                    <td>{formatNumber(row.valueA)}{row.option.unit ? ` ${row.option.unit}` : ""}</td>
                    <td>{formatNumber(row.valueB)}{row.option.unit ? ` ${row.option.unit}` : ""}</td>
                    <td className={`change ${getTone(row.option.direction, delta)}`}>
                      {getDeltaText(row.option.direction, row.option.unit, delta)}
                    </td>
                    <td className={`change ${getTone(row.option.direction, delta)}`}>
                      {formatRelativeChange(row.valueA, row.valueB)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          ))}
        </table>
      </div>

      <div className="panel-head">
        <h2>用例级 {selectedMetric.label} 对比</h2>
        <small>按用例名称对齐</small>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>用例</th>
              <th>状态</th>
              <th>A</th>
              <th>B</th>
              <th>变化</th>
              <th>变化率</th>
            </tr>
          </thead>
          <tbody>
            {caseRows.map((row) => {
              const delta = numericDelta(row.valueA, row.valueB);
              const statusChanged = row.statusA !== row.statusB;
              return (
                <tr key={row.name}>
                  <td><span className="case-name">{row.name}</span></td>
                  <td>
                    {statusChanged ? `${row.statusA || "无"} → ${row.statusB || "无"}` : row.statusA || "无"}
                  </td>
                  <td>{formatNumber(row.valueA)}{selectedMetric.unit ? ` ${selectedMetric.unit}` : ""}</td>
                  <td>{formatNumber(row.valueB)}{selectedMetric.unit ? ` ${selectedMetric.unit}` : ""}</td>
                  <td className={`change ${getTone(selectedMetric.direction, delta)}`}>
                    {getDeltaText(selectedMetric.direction, selectedMetric.unit, delta)}
                  </td>
                  <td className={`change ${getTone(selectedMetric.direction, delta)}`}>
                    {formatRelativeChange(row.valueA, row.valueB)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {onlyA.length || onlyB.length ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>对齐情况</th>
                <th>用例</th>
              </tr>
            </thead>
            <tbody>
              {onlyA.map(([name]) => (
                <tr key={`a-${name}`}>
                  <td>仅 A</td>
                  <td><span className="case-name">{name}</span></td>
                </tr>
              ))}
              {onlyB.map(([name]) => (
                <tr key={`b-${name}`}>
                  <td>仅 B</td>
                  <td><span className="case-name">{name}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}

function casesByName(cases: Case[]): Map<string, Case> {
  const map = new Map<string, Case>();
  for (const [index, entry] of cases.entries()) {
    map.set(entry.name || entry.id || `用例 ${index + 1}`, entry);
  }
  return map;
}

function metricValue(entry: Case | undefined, key: string): number | null {
  const value = entry?.result?.metrics?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function candidateMean(values: (number | null | undefined)[]): number | null {
  return finiteMean(values);
}

function numericDelta(valueA: number | null, valueB: number | null): number | null {
  return valueA === null || valueB === null ? null : valueB - valueA;
}

function formatRelativeChange(valueA: number | null, valueB: number | null): string {
  if (valueA === null || valueB === null || valueA === 0) return "—";
  const change = ((valueB - valueA) / Math.abs(valueA)) * 100;
  return Number.isFinite(change) ? `${change > 0 ? "+" : ""}${change.toFixed(1)}%` : "—";
}

function getTone(direction: MetricDirection, delta: number | null): "neutral" | "positive" | "negative" {
  if (delta === null || delta === 0 || direction === "neutral") return "neutral";
  const improved = direction === "higher" ? delta > 0 : delta < 0;
  return improved ? "positive" : "negative";
}

function getDeltaText(direction: MetricDirection, unit: string, delta: number | null): string {
  if (delta === null) return "—";
  if (delta === 0) return "持平";
  const improved = direction === "higher" ? delta > 0 : delta < 0;
  const marker = improved ? "↑" : "↓";
  return `${marker} ${formatNumber(Math.abs(delta), 1)}${unit ? ` ${unit}` : ""}`;
}

function truncateLabel(label: string): string {
  return label.length > 46 ? `${label.slice(-46)}…` : label;
}
