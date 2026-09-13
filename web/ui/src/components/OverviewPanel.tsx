import {
  caseConcurrency,
  caseLabel,
  caseMetric,
  caseParamNumber,
  caseStringParam,
  environmentFacts,
  getSummaryCards,
  caseKey,
  statusSegments,
} from "../metrics";
import { useState } from "react";
import { formatDate, formatDuration, formatNumber } from "../format";
import CaseDetailPanel from "./CaseDetailPanel";
import { LineChart } from "./charts";
import StatusBadge from "./StatusBadge";
import type { Report } from "../types";

type OverviewMetricKey = "p50_ttft" | "p50_tpot" | "overall_throughput" | "goodput_pct";

const overviewMetricLabels: Record<OverviewMetricKey, string> = {
  p50_ttft: "TTFT P50",
  p50_tpot: "TPOT P50",
  overall_throughput: "吞吐量",
  goodput_pct: "Goodput",
};

const overviewChartColors: Record<OverviewMetricKey, string> = {
  p50_ttft: "chart-color-purple",
  p50_tpot: "chart-color-green",
  overall_throughput: "chart-color-blue",
  goodput_pct: "chart-color-amber",
};

const tableZoomOptions = [1, 1.25, 1.5, 1.75, 2, 2.5, 3];

export function OverviewPanel({ report }: { report: Report | null }) {
  const reportData = report ?? { cases: [] };
  const cases = (reportData.cases ?? []).filter((caseEntry) => caseMetric(caseEntry, "p50_ttft") !== null);
  const summaryCards = getSummaryCards(reportData);
  const statusString = statusSegments(reportData);
  const environment = environmentFacts(reportData);
  const [tableZoom, setTableZoom] = useState(1);
  const [selectedCase, setSelectedCase] = useState<typeof cases[number] | null>(null);

  if (selectedCase) {
    return <CaseDetailPanel caseEntry={selectedCase} onClose={() => setSelectedCase(null)} />;
  }

  return (
    <section className="overview">
      <article className="panel hero-panel">
        <div className="panel-head">
          <div>
            <h2>{report?.suite?.name ?? "执行概览"}</h2>
            <p className="hero-subtitle">
              {formatDate(report?.suite?.started_at)} → {formatDate(report?.suite?.finished_at)}
            </p>
          </div>
          <div className="hero-actions">
            <StatusBadge status={report?.suite?.run_state} />
            {cases[0] ? (
              <StatusBadge status={cases[0].status} />
            ) : null}
          </div>
        </div>
        {statusString.some((item) => item.value > 0) ? (
          <div className="status-strip">
            {statusString.map((segment) => (
              <span
                key={segment.key}
                className={`status-segment ${segment.key}`}
                style={{ flexGrow: segment.value }}
                title={`${segment.label} ${segment.value}`}
              >
                {segment.label} {segment.value}
              </span>
            ))}
          </div>
        ) : null}
        <div className="summary-grid">
          {summaryCards.map((card) => (
            <div key={card.label} className={`metric-card ${card.tone}`}>
              <span className="metric-card-label">{card.label}</span>
              <strong className="metric-card-value">{card.value}</strong>
              <p className="metric-card-detail">{card.detail}</p>
            </div>
          ))}
        </div>
      </article>

      <div className="chart-grid">
        {Object.keys(overviewMetricLabels).map((metricKey) => {
          const key = metricKey as OverviewMetricKey;
          const values = cases.map((caseEntry) => caseMetric(caseEntry, key));
          if (!values.some((value) => value !== null)) return null;
          const unit = key === "overall_throughput" ? "tok/s" : key === "p50_ttft" || key === "p50_tpot" ? "s" : "%";
          return (
            <article key={key} className="panel chart-panel">
              <div className="panel-head">
                <h3>
                  {overviewMetricLabels[key]}
                  <span className="unit">{unit}</span>
                </h3>
              </div>
              <LineChart
                labels={cases.map((caseEntry, index) => caseLabel(caseEntry, index))}
                series={[{ name: overviewMetricLabels[key], color: overviewChartColors[key], values }]}
                unit={unit}
              />
            </article>
          );
        })}
      </div>

      <div className="panel-grid">
        <section className="panel table-panel" style={{ ["--table-zoom" as string]: tableZoom } as React.CSSProperties}>
          <div className="panel-head table-toolbar">
            <h3>用例结果</h3>
            <label className="table-zoom" htmlFor="overview-table-zoom">
              <span>文字大小</span>
              <select id="overview-table-zoom" value={tableZoom} onChange={(event) => setTableZoom(Number(event.target.value))}>
                {tableZoomOptions.map((value) => (
                  <option key={value} value={value}>
                    {Math.round(value * 100)}%
                  </option>
                ))}
              </select>
            </label>
            <span>{cases.length} 个</span>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>状态</th>
                  <th>用例</th>
                  <th>模型</th>
                  <th>负载</th>
                  <th>并发</th>
                  <th>TTFT P50</th>
                  <th>TPOT P50</th>
                  <th>吞吐量</th>
                  <th>Goodput</th>
                </tr>
              </thead>
              <tbody>
                {cases.map((caseEntry, index) => (
                  <tr
                    key={`${caseKey(caseEntry)}-${index}`}
                    className={caseEntry.status === "passed" ? "case-row passed" : "case-row"}
                    tabIndex={caseEntry.status === "passed" ? 0 : undefined}
                    onClick={
                      caseEntry.status === "passed"
                        ? () => setSelectedCase(caseEntry)
                        : undefined
                    }
                    onKeyDown={
                      caseEntry.status === "passed"
                        ? (event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              setSelectedCase(caseEntry);
                            }
                          }
                        : undefined
                    }
                  >
            <td><StatusBadge status={caseEntry.status} /></td>
            <td className="case-cell">{caseEntry.name ?? "—"}</td>
                    <td className="case-select">{caseStringParam(caseEntry, "model") || "—"}</td>
                    <td className="case-select">
                      {typeof caseParamNumber(caseEntry, "random_input_len") === "number"
                        ? `${caseParamNumber(caseEntry, "random_input_len")} / ${caseParamNumber(caseEntry, "random_output_len")} tok`
                        : "—"}
                    </td>
                    <td>{caseConcurrency(caseEntry) ?? "—"}</td>
                    <td>{formatNumber(caseMetric(caseEntry, "p50_ttft"))}</td>
                    <td>{formatNumber(caseMetric(caseEntry, "p50_tpot"))}</td>
                    <td>{formatNumber(caseMetric(caseEntry, "overall_throughput"))}</td>
                    <td>{formatNumber(caseMetric(caseEntry, "goodput_pct"), 1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel environment-panel">
          <div className="panel-head">
            <h3>运行环境</h3>
            <span>{formatDuration(report?.suite?.duration_seconds)}</span>
          </div>
          <dl className="environment-list">
            {environment.map((item) => (
              <div key={item.label}>
                <dt>{item.label}</dt>
                <dd>{item.value}</dd>
              </div>
            ))}
          </dl>
          {report?.suite?.description ? (
            <p className="description">{report.suite.description}</p>
          ) : null}
        </section>
      </div>
    </section>
  );
}

export default OverviewPanel;
