import { useEffect, useMemo, useState } from "react";
import "./index.css";
import { concurrency, finiteMean, formatDate, formatNumber, metric } from "./format";
import { Report, Run } from "./types";

type View = "overview" | "compare";
type MetricDirection = "higher" | "lower";
type ReportPair = { reportA: Report | null; reportB: Report | null };

type MetricOption = {
  key: string;
  label: string;
  unit: string;
  direction: MetricDirection;
};

const metricOptions: MetricOption[] = [
  { key: "p50_ttft", label: "TTFT P50", unit: "s", direction: "lower" },
  { key: "p99_ttft", label: "TTFT P99", unit: "s", direction: "lower" },
  { key: "p50_tpot", label: "TPOT P50", unit: "s", direction: "lower" },
  { key: "p99_tpot", label: "TPOT P99", unit: "s", direction: "lower" },
  { key: "overall_throughput", label: "吞吐量", unit: "tok/s", direction: "higher" },
  { key: "goodput_pct", label: "Goodput", unit: "%", direction: "higher" },
];

export default function App() {
  const [view, setView] = useState<View>("overview");
  const [runs, setRuns] = useState([] as Run[]);
  const [report, setReport] = useState<Report | null>(null);
  const [compareLeft, setCompareLeft] = useState("");
  const [compareRight, setCompareRight] = useState("");
  const [comparisonReports, setComparisonReports] = useState<ReportPair>({ reportA: null, reportB: null });
  const [selectedReport, setSelectedReport] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const response = await fetch("/api/runs");
        const payload = (await response.json()) as Run[];
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        if (cancelled) return;
        setRuns(payload);
        setSelectedReport(payload[0]?.filename ?? "");
        setCompareLeft(payload[0]?.filename ?? "");
        setCompareRight(payload[1]?.filename ?? "");
      } catch (failure) {
        if (!cancelled) setError(failure instanceof Error ? failure.message : "报告列表加载失败");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    if (!selectedReport) {
      setReport(null);
      return () => {
        cancelled = true;
      };
    }

    void (async () => {
      try {
        const reportData = await fetchReport(selectedReport);
        if (!cancelled) {
          setReport(reportData);
          setError("");
        }
      } catch (failure) {
        if (!cancelled) setError(failure instanceof Error ? failure.message : "报告加载失败");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedReport]);

  useEffect(() => {
    let cancelled = false;

    if (!compareLeft || !compareRight) {
      setComparisonReports({ reportA: null, reportB: null });
      return () => {
        cancelled = true;
      };
    }

    void (async () => {
      try {
        const [reportA, reportB] = await Promise.all([fetchReport(compareLeft), fetchReport(compareRight)]);
        if (!cancelled) {
          setComparisonReports({ reportA, reportB });
          setError("");
        }
      } catch (failure) {
        if (!cancelled) setError(failure instanceof Error ? failure.message : "对比报告加载失败");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [compareLeft, compareRight]);

  const summaryCards = useMemo(() => getSummaryCards(report), [report]);
  const detailCards = useMemo(() => getDetailCards(report), [report]);
  const chartPoints = useMemo(() => getChartPoints(report), [report]);

  async function fetchReport(filename: string): Promise<Report> {
    const response = await fetch(`/runs/${encodeURIComponent(filename)}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return (await response.json()) as Report;
  }

  return (
    <div className="app">
      <header className="topbar" role="banner">
        <div className="brand">
          <div className="mark">LB</div>
          <div className="brand-copy">
            <h1>LLM Benchmark</h1>
            <p>清晰查看推理基准报告</p>
          </div>
        </div>

        <nav className="view-switch" aria-label="页面切换">
          <button type="button" className={view === "overview" ? "active" : ""} onClick={() => setView("overview")}>
            概览
          </button>
          <button type="button" className={view === "compare" ? "active" : ""} onClick={() => setView("compare")}>
            对比
          </button>
        </nav>

        <label className="run-picker" htmlFor="run-picker">
          <span>当前报告</span>
          <select
            id="run-picker"
            value={selectedReport}
            onChange={(event) => setSelectedReport(event.target.value)}
          >
            {runs.map((run) => (
              <option key={run.filename} value={run.filename}>
                {run.filename}
              </option>
            ))}
          </select>
        </label>
      </header>

      {view === "overview" ? (
        <main className="overview" aria-label="报告概览">
          <section className="summary-grid" aria-label="报告摘要">
            {summaryCards.map((item) => (
              <article className={`card metric ${item.tone}`} key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
                <small>{item.detail}</small>
              </article>
            ))}
          </section>

          <section className="panel-grid">
            <section className="panel" aria-label="用例列表">
              <div className="panel-head">
                <h2>用例</h2>
                <small>{report?.cases?.length ?? 0} 个</small>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>名称</th>
                      <th>并发</th>
                      <th>状态</th>
                      <th>TTFT</th>
                      <th>吞吐</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(report?.cases ?? []).map((entry, index) => (
                      <tr key={`${entry.id ?? entry.name ?? "case"}-${index}`}>
                        <td><span className="case-name">{entry.name ?? "未命名"}</span></td>
                        <td>{concurrency(entry) === null ? "—" : concurrency(entry)}</td>
                        <td><span className={`status-badge ${getStatus(entry.status)}`}>{entry.status ?? "未知"}</span></td>
                        <td>{formatNumber(metric(entry, "p50_ttft"))}</td>
                        <td>{formatNumber(metric(entry, "overall_throughput"))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="detail-column" aria-label="关键指标">
              <section className="mini-grid">
                {detailCards.map((item) => (
                  <article className="card" key={item.title}>
                    <span>{item.title}</span>
                    <strong>{item.value}</strong>
                    <small>平均值</small>
                  </article>
                ))}
              </section>

              <section className="panel chart-panel" aria-label="并发趋势">
                <div className="panel-head">
                  <h2>并发曲线</h2>
                  <small>{chartPoints.length} 点</small>
                </div>
                <Chart points={chartPoints} unit="s" />
              </section>
            </section>
          </section>
        </main>
      ) : (
        <main className="compare" aria-label="报告对比">
          <section className="panel" aria-label="对比设置">
            <div className="panel-head">
              <h2>选择报告</h2>
              <small>按用例名称自动匹配</small>
            </div>
            <div className="compare-grid">
              <label>
                <span>基准报告 A</span>
                <select value={compareLeft} onChange={(event) => setCompareLeft(event.target.value)}>
                  {runs.map((run) => (
                    <option key={run.filename} value={run.filename}>
                      {run.filename}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>对比报告 B</span>
                <select value={compareRight} onChange={(event) => setCompareRight(event.target.value)}>
                  {runs.map((run) => (
                    <option key={run.filename} value={run.filename}>
                      {run.filename}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </section>

          <Comparison
            reports={comparisonReports}
            labelA={compareLeft}
            labelB={compareRight}
          />
        </main>
      )}

      {loading ? <div className="status">加载中…</div> : null}
      {error ? <div className="status error">{error}</div> : null}
    </div>
  );
}

function Comparison({ labelA, labelB, reports }: {
  labelA: string;
  labelB: string;
  reports: { reportA: Report | null; reportB: Report | null };
}) {
  const casesA = reports.reportA?.cases ?? [];
  const casesB = reports.reportB?.cases ?? [];
  const [choice, setChoice] = useState(metricOptions[0].key);
  const selectedMetric = metricOptions.find((option) => option.key === choice)!;
  const rows = metricOptions.map((option) => ({
    ...option,
    valueA: finiteMean(casesA.map((entry) => metric(entry, option.key))),
    valueB: finiteMean(casesB.map((entry) => metric(entry, option.key))),
  }));

  const casesByNameA = new Map(casesA.filter((entry) => entry.name).map((entry) => [entry.name as string, entry]));
  const casesByNameB = new Map(casesB.filter((entry) => entry.name).map((entry) => [entry.name as string, entry]));
  const matchedNames = [...casesByNameA.keys()].filter((name) => casesByNameB.has(name));
  const matchedCases = matchedNames
    .map((name) => ({
      entryA: casesByNameA.get(name)!,
      entryB: casesByNameB.get(name)!,
      name,
      valueA: metric(casesByNameA.get(name), choice),
      valueB: metric(casesByNameB.get(name), choice),
    }))
    .filter((row) => row.valueA !== null && row.valueB !== null);

  return (
    <section className="panel compare-panel" aria-label="指标对比">
      <div className="panel-head">
        <h2>指标对比</h2>
        <small>按平均值计算</small>
      </div>

      <div className="compare-grid">
        <label>
          <span>对比指标</span>
          <select value={choice} onChange={(event) => setChoice(event.target.value)}>
            {metricOptions.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <div className="tag">
          <span>报告 A</span>
          <strong>{truncateLabel(labelA)}</strong>
        </div>
        <div className="tag">
          <span>报告 B</span>
          <strong>{truncateLabel(labelB)}</strong>
        </div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>指标</th>
              <th>A</th>
              <th>B</th>
              <th>变化</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const delta = row.valueA !== null && row.valueB !== null ? row.valueB - row.valueA : null;
              return (
                <tr key={row.key}>
                  <td>{row.label}</td>
                  <td>{formatNumber(row.valueA)}</td>
                  <td>{formatNumber(row.valueB)}</td>
                  <td className={`change ${getTone(row.direction, delta)}`}>
                    {formatDelta(delta, row.direction, row.unit)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="table-wrap">
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
            {matchedCases.map((row) => {
              const delta = row.valueA !== null && row.valueB !== null ? row.valueB - row.valueA : null;
              return (
                <tr key={row.name}>
                  <td><span className="case-name">{row.name}</span></td>
                  <td>{formatNumber(row.valueA)}</td>
                  <td>{formatNumber(row.valueB)}</td>
                  <td className={`change ${getTone(selectedMetric.direction, delta)}`}>
                    {formatDelta(delta, selectedMetric.direction, selectedMetric.unit)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function getTone(direction: MetricDirection, delta: number | null): "positive" | "negative" | "neutral" {
  if (delta === null || delta === 0) return "neutral";
  const improved = direction === "higher" ? delta > 0 : delta < 0;
  return improved ? "positive" : "negative";
}

function formatDelta(delta: number | null, direction: MetricDirection, unit: string): string {
  if (delta === null) return "—";
  if (delta === 0) return "持平";
  const improved = direction === "higher" ? delta > 0 : delta < 0;
  const marker = improved ? "↑" : "↓";
  return `${marker} ${formatNumber(Math.abs(delta), 1)}${unit ? ` ${unit}` : ""}`;
}

function truncateLabel(label: string): string {
  return label.length > 36 ? `${label.slice(-36)}…` : label;
}

function getStatus(status: string | undefined): string {
  return status === "passed" || status === "failed" || status === "skipped" ? status : "unknown";
}

function getSummaryCards(report: Report | null): { label: string; value: string; detail: string; tone: string }[] {
  const cases = report?.cases ?? [];
  const summary = report?.summary ?? {};
  const total = Number(summary.total);
  const passed = Number(summary.passed);
  const failed = Number(summary.failed);
  const skipped = Number(summary.skipped);
  const totalCases = Number.isFinite(total) ? total : cases.length;
  const passedCases = Number.isFinite(passed) ? passed : cases.filter((entry) => entry.status === "passed").length;
  const failedCases = Number.isFinite(failed) ? failed : cases.filter((entry) => entry.status === "failed").length;
  const skippedCases = Number.isFinite(skipped) ? skipped : cases.filter((entry) => entry.status === "skipped").length;

  return [
    { label: "基准", value: report?.suite?.name ?? "—", detail: formatDate(report?.suite?.started_at), tone: "reference" },
    { label: "通过", value: String(passedCases), detail: totalCases ? `${((passedCases / totalCases) * 100).toFixed(1)}%` : "—", tone: "passed" },
    { label: "失败", value: String(failedCases), detail: `${skippedCases} skipped`, tone: "failed" },
    { label: "运行耗时", value: report?.suite?.duration_seconds ? `${(report.suite.duration_seconds / 60).toFixed(1)} 分钟` : "—", detail: `${totalCases} 用例`, tone: "elapsed" },
    { label: "平均 Prompt", value: formatNumber(finiteMean(cases.map((entry) => metric(entry, "total_prompt_tokens"))), 0), detail: "tokens", tone: "prompt" },
    { label: "平均 Decode", value: formatNumber(finiteMean(cases.map((entry) => metric(entry, "total_generated_tokens"))), 0), detail: "tokens", tone: "completion" },
  ];
}

function getDetailCards(report: Report | null): { title: string; key: string; value: string }[] {
  const cases = report?.cases ?? [];
  const entries = [
    { title: "TTFT P99", key: "p99_ttft" },
    { title: "TPOT P99", key: "p99_tpot" },
    { title: "Goodput", key: "goodput_pct" },
    { title: "E2E P99", key: "p99_e2e" },
  ];
  return entries.map((entry) => ({
    ...entry,
    value: formatNumber(finiteMean(cases.map((caseData) => metric(caseData, entry.key)))),
  }));
}

function getChartPoints(report: Report | null): { x: number; y: number }[] {
  const cases = report?.cases ?? [];
  return cases
    .filter((entry) => metric(entry, "p50_ttft") !== null && concurrency(entry) !== null)
    .map((entry) => ({
      x: concurrency(entry) as number,
      y: metric(entry, "p50_ttft") as number,
    }))
    .sort((left, right) => left.x - right.x);
}

function Chart({ points, unit }: { points: { x: number; y: number }[]; unit: string }) {
  if (!points.length) return <div className="placeholder" />;

  const height = 240;
  const paddingLeft = 56;
  const paddingRight = 24;
  const paddingTop = 28;
  const paddingBottom = 42;
  const chartWidth = 640;
  const charWidth = chartWidth - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;
  const values = points.map((point) => point.y).filter((value) => Number.isFinite(value));
  const bounds = points.map((point) => point.x).filter((value) => Number.isFinite(value));
  const minimumY = Math.min(...values);
  const maximumY = Math.max(...values);
  const minimumX = Math.min(...bounds);
  const maximumX = Math.max(...bounds);
  const yRange = Math.max(maximumY - minimumY, 1e-6);
  const xRange = Math.max(maximumX - minimumX, 1);

  function coordinateX(value: number): number {
    if (points.length === 1) return paddingLeft + charWidth / 2;
    return paddingLeft + ((value - minimumX) / xRange) * charWidth;
  }

  function coordinateY(value: number): number {
    return height - paddingBottom - ((value - minimumY) / yRange) * chartHeight;
  }

  const linePath = points.map((point, index) => `${index ? "L" : "M"}${coordinateX(point.x).toFixed(1)},${coordinateY(point.y).toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${coordinateX(points[points.length - 1].x).toFixed(1)},${height - paddingBottom} L${coordinateX(points[0].x).toFixed(1)},${height - paddingBottom} Z`;
  const firstX = coordinateX(points[0].x);
  const lastX = coordinateX(points[points.length - 1].x);
  const midX = (firstX + lastX) / 2;

  return (
    <svg className="line-chart" viewBox={`0 0 ${chartWidth} ${height}`} role="img" aria-label="并发和 TTFT 曲线">
      <path d={areaPath} />
      <path className="line" d={linePath} />
      {points.map((point, index) => (
        <circle key={`${point.x}-${point.y}-${index}`} cx={coordinateX(point.x)} cy={coordinateY(point.y)} r="3.5" />
      ))}
      <text x={paddingLeft - 8} y={height - paddingBottom} textAnchor="end">{formatNumber(minimumY)}{unit}</text>
      <text x={paddingLeft - 8} y={paddingTop} textAnchor="end">{formatNumber(maximumY)}{unit}</text>
      <text x={firstX} y={height - 16} textAnchor="start">{formatNumber(minimumX, 0)}</text>
      <text x={midX} y={height - 16} textAnchor="middle">{formatNumber((minimumX + maximumX) / 2, 0)}</text>
      <text x={lastX} y={height - 16} textAnchor="end">{formatNumber(maximumX, 0)}</text>
    </svg>
  );
}
