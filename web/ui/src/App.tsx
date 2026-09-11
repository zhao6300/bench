import { useEffect, useMemo, useState } from "react";
import Comparison from "./components/Comparison";
import { concurrency, formatBytes, formatDate, formatDuration, formatNumber, metric } from "./format";
import type { Case, Report, ReportPair, Run } from "./types";

type View = "overview" | "compare";

export default function App() {
  const [view, setView] = useState<View>("overview");
  const [runs, setRuns] = useState<Run[]>([]);
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
        const payload = await response.json() as Run[];
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
  const environmentFacts = useMemo(() => getEnvironmentFacts(report), [report]);

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
            <section className="panel" aria-label="用例结果">
              <div className="panel-head">
                <h2>用例结果</h2>
                <small>{report?.cases?.length ?? 0} 个</small>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>状态</th>
                      <th>用例</th>
                      <th>模型</th>
                      <th>负载</th>
                      <th>并发</th>
                      <th>TTFT P50</th>
                      <th>TPOT P50</th>
                      <th>吞吐</th>
                      <th>Goodput</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(report?.cases ?? []).map((entry, index) => {
                      const concurrencyValue = concurrency(entry);
                      const workload = getWorkload(entry);
                      return (
                        <tr key={`${entry.id ?? entry.name ?? "case"}-${index}`}>
                          <td><span className={`status-badge ${getStatus(entry.status)}`}>{entry.status ?? "未知"}</span></td>
                          <td><span className="case-name">{entry.name ?? "未命名"}</span></td>
                          <td>{getStringParam(entry, "model") || "—"}</td>
                          <td>{workload}</td>
                          <td>{concurrencyValue === null ? "—" : concurrencyValue}</td>
                          <td>{formatNumber(metric(entry, "p50_ttft"))} s</td>
                          <td>{formatNumber(metric(entry, "p50_tpot"))} s</td>
                          <td>{formatNumber(metric(entry, "overall_throughput"))} tok/s</td>
                          <td>{formatNumber(metric(entry, "goodput_pct"))}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="detail-column" aria-label="关键信息">
              <section className="mini-grid">
                {detailCards.map((item) => (
                  <article className="card" key={item.title}>
                    <span>{item.title}</span>
                    <strong>{item.value}</strong>
                    <small>均值</small>
                  </article>
                ))}
              </section>

              <section className="panel environment-panel" aria-label="环境和元数据">
                <div className="panel-head">
                  <h2>环境与元数据</h2>
                  <small>{report?.suite?.run_state ?? "unknown"}</small>
                </div>
                <dl className="fact-list">
                  {environmentFacts.map((fact) => (
                    <div key={fact.label}>
                      <dt>{fact.label}</dt>
                      <dd>{fact.value}</dd>
                    </div>
                  ))}
                </dl>
                {report?.suite?.description ? <p className="description">{report.suite.description}</p> : null}
              </section>

              <section className="panel chart-panel" aria-label="并发趋势">
                <div className="panel-head">
                  <h2>TTFT 随并发变化</h2>
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
              <small>按用例名称自动匹配，支持部分对齐</small>
            </div>
            <div className="compare-grid">
              <label>
                <span>基准报告 A</span>
                <select value={compareLeft} onChange={(event) => setCompareLeft(event.target.value)}>
                  {runs.map((run) => (
                    <option key={run.filename} value={run.filename}>{run.filename}</option>
                  ))}
                </select>
              </label>
              <label>
                <span>对比报告 B</span>
                <select value={compareRight} onChange={(event) => setCompareRight(event.target.value)}>
                  {runs.map((run) => (
                    <option key={run.filename} value={run.filename}>{run.filename}</option>
                  ))}
                </select>
              </label>
            </div>
          </section>

          <Comparison
            reportA={comparisonReports.reportA}
            reportB={comparisonReports.reportB}
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

async function fetchReport(filename: string): Promise<Report> {
  const response = await fetch(`/runs/${encodeURIComponent(filename)}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return await response.json() as Report;
}

function getStatus(status: string | undefined): string {
  return status === "passed" || status === "failed" || status === "skipped" ? status : "unknown";
}

function getStringParam(entry: Case, key: string): string {
  const value = entry.params?.[key];
  return typeof value === "string" ? value : "";
}

function getWorkload(entry: Case): string {
  const inputLen = entry.params?.random_input_len;
  const outputLen = entry.params?.random_output_len;
  if (typeof inputLen === "number" && typeof outputLen === "number") {
    return `${formatNumber(inputLen, 0)} / ${formatNumber(outputLen, 0)} tok`;
  }
  return entry.params?.dataset === "text" ? "Text" : "—";
}

function count(cases: Case[], summary: Report["summary"], status: string): number {
  const value = summary?.[status];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return cases.filter((entry) => entry.status === status).length;
}

function getSummaryCards(report: Report | null): { label: string; value: string; detail: string; tone: string }[] {
  const cases = report?.cases ?? [];
  const summary = report?.summary ?? {};
  const total = typeof summary.total === "number" ? summary.total : cases.length;
  const passed = count(cases, summary, "passed");
  const failed = count(cases, summary, "failed");
  const interrupted = count(cases, summary, "interrupted");
  const running = count(cases, summary, "running");
  const pending = count(cases, summary, "pending");
  const skipped = count(cases, summary, "skipped");
  const abnormal = failed + interrupted + running + pending;
  const passRate = total ? (passed * 100) / total : null;
  const environment = report?.environment;
  const accelerator = environment?.host_inventory?.accelerators?.devices?.[0]?.name;

  return [
    {
      label: "基准",
      value: report?.suite?.name ?? "—",
      detail: `${report?.suite?.run_state ?? "unknown"} · ${formatDate(report?.suite?.started_at)}`,
      tone: "reference",
    },
    {
      label: "通过",
      value: `${passed} / ${total}`,
      detail: `${passRate === null ? "—" : `${passRate.toFixed(1)}%`} · ${skipped} skipped`,
      tone: "passed",
    },
    {
      label: "异常",
      value: String(abnormal),
      detail: `${failed} failed · ${interrupted} interrupted`,
      tone: "failed",
    },
    {
      label: "用时",
      value: formatDuration(report?.suite?.duration_seconds),
      detail: `${skipped} skipped · ${running + pending} active`,
      tone: "elapsed",
    },
    {
      label: "平均 TTFT",
      value: `${formatNumber(finiteMean(cases.map((entry) => metric(entry, "p50_ttft"))))} s`,
      detail: "P50 均值",
      tone: "prompt",
    },
    {
      label: "平均 TPOT",
      value: `${formatNumber(finiteMean(cases.map((entry) => metric(entry, "p50_tpot"))))} s`,
      detail: "P50 均值",
      tone: "completion",
    },
    {
      label: "吞吐量",
      value: `${formatNumber(finiteMean(cases.map((entry) => metric(entry, "overall_throughput"))))}`,
      detail: "tok/s · 用例均值",
      tone: "prompt",
    },
    {
      label: "环境",
      value: accelerator ?? `${environment?.host_inventory?.cpu?.logical_cores ?? "—"} cores`,
      detail: accelerator ? "加速器" : environment?.host_inventory?.cpu?.model ?? "CPU",
      tone: "reference",
    },
  ];
}

function getDetailCards(report: Report | null): { title: string; key: string; value: string }[] {
  const cases = report?.cases ?? [];
  const entries = [
    { title: "TTFT P99", key: "p99_ttft" },
    { title: "TPOT P99", key: "p99_tpot" },
    { title: "E2E P99", key: "p99_e2e" },
    { title: "Goodput", key: "goodput_pct" },
    { title: "QPS", key: "qps" },
    { title: "失败率", key: "failure_rate" },
  ];
  return entries.map((entry) => ({
    ...entry,
    value: formatNumber(finiteMean(cases.map((caseData) => metric(caseData, entry.key)))),
  }));
}

function getEnvironmentFacts(report: Report | null): { label: string; value: string }[] {
  const environment = report?.environment;
  const host = environment?.host_inventory;
  const cpu = host?.cpu;
  const devices = host?.accelerators?.devices ?? [];
  return [
    { label: "Python", value: environment?.python ?? "—" },
    { label: "主机", value: environment?.hostname ?? "—" },
    { label: "平台", value: `${host?.operating_system?.system ?? "—"} ${host?.operating_system?.release ?? ""}`.trim() },
    { label: "CPU", value: cpu?.model ? `${cpu.model} · ${cpu.logical_cores ?? "—"}C` : "—" },
    { label: "内存", value: formatBytes(host?.memory?.total_bytes) },
    { label: "加速器", value: devices.map((device) => device.name).filter(Boolean).join(", ") || "未发现" },
    { label: "开始", value: formatDate(report?.suite?.started_at) },
    { label: "结束", value: formatDate(report?.suite?.finished_at) },
  ];
}

function finiteMean(values: (number | null | undefined)[]): number | null {
  const numeric = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return numeric.length ? numeric.reduce((sum, value) => sum + value, 0) / numeric.length : null;
}

function getChartPoints(report: Report | null): { x: number; y: number }[] {
  return (report?.cases ?? [])
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
  const chartInnerWidth = chartWidth - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;
  const minimumY = Math.min(...points.map((point) => point.y));
  const maximumY = Math.max(...points.map((point) => point.y));
  const minimumX = Math.min(...points.map((point) => point.x));
  const maximumX = Math.max(...points.map((point) => point.x));
  const yRange = Math.max(maximumY - minimumY, 1e-6);
  const xRange = Math.max(maximumX - minimumX, 1);

  function coordinateX(value: number): number {
    if (points.length === 1) return paddingLeft + chartInnerWidth / 2;
    return paddingLeft + ((value - minimumX) / xRange) * chartInnerWidth;
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
    <svg className="line-chart" viewBox={`0 0 ${chartWidth} ${height}`} role="img" aria-label="TTFT 随并发变化的曲线">
      <path d={areaPath} />
      <path className="line" d={linePath} />
      {points.map((point, index) => (
        <circle key={`${point.x}-${point.y}-${index}`} cx={coordinateX(point.x)} cy={coordinateY(point.y)} r="3.5" />
      ))}
      <text x={paddingLeft - 8} y={height - paddingBottom} textAnchor="end">{formatNumber(minimumY)}{unit}</text>
      <text x={paddingLeft - 8} y={paddingTop + 6} textAnchor="end">{formatNumber(maximumY)}{unit}</text>
      <text x={firstX} y={height - 16} textAnchor="start">{formatNumber(minimumX, 0)}</text>
      <text x={midX} y={height - 16} textAnchor="middle">{formatNumber((minimumX + maximumX) / 2, 0)}</text>
      <text x={lastX} y={height - 16} textAnchor="end">{formatNumber(maximumX, 0)}</text>
    </svg>
  );
}
