import { useEffect, useState } from "react";
import AuthPanel from "./components/AuthPanel";
import ComparePanel from "./components/ComparePanel";
import OverviewPanel from "./components/OverviewPanel";
import { fetchAuthStatus, fetchReport, logout } from "./services";
import type { Report, Run } from "./types";

type View = "overview" | "compare";

export default function App() {
  const [view, setView] = useState<View>("overview");
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [compareAFile, setCompareAFile] = useState("");
  const [compareBFile, setCompareBFile] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [reportA, setReportA] = useState<Report | null>(null);
  const [reportB, setReportB] = useState<Report | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [auth, setAuth] = useState<{ authenticated: boolean; username?: string } | undefined>(undefined);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const status = await fetchAuthStatus();
        setAuth(status);
      } catch {
        setAuth({ authenticated: false });
      }
    })();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (auth?.authenticated !== true) return;
    let cancelled = false;
    void (async () => {
      try {
        const response = await fetch("/api/runs");
        const payload = await response.json() as Run[];
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        if (!cancelled) {
          setRuns(payload);
          setSelectedFile(payload[0]?.filename ?? "");
          setCompareAFile(payload[0]?.filename ?? "");
          setCompareBFile(payload[1]?.filename ?? "");
        }
      } catch (failure) {
        if (!cancelled) setError(failure instanceof Error ? failure.message : "报告列表加载失败");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [auth?.authenticated]);

  useEffect(() => {
    const controller = new AbortController();
    if (!selectedFile) {
      setReport(null);
      return () => controller.abort();
    }
    void (async () => {
      try {
        const detail = await fetchReport(selectedFile, controller.signal);
        if (!controller.signal.aborted) {
          setReport(detail);
          setError("");
        }
      } catch (failure) {
        if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : "报告加载失败");
      }
    })();
    return () => controller.abort();
  }, [selectedFile]);

  useEffect(() => {
    const controller = new AbortController();
    if (!compareAFile || !compareBFile) {
      setReportA(null);
      setReportB(null);
      setCompareLoading(false);
      return () => controller.abort();
    }

    setCompareLoading(true);
    void (async () => {
      try {
        const [dataA, dataB] = await Promise.all([
          fetchReport(compareAFile, controller.signal),
          fetchReport(compareBFile, controller.signal),
        ]);
        if (!controller.signal.aborted) {
          setReportA(dataA);
          setReportB(dataB);
          setCompareLoading(false);
          setError("");
        }
      } catch (failure) {
        if (!controller.signal.aborted) {
          setCompareLoading(false);
          setError(failure instanceof Error ? failure.message : "对比报告加载失败");
        }
      }
    })();
    return () => controller.abort();
  }, [compareAFile, compareBFile]);

  if (auth === undefined) {
    return <div className="app auth-loading">登录检查中…</div>;
  }

  if (!auth.authenticated) {
    return (
      <AuthPanel
        onAuthenticated={(username) => {
          setAuth({ authenticated: true, username });
          setSelectedFile("");
          setReport(null);
        }}
      />
    );
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">LB</div>
          <div className="brand-copy">
            <h1>LLM Benchmark</h1>
            <p>实时查看推理性能指标</p>
          </div>
        </div>
        <nav className="tabs" aria-label="页面切换">
          <button type="button" className={view === "overview" ? "active" : ""} onClick={() => setView("overview")}>
            概览
          </button>
          <button type="button" className={view === "compare" ? "active" : ""} onClick={() => setView("compare")}>
            对比
          </button>
        </nav>
        <label className="run-picker" htmlFor="run-picker">
          <span>当前报告</span>
          <select id="run-picker" value={selectedFile} onChange={(event) => setSelectedFile(event.target.value)}>
            {runs.map((run) => (
              <option key={run.filename} value={run.filename}>
                {run.filename}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="logout-button" onClick={() => void logout()}>
          {auth.username ?? "管理员"}
        </button>
      </header>

      {view === "overview" ? (
        <OverviewPanel report={report} />
      ) : (
        <ComparePanel
          reportA={reportA}
          reportB={reportB}
          labelA={compareAFile}
          labelB={compareBFile}
          items={runs}
          onChangeA={setCompareAFile}
          onChangeB={setCompareBFile}
        />
      )}

      {loading ? <div className="status">加载中…</div> : null}
      {compareLoading && !loading ? <div className="status">对比报告加载中…</div> : null}
      {error ? <div className="status error">{error}</div> : null}
    </div>
  );
}
