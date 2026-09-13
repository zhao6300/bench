import { caseMetric, caseStringParam, metricOptions, type CasePair } from "../metrics";
import StatusBadge from "./StatusBadge";
import { useEffect, useMemo, useRef, useState } from "react";

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

const summaryMetricKeys = [
  "p50_ttft",
  "p50_tpot",
  "overall_throughput",
  "goodput_pct",
] as const;

const summaryMetrics = metricOptions.filter((metric) =>
  (summaryMetricKeys as readonly string[]).includes(metric.key),
);

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
  const [hoveredPair, setHoveredPair] = useState<
    { pair: CasePair; left: number; top?: number; bottom?: number; width: number; maxHeight: number } | undefined
  >();
  const [query, setQuery] = useState("");
  const [collapsedPairs, setCollapsedPairs] = useState<Set<string>>(new Set());
  const hoverDelayRef = useRef<number | undefined>(undefined);

  const quickKeywords = useMemo(() => {
    const unique = new Set<string>();
    for (const pair of pairs) {
      for (const entry of [pair.entryA, pair.entryB]) {
        if (entry?.status) unique.add(entry.status);
        const model = caseStringParam(entry, "model");
        if (model) unique.add(model);
        const dataset = caseStringParam(entry, "dataset");
        if (dataset) unique.add(dataset);
      }
    }
    return [...unique].sort((left, right) => left.localeCompare(right, "zh-CN"));
  }, [pairs]);

  const filteredPairs = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return pairs;
    const searchable = (pair: CasePair) => [
      pair.entryA?.name,
      pair.entryA?.id,
      pair.entryA?.case_key,
      pair.statusA,
      caseStringParam(pair.entryA, "model"),
      caseStringParam(pair.entryA, "dataset"),
      pair.entryB?.name,
      pair.entryB?.id,
      pair.entryB?.case_key,
      pair.statusB,
      caseStringParam(pair.entryB, "model"),
      caseStringParam(pair.entryB, "dataset"),
    ]
      .filter((value): value is string => typeof value === "string")
      .map((value) => String(value).toLowerCase());
    return pairs.filter((pair) => searchable(pair).some((text) => text.includes(needle)));
  }, [pairs, query]);

  useEffect(() => () => window.clearTimeout(hoverDelayRef.current), []);

  const clearHoverDelay = () => window.clearTimeout(hoverDelayRef.current);

  const showHoverDetail = (pair: CasePair, event: React.PointerEvent<HTMLElement>) => {
    clearHoverDelay();
    const cardRect = event.currentTarget.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const width = Math.min(Math.max(cardRect.width, 584), viewportWidth - 32);
    const left = Math.max(16, Math.min(cardRect.left, viewportWidth - width - 16));
    const availableBelow = viewportHeight - cardRect.bottom - 24;
    const availableAbove = cardRect.top - 24;
    const below = availableBelow > availableAbove;
    const top = below ? cardRect.bottom + 8 : undefined;
    const bottom = below ? undefined : viewportHeight - cardRect.top + 8;
    const availableHeight = below ? availableBelow : availableAbove;

    setHoveredPair({
      pair,
      left,
      top,
      bottom,
      width,
      maxHeight: Math.min(560, Math.max(240, availableHeight)),
    });
  };

  const scheduleHideHoverDetail = () => {
    clearHoverDelay();
    hoverDelayRef.current = window.setTimeout(() => setHoveredPair(undefined), 90);
  };

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
      <div className="case-comparison-filter" aria-label="用例对照快速过滤">
        <label className="case-comparison-search" htmlFor="case-comparison-search-input">
          <span>过滤用例</span>
          <input
            id="case-comparison-search-input"
            value={query}
            list="case-comparison-quick-keywords"
            placeholder={`按名称、ID、状态、模型或数据集过滤，如 ${query || "DeepSeek-V4-Flash"}`}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <span className="case-comparison-result-count">
          {filteredPairs.length} / {pairs.length}
        </span>
        {query ? (
          <button type="button" onClick={() => setQuery("")}>清除</button>
        ) : null}
        <datalist id="case-comparison-quick-keywords">
          {quickKeywords.map((keyword) => (
            <option key={keyword} value={keyword} />
          ))}
        </datalist>
      </div>
      <div className="case-comparison-quick-keywords" aria-label="已有用例关键字">
        {quickKeywords.map((keyword) => (
          <button
            key={keyword}
            type="button"
            className={query === keyword ? "active" : ""}
            onClick={() => setQuery(keyword)}
          >
            {keyword}
          </button>
        ))}
      </div>
      {pairs.length === 0 ? (
        <div className="case-comparison-empty">暂无可对齐的用例</div>
      ) : (
        <>
        {selectedPair ? (
          <div className="case-pair-modal" role="dialog" aria-modal="true">
            <button
              type="button"
              className="case-pair-modal-backdrop"
              aria-label="关闭详细对比"
              onClick={() => setSelectedPair(undefined)}
            />
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
                <button type="button" onClick={() => setSelectedPair(undefined)}>关闭</button>
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
          </div>
        ) : null}

        {hoveredPair ? (
          <section
            className="case-comparison-hover-detail"
            style={{
              left: hoveredPair.left,
              width: hoveredPair.width,
              maxHeight: hoveredPair.maxHeight,
              ...(hoveredPair.top !== undefined ? { top: hoveredPair.top } : { bottom: hoveredPair.bottom }),
            }}
            aria-hidden="true"
            onPointerEnter={clearHoverDelay}
            onPointerLeave={scheduleHideHoverDetail}
          >
            <div className="case-comparison-hover-head">
              <strong>{hoveredPair.pair.entryA?.name ?? hoveredPair.pair.entryB?.name ?? hoveredPair.pair.label}</strong>
              <button
                type="button"
                onClick={() => {
                  setHoveredPair(undefined);
                  setSelectedPair(hoveredPair.pair);
                }}
              >
                打开
              </button>
            </div>
            <div className="case-pair-detail-metrics">
              {detailMetrics.map((metric) => (
                <MetricComparison
                  key={metric.key}
                  pair={hoveredPair.pair}
                  metric={metric}
                  max={pairMetricMax(hoveredPair.pair, metric.key)}
                />
              ))}
            </div>
          </section>
        ) : null}

        <div className="case-comparison-grid">
          {filteredPairs.map((pair) => {
            const title = pair.entryA?.name ?? pair.entryB?.name ?? pair.label;
            const detail = pair.entryA?.id ?? pair.entryB?.id ?? "";
            const isCollapsed = collapsedPairs.has(pair.key);
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
                onPointerEnter={(event) => showHoverDetail(pair, event)}
                onPointerLeave={scheduleHideHoverDetail}
              >
                <header className="case-comparison-case-head">
                  <div className="case-comparison-title-block">
                    <span className="case-comparison-title">{title}</span>
                    {detail ? <span className="case-comparison-detail">{detail}</span> : null}
                  </div>
                    <div className="case-comparison-status">
                      <StatusBadge status={pair.statusA} />
                      <StatusBadge status={pair.statusB} />
                      <button
                        type="button"
                        aria-expanded={!isCollapsed}
                        aria-label={isCollapsed ? "展开用例" : "折叠用例"}
                        onClick={(event) => {
                          event.stopPropagation();
                          setCollapsedPairs((previous) => {
                            const next = new Set(previous);
                            if (next.has(pair.key)) {
                              next.delete(pair.key);
                            } else {
                              next.add(pair.key);
                            }
                            return next;
                          });
                        }}
                      >
                        <svg
                          aria-hidden="true"
                          viewBox="0 0 12 12"
                          style={isCollapsed ? { transform: "rotate(-90deg)" } : undefined}
                        >
                          <path
                            d="M2.5 4.5H9.5L6 8.2Z"
                            fill="currentColor"
                            stroke="currentColor"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </button>
                  </div>
                </header>
                {isCollapsed ? null : (
                  <div className="case-comparison-metrics">
                    {summaryMetrics.map((metric) => (
                      <MetricComparison
                        key={metric.key}
                        pair={pair}
                        metric={metric}
                        max={pairMetricMax(pair, metric.key)}
                      />
                    ))}
                  </div>
                )}
              </article>
            );
          })}
        </div>
        </>
      )}
    </section>
  );
}
