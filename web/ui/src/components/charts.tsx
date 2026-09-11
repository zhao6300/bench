import { formatNumber } from "../format";

type ChartSeries = { name: string; color: string; values: (number | null)[] };

interface LineChartProps {
  labels: string[];
  series: ChartSeries[];
  unit: string;
}

export function LineChart({ labels, series, unit }: LineChartProps) {
  const values = series.flatMap((item) => item.values).filter((value): value is number => value !== null && Number.isFinite(value));
  const seriesPointCount = labels.length;
  if (seriesPointCount === 0 || values.length === 0) {
    return <div className="chart-empty" />;
  }

  const width = 860;
  const height = 280;
  const paddingLeft = 54;
  const paddingRight = 24;
  const paddingTop = 24;
  const paddingBottom = 52;
  const chartWidth = width - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max >= min ? max - min : 1;
  const yScale = (value: number) => paddingTop + chartHeight - ((value - min) / range) * chartHeight;
  const xScale = (index: number) => paddingLeft + (seriesPointCount === 1 ? chartWidth / 2 : (index / (seriesPointCount - 1)) * chartWidth);
  const xValues = Array.from({ length: seriesPointCount }, (_, index) => xScale(index));

  const paths = series.map((item) => {
    let path = "";
    let active = false;
    item.values.forEach((value, index) => {
      if (value === null || !Number.isFinite(value)) {
        active = false;
        return;
      }
      const x = xValues[index];
      const y = yScale(value);
      path += `${active ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)} `;
      active = true;
    });
    return { ...item, path: path.trim() };
  });

  const gridLines = [min, (min + max) / 2, max];
  const axisLabelIndexes = [0, Math.floor((seriesPointCount - 1) / 2), seriesPointCount - 1];

  return (
    <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`性能指标曲线${unit ? `（${unit}）` : ""}`}>
      <rect x={paddingLeft} y={paddingTop} width={chartWidth} height={chartHeight} className="chart-frame" />
      {gridLines.map((value, index) => (
        <line
          key={`${value}-${index}`}
          x1={paddingLeft}
          x2={paddingLeft + chartWidth}
          y1={yScale(value)}
          y2={yScale(value)}
          className="chart-grid"
        />
      ))}
      {gridLines.map((value) => (
        <text key={value} x={paddingLeft - 8} y={yScale(value)} textAnchor="end" className="axis-label">
          {formatNumber(value, 1)}
        </text>
      ))}
      {axisLabelIndexes.map((index) => (
        <text key={`${labels[index] ?? index}`} x={xValues[index]} y={height - 18} textAnchor="middle" className="axis-label">
          {labels[index] ?? ""}
        </text>
      ))}
      {paths.filter((item) => item.path).map((item) => (
        <path key={item.name} d={item.path} className={`chart-line ${item.color}`} />
      ))}
      {series.flatMap((item) =>
        item.values.map((value, index) =>
          value === null || !Number.isFinite(value) ? null : (
            <circle
              key={`${item.name}-${index}`}
              cx={xValues[index]}
              cy={yScale(value)}
              r={3}
              className={`chart-point ${item.color}`}
            />
          ),
        ),
      )}
    </svg>
  );
}

type DeltaRow = {
  label: string;
  unit: string;
  valueA: number | null;
  valueB: number | null;
  delta: number | null;
  deltaPct: number | null;
  tone: "positive" | "negative" | "neutral";
};

export function MetricDeltaBars({ rows }: { rows: DeltaRow[] }) {
  const percentages = rows.map((row) => Math.abs(row.deltaPct ?? 0));
  const maxPercent = Math.max(...percentages, 1);

  return (
    <div className="metric-delta-list">
      {rows.map((row) => {
        const width = row.deltaPct === null ? 0 : Math.max(Math.abs(row.deltaPct) / maxPercent * 100, 2);
        return (
          <div key={`${row.label}-${row.unit}`} className="delta-row">
            <div className="delta-label">{row.label}</div>
            <div className="delta-track">
              <span
                className={`delta-bar ${row.tone}`}
                style={{ width: `${width}%` }}
                title={[
                  `${row.label}: A ${formatNumber(row.valueA)} → B ${formatNumber(row.valueB)}${row.unit ? ` ${row.unit}` : ""}`,
                  row.delta === null ? "—" : `Δ ${formatNumber(row.delta, 2)}${row.unit ? ` ${row.unit}` : ""}`,
                  row.deltaPct === null ? "—" : `${row.deltaPct >= 0 ? "+" : ""}${formatNumber(row.deltaPct, 1)}%`,
                ].join(" | ")}
              />
            </div>
            <div className={`delta-value ${row.tone}`}>
              {row.deltaPct === null ? "—" : `${row.deltaPct >= 0 ? "+" : ""}${formatNumber(row.deltaPct, 1)}%`}
            </div>
          </div>
        );
      })}
    </div>
  );
}
