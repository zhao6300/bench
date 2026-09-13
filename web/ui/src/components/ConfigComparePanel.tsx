import { Fragment } from "react";
import type { CasePair } from "../metrics";
import { caseParamNumber, caseStringParam } from "../metrics";

type ConfigFieldType = "boolean" | "number" | "string";

interface ConfigField {
  key: string;
  label: string;
  type: ConfigFieldType;
}

interface ConfigFieldGroup {
  name: string;
  fields: ConfigField[];
}

type ConfigStatus = "一致" | "仅 A" | "仅 B" | "不同" | "—";

interface ConfigRowDisplay {
  key: string;
  label: string;
  valueA: string;
  valueB: string;
  hasValue: boolean;
  status: ConfigStatus;
}

const configFieldGroups: ConfigFieldGroup[] = [
  {
    name: "模型与负载",
    fields: [
      { label: "运行模式", key: "mode", type: "string" },
      { label: "模型", key: "model", type: "string" },
      { label: "分词器", key: "tokenizer", type: "string" },
      { label: "数据集", key: "dataset", type: "string" },
      { label: "上下文长度", key: "context_len", type: "number" },
      { label: "随机输入长度", key: "random_input_len", type: "number" },
      { label: "随机输出长度", key: "random_output_len", type: "number" },
      { label: "随机前缀长度", key: "random_prefix_len", type: "number" },
      { label: "随机范围比例", key: "random_range_ratio", type: "number" },
      { label: "共享前缀比例", key: "prefix_ratio", type: "number" },
      { label: "最大输出 Token", key: "max_tokens", type: "number" },
      { label: "请求数", key: "num_prompts", type: "number" },
    ],
  },
  {
    name: "执行与资源",
    fields: [
      { label: "并发", key: "concurrency", type: "number" },
      { label: "TP 大小", key: "tp_size", type: "number" },
      { label: "GPU 内存利用率", key: "gpu_mem_util", type: "number" },
      { label: "预热轮数", key: "warmup_rounds", type: "number" },
      { label: "每轮预热请求数", key: "warmup_requests_per_round", type: "number" },
      { label: "忽略 EOS", key: "ignore_eos", type: "boolean" },
      { label: "扫描最大并发", key: "sweep_max_concurrency", type: "number" },
    ],
  },
  {
    name: "质量目标",
    fields: [
      { label: "SLO TTFT", key: "slo_ttft", type: "number" },
      { label: "SLO TPOT", key: "slo_tpot", type: "number" },
      { label: "最大失败率", key: "max_failure_rate", type: "number" },
      { label: "最小 Goodput", key: "min_goodput_pct", type: "number" },
    ],
  },
  {
    name: "API",
    fields: [
      { label: "API 地址", key: "api_base", type: "string" },
      { label: "API 传输", key: "api_transport", type: "string" },
      { label: "API 超时", key: "api_timeout_seconds", type: "number" },
      { label: "使用 API Key", key: "api_key_configured", type: "boolean" },
      { label: "允许明文 HTTP", key: "allow_insecure_api_key", type: "boolean" },
    ],
  },
];

function paramText(pair: CasePair, field: ConfigField, side: "a" | "b"): string {
  const caseEntry = side === "a" ? pair.entryA : pair.entryB;
  if (field.type === "number") return caseParamNumber(caseEntry, field.key)?.toLocaleString() ?? "—";
  if (field.type === "boolean") {
    const value = caseStringParam(caseEntry, field.key);
    return value === "true" || value === "false" ? (value === "true" ? "是" : "否") : "—";
  }
  return caseStringParam(caseEntry, field.key) || "—";
}

function configRow(pairs: CasePair[], field: ConfigField): ConfigRowDisplay {
  const valuesA = new Set(pairs.map((pair) => paramText(pair, field, "a")));
  const valuesB = new Set(pairs.map((pair) => paramText(pair, field, "b")));
  const valueA = summarize(valuesA);
  const valueB = summarize(valuesB);
  const hasValue = valueA !== "—" || valueB !== "—";
  let status: ConfigStatus = "—";
  if (valueA !== "—" && valueB !== "—") {
    status = valueA === valueB ? "一致" : "不同";
  } else if (valueA !== "—") {
    status = "仅 A";
  } else if (valueB !== "—") {
    status = "仅 B";
  }
  return { key: field.key, label: field.label, valueA, valueB, hasValue, status };
}

function summarize(values: Set<string>): string {
  if (!values.size || values.has("—")) return "—";
  if (values.size === 1) return Array.from(values)[0] ?? "—";
  return `按用例变化（${values.size} 种）`;
}

export function ConfigComparePanel({ pairs }: { pairs: CasePair[] }) {
  if (!pairs.length) {
    return (
      <article className="panel config-compare-panel">
        <div className="panel-head">
          <div>
            <h3>配置对比</h3>
            <p className="config-subtitle">两份报告暂无对齐用例。</p>
          </div>
          <span>0 个对齐用例</span>
        </div>
        <div className="config-empty">暂无可用配置。</div>
      </article>
    );
  }

  return (
    <article className="panel config-compare-panel">
      <div className="panel-head">
        <div>
          <h3>配置对比</h3>
          <p className="config-subtitle">基于 A / B 对齐用例的参数，不重复测量结果。</p>
        </div>
        <span>{pairs.length} 个对齐用例</span>
      </div>
      <div className="table-wrap">
        <table className="data-table config-table">
          <thead>
            <tr>
              <th scope="col">配置项</th>
              <th scope="col">报告 A</th>
              <th scope="col">报告 B</th>
              <th scope="col">状态</th>
            </tr>
          </thead>
          <tbody>
            {configFieldGroups.map((group) => {
              const rows = group.fields.map((field) => configRow(pairs, field));
              if (!rows.some((row) => row.hasValue)) return null;
              return (
                <Fragment key={group.name}>
                  <tr className="metric-group-header">
                    <th colSpan={4} scope="colgroup">{group.name}</th>
                  </tr>
                  {rows.map((row) => (
                    <tr key={row.key}>
                      <th scope="row">{row.label}</th>
                      <td>{row.valueA}</td>
                      <td>{row.valueB}</td>
                      <td>
                        <span className={`config-status ${row.status === "一致" ? "is-equal" : "is-different"}`}>
                          {row.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </article>
  );
}

export default ConfigComparePanel;
