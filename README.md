# LLM Inference Benchmark

面向 **vLLM 离线引擎**和 **OpenAI 兼容推理服务**的 LLM 性能基准工具。它使用可配置的文本或随机 token 工作负载，测量长上下文 Prefill、生成 Decode、流式延迟和 SLO 容量。

所有可运行的基准示例均位于 [`examples/`](examples/)。README 不重复维护手写参数组合：请从真实 JSON 配置复制本地副本、修改部署参数后运行，确保文档与仓库示例保持一致。

## 能力概览

- **两种后端**：本地 GPU 上的 `offline` vLLM 路径，或已部署服务的 `api` 路径。
- **核心指标**：Prefill/Decode/整体吞吐、TTFT、TPOT、估算 ITL、QPS、失败率与 SLO Goodput。
- **负载控制**：并发、请求数、上下文/输出长度、共享前缀、随机 token、长度抖动和混合工作负载。
- **容量分析**：吞吐扫描、按 SLO 搜索最大并发，以及 Prefill/Decode（P/D）比例建议。
- **可复现套件**：JSON 配置支持默认参数、矩阵展开、重复执行、筛选、报告和断点恢复。

## 环境要求

- Python 3.10 或更高版本。
- `api` 模式：一个支持 `/v1/chat/completions` 流式响应的 OpenAI 兼容服务，以及可加载的 Hugging Face tokenizer。
- `offline` 模式：NVIDIA GPU、CUDA/驱动，以及与环境兼容的 `vllm`、`torch` 和 `transformers`。
- `random` 数据集：NumPy。
- 配置报告恢复依赖 `fcntl`，推荐在 Linux 或 macOS 上运行。

> API 模式会用 tokenizer 计算实际 token 数；即使模型由远程服务提供，也通常应在配置的 `tokenizer` 字段指向同版本的本地 tokenizer 或 Hugging Face repo。

## 安装

在仓库根目录创建虚拟环境并安装 API、随机数据集与测试依赖：

```zsh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

安装后可使用 `llm-benchmark` 命令；也可直接执行 `python3 benchmark/benchmark.py`。离线 vLLM 依赖与 CUDA、PyTorch 的组合强相关，请按照目标环境安装兼容版本后再使用 `offline` 模式。

验证本地安装和示例配置（不会发送模型或 API 请求）：

```zsh
llm-benchmark --help
llm-benchmark --config examples/benchmark-config.example.json --validate-config
llm-benchmark --config examples/benchmark-config.example.json --list-cases
```

## 从 examples 开始

### 1. 创建本地可运行配置

[`examples/benchmark-config.example.json`](examples/benchmark-config.example.json) 是通用 API 套件模板，包含 `smoke`、前缀缓存、Decode 矩阵和混合负载用例。先将它复制为忽略的本地文件，再替换服务地址、模型名和 tokenizer 路径：

```zsh
cp examples/benchmark-config.example.json examples/benchmark-config.local.json
# 编辑 examples/benchmark-config.local.json 的 api_base、model 和 tokenizer

# 配置和展开检查不会发送请求
llm-benchmark --config examples/benchmark-config.local.json --validate-config
llm-benchmark --config examples/benchmark-config.local.json --list-cases

# 完成服务配置后，先运行低成本 smoke 用例
llm-benchmark --config examples/benchmark-config.local.json --tag smoke
```

需要鉴权时，在本地配置中使用 `api_key_env`，例如 `"api_key_env": "BENCHMARK_API_KEY"`，然后从 shell 注入密钥：

```zsh
export BENCHMARK_API_KEY='replace-me'
llm-benchmark --config examples/benchmark-config.local.json --tag smoke
```

不要将密钥写入 JSON 或提交 `*.local.json`。工具默认拒绝将 bearer key 发往非 loopback 的明文 HTTP 服务；仅在受信任内网且明确知悉风险时，才在本地配置启用 `allow_insecure_api_key`。

### 2. 选择与目标匹配的现有配置

| 目标 | 配置文件 | 内容 |
| --- | --- | --- |
| 通用 API 回归套件 | [`benchmark-config.example.json`](examples/benchmark-config.example.json) | 冒烟、缓存、Decode 矩阵、混合负载与禁用的昂贵用例。 |
| 128K Prefill / Decode 峰值吞吐 | [`benchmark-config-throughput-sweep-128k-2k.json`](examples/benchmark-config-throughput-sweep-128k-2k.json) | 随机 128K 输入 / 2K 输出的吞吐扫描。 |
| 128K / 2K、70% 缓存命中的 SLO 容量 | [`benchmark-config-slo-capacity-128k-2k-cache-hit-0.7.json`](examples/benchmark-config-slo-capacity-128k-2k-cache-hit-0.7.json) | 随机共享前缀、线性精扫和候选确认。 |
| 128K 并发阶梯 | [`benchmark-config-concurrency-staircase-128k.json`](examples/benchmark-config-concurrency-staircase-128k.json) | 固定 128K 请求形状的多档并发测试。 |
| 256K 并发阶梯 | [`benchmark-config-concurrency-staircase-256k.json`](examples/benchmark-config-concurrency-staircase-256k.json) | 固定 256K 请求形状的多档并发测试。 |
| 64K / 128K / 240K 并发矩阵 | [`benchmark-config-concurrency-matrix-64k-128k-240k.json`](examples/benchmark-config-concurrency-matrix-64k-128k-240k.json) | 多上下文长度和并发组合。 |

长上下文示例包含部署相关地址、模型名、tokenizer 路径或环境变量名。它们是参数参考，不应直接对陌生环境运行。以下流程以真实的 128K 吞吐配置为例：

```zsh
cp examples/benchmark-config-throughput-sweep-128k-2k.json \
  examples/benchmark-config-throughput-sweep-128k-2k.local.json
# 编辑本地副本中的 api_base、model、tokenizer 和鉴权设置

# 先做无流量检查
llm-benchmark \
  --config examples/benchmark-config-throughput-sweep-128k-2k.local.json \
  --validate-config
llm-benchmark \
  --config examples/benchmark-config-throughput-sweep-128k-2k.local.json \
  --list-cases

# 确认服务和负载成本后，运行本地副本
llm-benchmark \
  --config examples/benchmark-config-throughput-sweep-128k-2k.local.json
```

配置模式支持 `--tag`、`--case 'pattern-*'`、`--report PATH`、`--no-resume` 和 `--fail-fast`。相对报告路径以配置文件所在目录为基准；固定报告路径默认会恢复此前成功且无请求级失败的用例。

## 数据集与长度语义

`text`（默认）使用中文填充文本。`context_len` 控制目标输入长度，`max_tokens` 控制输出上限；开启 `share_prefix` 后可用 `prefix_ratio` 设置共享部分。

`random` 生成可复现的 token 序列，更适合精确压力测试。它在以下现有配置中使用：

- [`benchmark-config-throughput-sweep-128k-2k.json`](examples/benchmark-config-throughput-sweep-128k-2k.json)
- [`benchmark-config-slo-capacity-128k-2k-cache-hit-0.7.json`](examples/benchmark-config-slo-capacity-128k-2k-cache-hit-0.7.json)
- [`benchmark-config-concurrency-staircase-128k.json`](examples/benchmark-config-concurrency-staircase-128k.json)
- [`benchmark-config-concurrency-staircase-256k.json`](examples/benchmark-config-concurrency-staircase-256k.json)

对 `random` 数据集，`random_input_len`、`random_output_len` 和 `random_prefix_len` 是权威参数，分别表示独有输入、输出上限和共享前缀；它们优先于通用的 `context_len`/`max_tokens`。实际统计以发送前 tokenizer 重编码后的 `DatasetBatch` 为准，报告可能与目标长度有少量差异。`share_prefix` 与 `prefix_ratio` 仅对 `text` 生效。

## 配置说明

每个 suite 配置的根对象使用 `version: 1`，并可定义：

- `defaults`：所有 case 的后端、数据集、SLO 和预热默认值。
- `cases`：一个或多个基准用例；可用 `enabled: false` 暂停昂贵用例。
- `matrix`：对指定参数做笛卡尔积展开。
- `repeat`：重复运行同一个 case。
- `report`：报告路径、缩进和请求详情保存策略。

请以 [`examples/benchmark-config.example.json`](examples/benchmark-config.example.json) 为 schema 参考，而不是复制 README 中的片段。配置中的 `${VAR}` 会读取环境变量；需要密钥时使用 `api_key_env`，不要使用明文 `api_key`。

## 指标说明

- **TTFT**：从请求发出到收到第一个内容 token 的时间。
- **TPOT**：相邻内容 token 的平均耗时；少于两个输出 token 时不计算。
- **Prefill 吞吐**：成功请求从发出到首 token 的活动区间内处理的 prompt token 速率。
- **Decode 吞吐**：成功请求从首 token 到最后一个内容 token 的活动区间内生成的 token 速率（不含首 token）。
- **整体吞吐 / QPS**：整个测试窗口的 token 交付速率 / 完成请求速率。
- **Goodput**：全部请求中同时满足 TTFT 和（输出超过一个 token 时）TPOT SLO 的比例；没有首 token 的请求计入失败。

阶段活动窗口包含客户端可观测的排队与网络延迟，不等同于服务端 scheduler 的纯 GPU 计算时间。若输出 token 数无法通过服务端 usage 或本地 tokenizer 可靠确定，依赖该计数的指标会被省略而非估算。

## 可选 Nsys Profiling

在已复制并配置好的本地 suite 上设置环境变量，即可让工具在基准前后调用 `docker exec ... nsys start/stop`：

```zsh
NSYS_PROFILE=true \
NSYS_CONTAINER=your-inference-container \
NSYS_SESSION=vllm \
llm-benchmark \
  --config examples/benchmark-config-throughput-sweep-128k-2k.local.json
```

这要求容器内已有 `nsys`，并且当前用户有 Docker 访问权限。profile 产物不会自动提交。

## 开发约定

贡献者请先阅读 [AGENTS.md](AGENTS.md)。提交前至少执行：

```zsh
python3 -m compileall benchmark
python3 benchmark/benchmark.py --help
python3 benchmark/benchmark.py --config examples/benchmark-config.example.json --validate-config
python3 -m pytest -q
```

本项目当前未声明开源许可证；在复制、分发或对外发布前，请先向仓库维护者确认许可条款。
