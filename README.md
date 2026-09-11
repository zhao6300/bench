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

在仓库根目录使用 `uv` 创建虚拟环境并安装 API、随机数据集、测试与 hook 依赖：

```zsh
# 若尚未安装 uv：curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/common.txt
uv pip install -r requirements/lint.txt
uv pip install -e .
web/bootstrap.sh
```

### 构建和安装 wheel

项目使用 `pyproject.toml` 中的标准 PEP 517 配置构建 wheel，不需要额外维护 `setup.py` 或 `setup.cfg`。在仓库根目录执行：

```zsh
uv build --wheel
uv pip install 'dist/llm_inference_benchmark-0.1.0-py3-none-any.whl[api,random]'
```

`uv build --wheel` 会将 wheel 写入 `dist/`；wheel 只包含 `benchmark` Python 包和 `llm-benchmark` 命令，不包含 `examples/`、`tests/` 或本地配置。使用 wheel 安装后，示例仍需从源码仓库获取。`[api,random]` 是可选 extras；如果只使用基础功能，可安装不带 extras 的 wheel。

安装后可使用 `llm-benchmark` 命令；也可直接执行 `.venv/bin/python benchmark/benchmark.py`。离线 vLLM 依赖与 CUDA、PyTorch 的组合强相关，请按照目标环境安装兼容版本后再使用 `offline` 模式。

通用依赖已默认包含 `boto3` 和 `aiohttp`；本地报告与 S3 报告均无需额外安装依赖，`aiohttp` 可用于 API 正式流式轮次的异步连接池。S3 凭据和 endpoint 在运行时通过环境变量配置。

验证本地安装和示例配置（不会发送模型或 API 请求）：

```zsh
llm-benchmark --help
llm-benchmark --config examples/benchmark-config.example.json --validate-config
llm-benchmark --config examples/benchmark-config.example.json --list-cases
```

### 实时进度显示

实际运行 benchmark 时，`--progress` 控制控制台实时状态显示，默认 `off`，不会显示 TUI 或行式实时进度。显式传入 `--progress rich` 时，交互式终端会使用接近全屏的 Rich dashboard；`--progress auto` 仍会按终端能力选择 Rich 或 plain，`--progress plain` 强制使用行式输出。dashboard 在独立终端屏幕中按顶部运行状态、suite/case 与当前 round 分区、最近事件和底部运行信息展示进度；Rich 模式下，初始化、服务诊断、预热、扫描、结果和运行错误等文本会进入“最近事件”区域，不会直接写入控制台破坏 TUI。宽终端使用 suite/round 双栏，窄终端自动改为纵向布局。benchmark 完成且（若启用）最终 JSON 报告已写入后，Rich 会切换到表格化结果总览。最终页采用“总览 + 详情”布局：上方总览表按终端宽度显示 3/5/6 列，用于快速比较用例、状态、负载、核心指标和结论；它会以当前选中用例为中心分页，避免长 suite 挤满屏幕。核心指标展示聚合吞吐、平均 TTFT 和达标率。下方详情面板展示选中用例的 TTFT/TPOT/E2E 平均值与分位数、吞吐、Goodput、服务端观测、场景结果和错误说明；按 `↑`/`↓` 或 `j`/`k` 可切换用例。TUI 将内部场景标识显示为用户名称，例如“单一负载接口”“混合负载接口”“吞吐扫描”“服务等级目标容量搜索”和“预填充/解码容量评估”，但 JSON 报告仍保留兼容的内部 key。列头仅说明指标类别；每项数值都会在本身旁边显示单位，例如毫秒、个 token、个 token/秒、个请求/秒、百分比或个请求。详情使用标准的 TTFT、TPOT、E2E 缩写，以及整体吞吐、每秒完成请求数、达标率和 KV Cache 观测。延迟分位数会显示平均值、P50、P90 和 P99；扫描显示峰值吞吐和最佳并发，服务等级目标容量搜索显示最大通过并发、确认并发与失败边界，预填充/解码容量评估显示两阶段吞吐和实例比建议。所有不适用或报告缺失的指标显示为 `-`，不会以零值伪造结果。此时按 `Q` 或 `Ctrl-C` 退出并恢复原有终端内容。plain/off 模式不读取键盘，仍会自动结束。dashboard 不会写入 JSON 报告。

```zsh
# 默认：关闭实时进度/TUI，保留最终指标、错误与 JSON 报告输出
llm-benchmark --config examples/benchmark-config.example.json

# 显式启用自动选择：交互终端使用 Rich，重定向输出时使用 plain
llm-benchmark --progress auto --config examples/benchmark-config.example.json

# 强制使用行式进度，适合日志采集
llm-benchmark --progress plain --config examples/benchmark-config.example.json

# 强制 Rich 面板；依赖未安装时会给出可操作错误
llm-benchmark --progress rich --config examples/benchmark-config.example.json

# 关闭实时进度，保留最终指标、错误与 JSON 报告输出
llm-benchmark --progress off --config examples/benchmark-config.example.json
```

### 调试日志

`debug` 可作为 suite JSON 的**根字段**配置，默认 `false`；`--debug` 保持可用，且命令行与配置任一为 `true` 即启用。它作用于整次 suite 执行，不属于 `defaults`、case `params` 或 `matrix`，因此不会改变 case 身份、恢复指纹或 JSON 报告 schema。启用后，benchmark 内部生命周期、API round/request 元数据、Compose 阶段和 readiness probe 会以统一格式写入 `stderr`；既有最终指标、进度输出和报告仍保持原有行为。

```json
{
  "version": 1,
  "debug": true
}
```

```zsh
# 保存调试日志；stdout 和 JSON 报告不受影响
llm-benchmark --debug --config examples/benchmark-config.example.json \
  2>benchmark-debug.log

# 与实时进度独立，可同时使用
llm-benchmark --debug --progress plain \
  --config examples/benchmark-config.example.json
```

调试日志只记录请求 ID、并发度、token 数量、耗时、状态码和安全的目标 host/path 等元数据，不记录 prompt、完整请求体、SSE 内容、生成文本、Authorization/API key、S3 凭据或捕获的 Compose stdout/stderr。Rich dashboard 运行时会动态捕获当前 `stderr`，因此 debug 日志会进入 dashboard 的最近事件区域；`--validate-config` 和 `--list-cases` 仍不会发送 API 请求、启动 Docker 或执行 GPU benchmark。

项目使用 Rich 管理实时请求计数和状态表，因此不需要额外引入 `tqdm`；两者同时使用会产生重复进度条并干扰重定向日志。

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

默认 `api_transport` 为 `requests`，以保持既有行为。高并发 API 基准可在命令行使用 `--api-transport aiohttp`，或在本地 JSON 的 `defaults` / case 参数中设置 `"api_transport": "aiohttp"`。`--api-timeout-seconds`（JSON：`api_timeout_seconds`）控制每条正式 chat-completions 流式请求的超时，默认 `600` 秒，两个 transport 都适用；`requests` 使用连接/读取超时语义，`aiohttp` 使用单请求总时长。

### 请求速率调度

这组参数只作用于正式 API round，不影响预热、服务诊断、Prometheus 指标采集或 automation `/models` preflight。`concurrency` 始终只限制**真实 HTTP 在途数**，不会改写已生成的到达计划。可在命令行设置，或在 suite 的 `defaults`、case `params`/`matrix` 使用同名 snake_case 字段；也支持写入 `requests` 对象：

```json
{
  "requests": {
    "concurrency": 8,
    "count": 64,
    "target_rps": 20,
    "rate_schedule": "gamma",
    "rate_burstiness": 0.5,
    "rate_ramp_up_strategy": "none",
    "rate_ramp_up_start_rps": null,
    "rate_ramp_up_end_rps": null,
    "rate_overload_policy": "no-catch-up"
  }
}
```

- `target_rps`：未启用 ramp 时的目标**平均计划到达速率** `λ`，单位 req/s。省略或设为 `null` 时，所有请求在计划时刻 `0` 到达，不会按并发倒数限流。小于 `1` 的正数合法，例如 `0.2` 表示固定计划下每 `5` 秒一个请求；它不保证实际 HTTP 准入率或完成 QPS。
- `rate_schedule`：生成到达间隔的分布。`fixed`（默认）为严格的 `1 / target_rps`；`poisson` 为指数分布（等价 Gamma shape=1）；`gamma` 配合 `rate_burstiness` 使用 Gamma 分布。随机计划预先采样并累加为绝对到达时刻，未启用 ramp 时会整体归一化，使最后一个请求约落在 `请求数 / target_rps` 秒。
- `rate_burstiness`：仅 `gamma` 必填，为 Gamma 的 shape；scale 为 `1 / (target_rps × shape)`，故平均间隔仍为 `1 / target_rps`。小于 `1` 更突发，大于 `1` 更均匀，`inf` 退化为固定间隔。
- `rate_ramp_up_strategy`：`none`（默认）、`linear` 或 `exponential`。启用后必须将 `target_rps` 设为 `null`，并同时设置 `rate_ramp_up_start_rps` 与 `rate_ramp_up_end_rps`；速率按请求序号插值，而非 wall-clock 时间。ramp 下不会进行总时长归一化。
- `concurrency`：计划到达时会创建/排队请求任务；只有获得线程 worker 或 aiohttp semaphore 后才真正发起 HTTP。服务变慢时，实际 HTTP 准入率可低于计划到达速率，但后续计划到达时间不变。
- `rate_overload_policy`：`no-catch-up`（默认）和 `catch-up` 均保留计划到达并排队，后者仅作为兼容别名；`drop` 在到达时并发容量已满则不发 HTTP，并记录客户端失败。

报告的 `metrics.pacing` 会写入计划请求数/计划时长、到达分布、Gamma shape、ramp 元数据、实际 HTTP 准入速率、相邻准入间隔、并发排队时间和 `dropped_requests`。实际准入速率只统计已发往 HTTP 的请求；`drop` 请求会写入失败记录，因此同时计入总请求数、失败数和失败率。TTFT 仍从实际 HTTP 请求开始计时，不包含客户端的计划等待或并发排队等待。当前数据集没有 trace timestamp 输入，因此尚未提供 vLLM 的 `self_timed` trace 回放或独立 probe 请求功能；无论 transport 如何选择，payload、SSE 解析、token/延迟指标与 bearer key 的安全限制保持一致。

### 单机 Docker Compose 推理服务

`automation.docker_service` 可在单机上受管启动一个 vLLM 或 SGLang Compose project，等待其 OpenAI `/v1/models` 接口就绪后再运行 suite，并在正常完成、测试失败或服务就绪失败后停止**本次创建的** project。该功能默认关闭；`--validate-config` 与 `--list-cases` 只校验 bundle 文件，绝不调用 Docker 或 HTTP。

`automation.budget` 是可选保护机制：省略或设为 `null` 时不限制请求数或输出 token，工具不会因预算拒绝 suite。若要在启动前阻止超出预估成本的运行，再显式设置 `max_total_requests` 与 `max_estimated_output_tokens`；该检查仅基于保守估算，不能替代人工确认真实 API、GPU 和运行时成本。

默认每次 automation 都要求带 `{timestamp}` 的独立报告，并从头执行，避免不同服务生命周期误用旧结果。若需要在 `Ctrl-C`、客户端容器中断或单个 profile 失败后恢复，请显式设置 `"automation.resume": true`，同时使用一个**不含** `{timestamp}` 的稳定本地 `report.path`；`--no-resume` 可强制从头重跑。默认恢复要求所选执行计划完全一致；若确认当前配置只应用于后续未完成用例，可额外传入 `--resume-allow-config-changes`。该开关按展开 case 的名称、场景、重复序号、矩阵值和启用状态保留无请求级失败的 `passed` case，即使其预热、模型或其他参数已变；失败、中断、跳过和新增 case 都会重跑，报告会记录前后执行计划指纹。因此必须由操作者确认被保留结果仍可用于当前比较。Compose 模式要求本地报告锁，因此不可用 S3 checkpoint。不要在同一稳定路径上并发运行多个 suite。

```json
{
  "automation": {
    "enabled": true,
    "resume": true
  },
  "report": {
    "path": "reports/vllm-resume.json"
  }
}
```

推理引擎参数不进入 benchmark schema。共享 Compose bundle 可通过 `base_environment` 传入基础替换变量，再由 `profiles[].environment` 覆盖或补充 TP、DP、EP、KV cache、attention backend 等任意引擎变量；这些值作为 `docker compose` 子进程环境注入。实际生效的合并参数会写入单 profile 报告的 `run.automation.docker_service.environment`，以及多 profile 聚合报告的 `profiles[].environment`；profile 变量优先于基础变量。`env_file` 仍可用于既有单 profile 配置，但不再是扩展参数的唯一方式。benchmark 始终以固定、无 shell 的 `docker compose [--env-file ...] -f ... -p ...` argv 拉起服务，因此新增 vLLM/SGLang 参数只需修改 Compose bundle 或环境变量，不需要改 Python。每个与所选引擎模板匹配的拓扑（例如 TP/DP）应使用唯一的 Compose project 和 report；同一主机上的 profile 必须串行运行。

从自动化示例复制本地 suite 后，先准备与 suite 同目录的 Compose bundle 和 `.env` 文件（它们不应提交）：

```json
{
  "automation": {
    "enabled": true,
    "api_preflight": {
      "enabled": true,
      "timeout_seconds": 5
    },
    "docker_service": {
      "enabled": true,
      "profile_name": "tp4-dp1",
      "engine": "vllm",
      "compose_files": [
        "deploy/vllm.compose.yml",
        "deploy/tp4-dp1.override.yml"
      ],
      "env_file": "deploy/tp4-dp1.local.env",
      "project_name": "benchmark-tp4-dp1",
      "start_timeout_seconds": 120,
      "ready_timeout_seconds": 600,
      "poll_interval_seconds": 2,
      "probe_timeout_seconds": 5
    }
  }
}
```

需要自动启动推理服务并运行 suite 时，使用 [`benchmark-config-automation.example.json`](examples/benchmark-config-automation.example.json) 作为统一入口。它用 `BENCHMARK_ENGINE` 与 `BENCHMARK_MODEL_PROFILE` 同时选择 `docker_service.engine` 和 `deploy/<engine>/<model-profile>/compose-profiles.example.yml`。公开示例提供 `qwen38-27b-fp8` 与 `deepseek-v4-flash-0731` 两个模型 profile，各自为 vLLM 和 SGLang 保留静态、可审阅的启动参数：Qwen 使用 TP/DP、Qwen3 reasoning parser；DeepSeek 使用 DeepSeek tokenizer 模式或 reasoning parser，并启用 MoE EP。`BENCHMARK_ENGINE_IMAGE`、`BENCHMARK_MODEL_ID`、`BENCHMARK_TOKENIZER`、`BENCHMARK_IMAGE` 与 `BENCHMARK_HOST_PORT` 是共享输入；模板内部对应 `ENGINE_IMAGE`、`MODEL_ID`、`TOKENIZER_ID`、`SERVED_MODEL_NAME` 等变量。若模型、镜像或引擎版本不匹配现有 profile，应复制最接近的目录，以新的受控名称建立静态模板；不要将任意 shell 参数字符串拼接到环境变量中。

示例内置 `tp1-dp1`、`tp4-dp1`、`tp2-dp2` 三个默认禁用的串行 profile，各自在 `profiles[].environment` 中设置 `GPU_DEVICES`、`TP_SIZE`、`DP_SIZE` 和 `EP_SIZE`。Qwen 模板只使用 TP/DP；DeepSeek 模板额外使用 EP。复制为本地文件后应按实际 GPU 拓扑、模型上下文上限和显存修改并启用所需 profile。每个启用的 profile 都独立执行 `up → 就绪检查 → suite → down`，并产生独立报告和包含合并部署参数的汇总项。模板的 `MAX_MODEL_LEN`、`GPU_MEMORY_UTILIZATION`（vLLM）及 `MEM_FRACTION_STATIC`（SGLang）均有安全的示例默认值，也可通过本地环境变量覆盖。示例还内置一个默认启用的低成本 smoke 及 32K/2K、64K/4K、128K/8K、240K/16K 并发矩阵，所有昂贵矩阵用例默认禁用。配置变量在加载时展开，即使只运行 `--validate-config` 也必须设置全部 `BENCHMARK_*` 变量。

```zsh
export BENCHMARK_ENGINE=vllm # 或 sglang
export BENCHMARK_MODEL_PROFILE=qwen38-27b-fp8 # 或 deepseek-v4-flash-0731
export BENCHMARK_ENGINE_IMAGE='reviewed-engine-image'
export BENCHMARK_IMAGE='reviewed-benchmark-image'
export BENCHMARK_MODEL_ID='model-id'
export BENCHMARK_TOKENIZER='model-or-tokenizer-id'
export BENCHMARK_HOST_PORT=8000

.venv/bin/python benchmark/benchmark.py \
  --config examples/benchmark-config-automation.example.json --validate-config
```

需要比较同一引擎的多个拓扑时，分别参考 [vLLM profile 示例](examples/benchmark-config-compose-profiles-vllm.example.json) 与 [SGLang profile 示例](examples/benchmark-config-compose-profiles-sglang.example.json)。两者均使用上述中立的镜像、模型、GPU、TP、DP 变量接口，并复用一个 `compose_files` bundle；profile 环境变量优先于 base 环境变量。模板位于 [`examples/deploy/vllm/`](examples/deploy/vllm/) 和 [`examples/deploy/sglang/`](examples/deploy/sglang/)。所有启用的 profile 会按声明顺序依次执行 `up → /v1/models 就绪检查 → suite → down`；每个 profile 得到带 profile 后缀的独立报告，另有包含各 profile 合并部署参数的 `-profiles-summary` 聚合报告。`--fail-fast` 会在第一个失败 profile 后将余下启用 profile 标记为 `not_run`；默认会继续下一个 profile。`automation.max_total_duration_seconds` 作用于每个 profile 的 suite，而非所有 profile 的共享总时长。默认聚合报告与 profile 报告必须是带 `{timestamp}` 的本地路径，避免多个服务生命周期争用同一输出；启用 `automation.resume: true` 后，固定根路径会自动生成 `<根名>-<profile>.json` 和 `<根名>-profiles-summary.json`，已完成 profile 不会重新启动 Docker，只有中断、失败或未完成 profile 重跑。

`compose_files` 和 `env_file` 必须是相对 suite 文件所在目录的常规本地文件，且不得逃逸出该目录树；每个 `project_name` 只能使用小写字母、数字、`_` 和 `-`，并且不得与另一个 profile 重复。启动前工具会执行 `docker compose config -q` 并拒绝接管已有 project；启动命令固定包含 `--no-build --pull never`，不会自动构建、拉取镜像或删除 volumes。报告不记录 Compose 或 `.env` 文件原文，但会记录前述合并后的部署环境参数；名称包含 `ACCESS_KEY`、`API_KEY`、`PASSWORD`、`SECRET`、`SESSION_TOKEN`、`TOKEN` 或 S3 endpoint 的值固定写为 `<redacted>`。不要将其他敏感值放入 `base_environment` 或 profile 环境变量。Docker daemon 对宿主机拥有高权限，只可使用已审阅的本地 Compose 文件。

#### 可选：在 benchmark 容器中执行

默认在宿主机运行 `llm-benchmark`。若需要隔离 Python、tokenizer 和网络命名空间，可在 `docker_service.benchmark_container` 显式开启测试容器：

```json
{
  "benchmark_container": {
    "enabled": true,
    "service": "benchmark",
    "api_base": "http://inference:8000/v1",
    "timeout_seconds": 3600
  }
}
```

启用后，每个 profile 按 `up → 宿主机 /v1/models 就绪 → docker compose run --rm --no-deps benchmark → down` 执行。`service` 必须是 Compose 中以 `llm-benchmark` 作为 entrypoint 的 service；工具固定挂载一个只读的 profile 专用临时配置到 `/benchmark-input/suite.json`，并把本地报告目录挂载到 `/benchmark-output`。默认它固定传入 `--config`、`--report`、`--no-resume` 和 `--progress off`，不会接受任意命令或 shell 字符串；启用 `automation.resume: true` 时才移除 `--no-resume`，保留容器在中断前写入的 case checkpoint。Compose service 的镜像必须预先构建或拉取，执行时使用 `--no-tty`、`--pull never`。

`benchmark_container.api_base` 是**测试容器内部**访问推理服务的地址。两个公开模板统一使用 `http://inference:8000/v1`；suite 的 `defaults.api_base` 仍供宿主机执行 `/v1/models` readiness，通常是已发布端口（例如 `http://localhost:8000/v1`）。开启容器模式时，工具会在临时容器配置中用前者覆盖 `defaults.api_base`，以避免使用宿主机的 loopback 地址。容器退出码和状态会写入独立 profile 报告；若容器未写报告、失败或被中断，工具仍会清理推理服务。

### 宿主机硬件清单

每份 suite 报告在 `environment.host_inventory` 下保存一次尽力采集的宿主机清单；多 profile 的独立报告和 `-profiles-summary` 聚合报告均包含该字段。字段包括 CPU（架构、型号、逻辑/物理核心和可用时的频率）、内存与 swap、根文件系统容量、Linux 静态块设备、网卡名称/MTU/状态/速率/驱动，以及加速器型号、驱动和显存。采集失败不会中断测试，而会在 `collection_errors` 或对应 `accelerators.probes` 中以安全的错误类型或状态记录。

加速器采用可扩展 collector 注册表：默认尝试 NVIDIA `nvidia-smi`、华为 `npu-smi info`，以及沐曦候选工具 `mx-smi -L` 和 `mthreads-gmi -L`。缺少厂商命令时相应 probe 为 `unavailable`，无需安装额外 Python 依赖。后续厂商可实现 `AcceleratorCollector.collect(command_runner)` 并返回 `AcceleratorProbeResult` 后注册到 `DEFAULT_ACCELERATOR_COLLECTORS`；返回结构无效的扩展也会被隔离为非致命采集错误。

为减少不必要的暴露，清单不记录 GPU UUID、序列号、进程或原始命令输出，也不记录网卡 MAC/IP、磁盘挂载来源、Compose 环境变量或密钥。benchmark 容器模式下，最终报告会以父进程采集的宿主机清单覆盖容器视角，避免容器 `/proc`、磁盘和网络命名空间造成误导。

### 2. 选择与目标匹配的现有配置

| 目标 | 配置文件 | 内容 |
| --- | --- | --- |
| 通用 API 回归套件 | [`benchmark-config.example.json`](examples/benchmark-config.example.json) | 冒烟、缓存、Decode 矩阵、混合负载与禁用的昂贵用例。 |
| standard-v1 标准评测 | [`benchmark-config-standard-v1.json`](examples/benchmark-config-standard-v1.json) | 六个固定 workload 的版本化性能协议；报告包含可比较的 `standard_summary`。 |
| 单机无人值守编排 | [`benchmark-config-automation.example.json`](examples/benchmark-config-automation.example.json) | 通过 `BENCHMARK_ENGINE` 与 `BENCHMARK_MODEL_PROFILE` 选择专用模型模板；内置 smoke 和 32K/64K/128K/240K 并发矩阵，昂贵用例默认禁用。 |
| 单机 Compose vLLM 多拓扑编排 | [`benchmark-config-compose-profiles-vllm.example.json`](examples/benchmark-config-compose-profiles-vllm.example.json) | 使用 vLLM 专用启动命令，以通用 `GPU_DEVICES`、`TP_SIZE`、`DP_SIZE` 覆盖拓扑参数；所有昂贵 profile 默认禁用。 |
| 单机 Compose SGLang 多拓扑编排 | [`benchmark-config-compose-profiles-sglang.example.json`](examples/benchmark-config-compose-profiles-sglang.example.json) | 使用 SGLang 专用启动命令，以通用 `GPU_DEVICES`、`TP_SIZE`、`DP_SIZE` 覆盖拓扑参数；所有昂贵 profile 默认禁用。 |
| 128K Prefill / Decode 峰值吞吐 | [`benchmark-config-throughput-sweep-128k-2k.json`](examples/benchmark-config-throughput-sweep-128k-2k.json) | 随机 128K 输入 / 2K 输出的吞吐扫描。 |
| 128K / 2K、70% 缓存命中的 SLO 容量 | [`benchmark-config-slo-capacity-128k-2k-cache-hit-0.7.json`](examples/benchmark-config-slo-capacity-128k-2k-cache-hit-0.7.json) | 随机共享前缀、线性精扫和候选确认。 |
| 128K 并发阶梯 | [`benchmark-config-concurrency-staircase-128k.json`](examples/benchmark-config-concurrency-staircase-128k.json) | 固定 128K 请求形状的多档并发测试。 |
| 256K 并发阶梯 | [`benchmark-config-concurrency-staircase-256k.json`](examples/benchmark-config-concurrency-staircase-256k.json) | 固定 256K 请求形状的多档并发测试。 |
| 64K / 128K / 240K 并发矩阵 | [`benchmark-config-concurrency-matrix-64k-128k-240k.json`](examples/benchmark-config-concurrency-matrix-64k-128k-240k.json) | 多上下文长度和并发组合。 |
| 32K / 64K / 128K / 240K 并发矩阵 | [`benchmark-config-concurrency-matrix-32k-64k-128k-240k.json`](examples/benchmark-config-concurrency-matrix-32k-64k-128k-240k.json) | 2K / 4K / 8K / 16K 输出；每组覆盖 18 个指定并发档位，70% 共享前缀，正式请求数为并发两倍；默认启用，失败时停止当前长度组的剩余档位。 |
| 128K / 2K P/D 分离评估 | [`benchmark-config-pd-ratio-128k-2k.json`](examples/benchmark-config-pd-ratio-128k-2k.json) | 分别测量单实例 Prefill/Decode 并给出 P:D 实例比例和调度参数建议；默认禁用。 |
| 混合负载 API 压测 | [`benchmark-config-mixed-workload.json`](examples/benchmark-config-mixed-workload.json) | 随机混合短入长出、中等请求与长入短出；默认禁用。 |

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

配置模式支持 `--tag`、`--case 'pattern-*'`、`--report PATH_OR_S3_URI`、`--no-resume`、`--resume-allow-config-changes` 和 `--fail-fast`。相对本地报告路径以配置文件所在目录为基准；固定本地路径默认会恢复此前成功且无请求级失败的用例。执行计划变更时默认拒绝恢复；确认当前配置只应用于后续未完成用例时，使用 `--resume-allow-config-changes`。它会按展开 case 身份保留无请求级失败的已通过用例，即使参数已变，并重跑失败、跳过、中断和新增用例。

### 3. 吞吐与 SLO 容量扫描策略

扫描会在多个并发档位执行真实请求。长上下文场景的单次请求成本很高，应先复制示例为 `*.local.json`，以 `--validate-config` 和 `--list-cases` 确认配置，再按小并发和较小 `sweep.max_concurrency` 试运行。

| 场景 / preset | 作用 | 并发与停止方式 | 关键结果 |
| --- | --- | --- | --- |
| `prefill-sweep` | 查找长输入、单输出 token 负载的峰值 Prefill 吞吐。 | 从 `1, 2, 4, ...` 扫到 `sweep.max_concurrency`；在合格档位的吞吐相对当前最佳值提升不足 5% 时提前停止。 | `best_concurrency`、`best_throughput` 和 `history` 中每档的 Prefill 吞吐、TTFT、QPS、失败率。 |
| `decode-sweep` | 查找多输出 token 负载的峰值 Decode 吞吐。 | 与 Prefill 相同；实际 batch 的所有输出上限必须都大于 1。 | `best_concurrency`、`best_throughput` 和每档 Decode 吞吐。 |
| `slo-capacity-sweep` | 查找同时满足 TTFT、TPOT、Goodput 和失败率门槛的最大并发。 | 先按 `1, 2, 4, ...` 粗扫，首次失败后按所选精扫策略定位和复核候选。 | `max_passing_concurrency`、`confirmed_concurrency`、`selected_metrics` 和完整 `history`。 |

`prefill-sweep` 与 `decode-sweep` 按**实际请求的输出上限**选择指标：全部为 1 token 时测 Prefill，全部大于 1 token 时测 Decode；同一轮混合两种输出长度会报错。因此 Prefill 配置应显式设置 `random_output_len: 1`，Decode 配置的 `random_output_len`（或实际生成的输出长度）必须始终大于 1。吞吐扫描的 5% 平台停止条件是快速定位峰值的启发式，而非逐并发穷举；当前实现不提供关闭该早停条件的选项，应结合 `history` 判断是否已充分覆盖目标并发范围。

每个正式并发档位默认发送 `max(2 × concurrency, 4)` 个请求，可通过 `sweep.requests_per_round`（CLI：`--sweep-requests-per-round`）固定覆盖。`warmup.rounds` 和 `warmup.requests_per_round` 不计入报告结果：对 `single` 随机共享前缀场景，它们控制每轮 `max_tokens: 1` 的前缀填充请求数量；对 sweep/SLO，它们控制正式扫描前的 workload 预热。基础服务预热始终只发送一个短请求；预热失败会输出警告，但正式扫描仍会继续。

SLO 容量扫描的单轮只有同时满足以下条件才通过：请求总数有效、`goodput_pct` 不低于要求值，且 `failure_rate` 不高于 `max_failure_rate`。Goodput 以每个请求的 TTFT、对多 token 输出的 TPOT 和输出 token 可验证性判断。`min_goodput_pct: 0` 在普通场景表示不启用 suite Goodput 质量门禁；但在 `slo-capacity-sweep` 中表示严格要求 **100%** 请求达到 SLO，只有设置正数才会放宽容量边界。

| `slo_capacity.strategy` | 适用场景与流程 | 取舍 |
| --- | --- | --- |
| `linear` | 从最后一个粗扫通过档到首个粗扫失败档按 `linear_step` 扫描；首次新失败后，额外逐并发探测最多 `confirm_window` 个后续档位。 | 默认 `linear_step: 1` 会逐并发检查，较适合噪声较大或需要细粒度边界的场景；步长大于 1 时会跳过部分并发值。 |
| `binary-confirm`（默认） | 在粗扫边界内二分定位候选，再逐并发检查候选两侧 `confirm_window` 范围。 | 请求量更少，适合近似单调的容量曲线；存在明显非单调波动时，应增大确认窗口或使用逐并发的 `linear`。 |

两种策略都会将候选按并发从高到低进行确认；`confirm_rounds` 是候选要求通过的总测量轮数，任一复测失败会回退到下一个已通过候选。`confirm_rounds: 1` 不发送额外复测。容量结果中的 `max_passing_concurrency` 是按所选探测路径验证的最大通过并发，不应将其视为未采样并发也必然通过的保证。

配置中可使用以下嵌套字段；CLI 等价参数分别为 `--sweep-*` 和 `--slo-capacity-*`：

```json
{
  "sweep": {
    "max_concurrency": 128,
    "requests_per_round": null
  },
  "slo_capacity": {
    "strategy": "binary-confirm",
    "linear_step": 1,
    "confirm_window": 8,
    "confirm_rounds": 3
  }
}
```

### 4. P/D 分离评估

[`benchmark-config-pd-ratio-128k-2k.json`](examples/benchmark-config-pd-ratio-128k-2k.json) 以平均 128K 输入、2K 输出为例，在**同一** OpenAI 兼容服务上分别测量 Prefill（业务输入、1 输出）和 Decode（相同业务输入、业务输出上限）的单实例客户端观测吞吐与延迟。两轮都使用发送前 `DatasetBatch` 的实际 token 长度计算初始 P:D 容量比例；`avg_output_tokens` 会完整传给 Decode 请求，不会静默截断，且 `ignore_eos` 遵循配置。它仍不是实际的分离部署压测：Decode 请求会重新执行普通 API 请求的首 token 前处理，不会启动 Prefill/Decode 实例、交接 KV Cache 或计入 KV 传输和 router 开销。因此 P:D 比例及 `max-num-seqs` / `max-num-batched-tokens` 仅是与业务形状一致的初始 sizing 建议，必须在真实分离部署和目标 SLO 下联合压测验证。

该用例默认 `enabled: false`，以防产生实际 API 流量。复制后，设置服务地址、模型、tokenizer、总 GPU 数和 TP 大小；完成无流量检查后再将本地副本中的 `enabled` 改为 `true`：

```zsh
cp examples/benchmark-config-pd-ratio-128k-2k.json \
  examples/benchmark-config-pd-ratio-128k-2k.local.json
# 编辑本地副本的 api_base、model、tokenizer、defaults.tp_size 和 params.pd.total_gpus

.venv/bin/python benchmark/benchmark.py \
  --config examples/benchmark-config-pd-ratio-128k-2k.local.json --validate-config
.venv/bin/python benchmark/benchmark.py \
  --config examples/benchmark-config-pd-ratio-128k-2k.local.json --list-cases

# 确认负载与服务成本后，将本地副本中 cases[0].enabled 改为 true，再执行：
.venv/bin/python benchmark/benchmark.py \
  --config examples/benchmark-config-pd-ratio-128k-2k.local.json
```

### 5. 混合负载评估

[`benchmark-config-mixed-workload.json`](examples/benchmark-config-mixed-workload.json) 提供一个独立的混合负载 API 示例：默认使用随机 token 数据集，以 30% 短输入长输出、50% 中等请求和 20% 长输入短输出组成 32 个请求，并发度为 8。`workload_mix` 中的 `input_tokens`、`output_tokens` 和 `weight` 分别表示每类请求的输入长度、输出上限和分配权重；实际请求会按种子随机打散，报告同时记录配置形状与 tokenizer 重编码后的实际长度。

该用例默认 `enabled: false`，不会因配置校验或列出用例而发送请求。复制后设置服务地址、模型、tokenizer 和鉴权环境变量，再启用本地副本中的 case：

```zsh
cp examples/benchmark-config-mixed-workload.json \
  examples/benchmark-config-mixed-workload.local.json
# 编辑本地副本的 api_base、model、tokenizer 和鉴权设置

.venv/bin/python benchmark/benchmark.py \
  --config examples/benchmark-config-mixed-workload.local.json --validate-config
.venv/bin/python benchmark/benchmark.py \
  --config examples/benchmark-config-mixed-workload.local.json --list-cases

# 确认服务和负载成本后，将本地副本中 cases[0].enabled 改为 true，再执行：
.venv/bin/python benchmark/benchmark.py \
  --config examples/benchmark-config-mixed-workload.local.json
```

### 6. 单机无人值守编排

[`benchmark-config-automation.example.json`](examples/benchmark-config-automation.example.json) 展示 suite 根级 `automation` 配置。复制它为 `*.local.json`，替换 endpoint、模型和 tokenizer，核对预算后再启用其中的 case。`--validate-config` 与 `--list-cases` 会展开并计算计划预算，但绝不执行 HTTP preflight 或 benchmark 请求。

```json
{
  "automation": {
    "enabled": true,
    "budget": {
      "max_total_requests": 5000,
      "max_estimated_output_tokens": 2000000
    },
    "api_preflight": {
      "enabled": true,
      "timeout_seconds": 5
    },
    "max_total_duration_seconds": 3600
  }
}
```

`automation.budget` 可省略或设为 `null`，此时不会限制请求数或预估输出 token；若显式启用预算，则 `max_total_requests` 与 `max_estimated_output_tokens` 必须同时设置。带 `{timestamp}` 的 `report.path`（或 `--report` 覆盖值）仍是默认的独立报告路径。预算只计算启用的 `api` case：包含正式请求、预热请求和其请求输出上限；扫描及容量搜索按保守上界估算。超限时工具会在发送 benchmark 或 preflight 流量前写出 `run.state: "rejected"` 的报告并返回状态码 `2`。默认 automation 不恢复旧 checkpoint；需要恢复时应按前述要求显式设置 `automation.resume: true` 与稳定的本地报告路径。

在预算通过后，工具会使用与正式请求相同的 bearer header 对每个 API 目标执行 `GET {api_base}/models`。该请求必须返回 2xx 和包含 `data` 数组的 JSON；目标模型没有列在数组中只记录 `model_listed: false`，不会阻止运行。preflight 失败会写出 `run.state: "preflight_failed"` 并返回 `2`，不会发送 benchmark 请求。离线 case 不执行 preflight。

`max_total_duration_seconds` 是协作式 case 边界期限：已开始的 case 会自然结束，随后尚未开始的 runnable case 会以 `skip_reason: "automation deadline exceeded"` 写入报告，运行状态为 `timed_out` 且退出码为 `124`。它不是对流式请求的强制取消，也不能代替底层客户端超时。

所有 suite 报告都包含无密钥的 `run` 与 `provenance` 对象。它们记录 run ID、automation 决策、preflight 结果、期限状态、解析后的报告目标和 `execution_plan_sha256`，便于单机调度器归档和审计；不会保存 `api_key`、展开后的密钥配置或 S3 凭据。

### 7. 报告输出与 S3 存储

#### 本地报告

配置文件中的 `report.path` 和 CLI 的 `--report` 都支持本地文件路径。保持 [`examples/benchmark-config.example.json`](examples/benchmark-config.example.json) 中的本地 `report.path` 作为默认选择；也可以通过 CLI 覆盖输出路径：

```zsh
llm-benchmark \
  --config examples/benchmark-config.local.json \
  --report reports/benchmark.json
```

本地固定路径使用原子替换和 POSIX 文件锁，适合单机 checkpoint 恢复。固定路径默认会恢复此前成功且无请求级失败的用例；包含 `{timestamp}` 的路径会生成带微秒和进程 ID 的唯一名称，正常情况下不会恢复旧 checkpoint。它是唯一命名约定，不是存储后端的严格 create-only 保证。

#### S3 报告

将 `report.path` 或 `--report` 设置为 `s3://bucket/key` 即可将 JSON 报告写入 S3-compatible 存储。`bucket` 名称和对象 key 都直接配置在 URI 中，例如：

```json
{
  "report": {
    "path": "s3://my-benchmark-reports/reports/benchmark-{timestamp}.json"
  }
}
```

其中 `my-benchmark-reports` 是可替换的 bucket 名称，`reports/benchmark-{timestamp}.json` 是对象 key。使用 S3 时，建议复制示例为本地配置并先执行无流量校验：

```zsh
cp examples/benchmark-config.example.json examples/benchmark-config.s3.local.json
# 编辑本地副本：将 report.path 改为 s3://my-benchmark-reports/reports/benchmark-{timestamp}.json
llm-benchmark --config examples/benchmark-config.s3.local.json --validate-config
llm-benchmark --config examples/benchmark-config.s3.local.json
```

S3 凭据、region 和 endpoint 只通过以下 `BENCHMARK_S3_*` 环境变量配置；AK/SK 绝不会写入 JSON、报告或日志：

- `BENCHMARK_S3_ACCESS_KEY_ID` 与 `BENCHMARK_S3_SECRET_ACCESS_KEY`：成对设置的访问密钥。
- `BENCHMARK_S3_SESSION_TOKEN`：可选的临时凭据会话令牌，必须与 AK/SK 一起使用。
- `BENCHMARK_S3_ENDPOINT_URL`：可选的 S3-compatible HTTP(S) endpoint，例如阿里云 OSS endpoint。
- `BENCHMARK_S3_REGION`：可选区域；使用第三方服务时应设置为其对应区域。

未设置自定义 AK/SK 时，boto3 继续使用标准 AWS 凭据链（`AWS_*` 环境变量、共享 credentials/config 文件、实例或 Pod IAM role 等）。例如使用阿里云 OSS：

```zsh
export BENCHMARK_S3_ENDPOINT_URL='https://oss-cn-hangzhou.aliyuncs.com'
export BENCHMARK_S3_REGION='cn-hangzhou'
export BENCHMARK_S3_ACCESS_KEY_ID='your-access-key-id'
export BENCHMARK_S3_SECRET_ACCESS_KEY='your-access-key-secret'
# 将 examples/benchmark-config.s3.local.json 的 report.path 设为 s3://your-bucket/reports/benchmark-{timestamp}.json
llm-benchmark --config examples/benchmark-config.s3.local.json
```

S3 URI 必须同时包含 bucket 和 object key，且不接受 query、fragment 或 URI 内嵌凭据。S3 路径支持读取同一对象来恢复 checkpoint，但没有分布式锁：同一个 `s3://bucket/key` 在任意时刻只能由一个 benchmark 进程写入。包含 `{timestamp}` 的 S3 路径会生成唯一命名的对象 key，正常情况下不会恢复旧 checkpoint；S3 `put_object` 仍允许具有相同 key 的写入覆盖，因此它不是严格 create-only 保证。

#### 本地报告查看器

`web/ui/` 使用 React + Vite 提供本地报告页面，Python 静态服务只负责读取 `web/runs/`。若要开发或重新构建前端：

```zsh
cd web/ui
pnpm install
pnpm build
cd ../..
```

在仓库根目录可用下面的一键初始化脚本安装 Python、Web UI 与 pre-commit 依赖：

```zsh
web/bootstrap.sh
```

使用下面的一键脚本安装缺失的前端依赖，自动检测前端是否需要重建，然后启动同一个本地服务：

```zsh
web/start-web.sh
```

如需修改端口，可运行：

```zsh
web/start-web.sh --host 127.0.0.1 --port 8080
```

执行如下命令后会自动列出固定到 `web/runs` 的 JSON 报告文件：

```zsh
uv run --no-project .venv/bin/python -m benchmark.web.serve
```

浏览器打开 `http://127.0.0.1:8000/`。页面提供报告概览、状态过滤、用例排序、延迟/吞吐趋势和单个用例详情。把新的 JSON 报告复制进 `web/runs/` 即可刷新列表；该页面只读取本地文件，不会发送推理请求。默认只绑定 loopback，不要把该未鉴权服务暴露到局域网。

## 数据集与长度语义

`text`（默认）使用中文填充文本。`context_len` 控制目标输入长度，`max_tokens` 控制输出上限；开启 `share_prefix` 后可用 `prefix_ratio` 设置共享部分。

`random` 生成可复现的 token 序列，更适合精确压力测试。它在以下现有配置中使用：

- [`benchmark-config-throughput-sweep-128k-2k.json`](examples/benchmark-config-throughput-sweep-128k-2k.json)
- [`benchmark-config-slo-capacity-128k-2k-cache-hit-0.7.json`](examples/benchmark-config-slo-capacity-128k-2k-cache-hit-0.7.json)
- [`benchmark-config-concurrency-staircase-128k.json`](examples/benchmark-config-concurrency-staircase-128k.json)
- [`benchmark-config-concurrency-staircase-256k.json`](examples/benchmark-config-concurrency-staircase-256k.json)
- [`benchmark-config-standard-v1.json`](examples/benchmark-config-standard-v1.json)
- [`benchmark-config-concurrency-matrix-32k-64k-128k-240k.json`](examples/benchmark-config-concurrency-matrix-32k-64k-128k-240k.json)
- [`benchmark-config-pd-ratio-128k-2k.json`](examples/benchmark-config-pd-ratio-128k-2k.json)
- [`benchmark-config-mixed-workload.json`](examples/benchmark-config-mixed-workload.json)

对 `random` 数据集，`random_input_len`、`random_output_len` 和 `random_prefix_len` 是权威参数，分别表示独有输入、输出上限和共享前缀；它们优先于通用的 `context_len`/`max_tokens`。工具会在发送前以本地 tokenizer（不添加 special token）反复 decode/re-encode 并补充非 special token，直到最终 prompt 严格达到目标长度；如果 tokenizer 在有限次修复内无法产生该长度，工具会报错而不会静默发送长度不符的请求。API 服务端仍可因 chat template 或不同 tokenizer 而报告不同的 `usage.prompt_tokens`。`share_prefix` 与 `prefix_ratio` 仅对 `text` 生效。

## 配置说明

每个 suite 配置的根对象使用 `version: 1`，并可定义：

- `defaults`：所有 case 的后端、数据集、SLO 和预热默认值。
- `cases`：一个或多个基准用例；可用 `enabled: false` 暂停昂贵用例。
- `matrix`：对指定参数做笛卡尔积展开。
- `repeat`：重复运行同一个 case。
- `continue_on_error`：是否在任一 case 失败后停止整个 suite，默认为 `true`。
- `failure_policy`：失败策略；默认 `continue`。设为 `stop-current-matrix` 时，质量门禁失败或执行异常会将同一原始 matrix case 的后续展开变体写为 `skipped`，并继续下一个 matrix 组。
- `matrix_failure_confirm_rounds`：`stop-current-matrix` 下同一 matrix 变体的失败确认总次数，默认 `1`。设为大于 `1` 时，只要任一次成功就继续下一个变体；全部失败后只保留最后一次的结果或错误，再跳过该 matrix 组剩余变体。
- `protocol`：可选的版本化评测协议声明。当前支持 `{ "id": "standard-v1" }`。
- `standard_workload`：`standard-v1` case 的固定 workload ID。
- `automation`：可选的单机无人值守策略；启用后需要预算和唯一命名报告路径。
- `report`：报告路径、缩进和请求详情保存策略。

`standard-v1` 必须包含且仅包含 `latency-short`、`prefill-long-context`、`decode-long-output`、`prefix-cache`、`concurrency-capacity` 和 `mixed-production` 六个启用的 workload；不允许使用 `matrix` 或 `repeat` 改变其合同。报告会额外写入 `standard_summary`：其中仅保留协议 ID、workload 合同摘要、状态和固定白名单指标，不包含时间戳、主机名、绝对路径、端点或模型名，可作为后续 baseline compare 的稳定输入。比较两份结果前必须确认其 `protocol_id` 和 `contract_sha256` 一致。请复制 [`benchmark-config-standard-v1.json`](examples/benchmark-config-standard-v1.json) 为本地配置，设置目标服务后先执行 `--validate-config` 和 `--list-cases`；运行会发送真实请求。

`continue_on_error: false` 或命令行 `--fail-fast` 始终优先于 `failure_policy` 与 `matrix_failure_confirm_rounds`，会在首次失败时按既有行为停止整个 suite，而不是确认当前变体或只停止当前 matrix 组。

请以 [`examples/benchmark-config.example.json`](examples/benchmark-config.example.json) 为 schema 参考，而不是复制 README 中的片段。配置中的 `${VAR}` 会读取环境变量；需要密钥时使用 `api_key_env`，不要使用明文 `api_key`。

## 指标说明

- **TTFT**：从请求发出到收到第一个内容 token 的时间。
- **TPOT**：相邻内容 token 的平均耗时；少于两个输出 token 时不计算。
- **估算 ITL**：按相邻内容 SSE Chunk 的到达间隔，并在该 round 的所有网络流结束后对每个后续 Chunk 本地分词加权得到；它是客户端观测的近似值。
- **Prefill 吞吐**：成功请求从发出到首 token 的活动区间内处理的 prompt token 速率。
- **Decode 吞吐**：成功请求从首 token 到最后一个内容 token 的活动区间内生成的 token 速率（不含首 token）。
- **整体吞吐 / QPS**：整个测试窗口的 token 交付速率 / 完成请求速率。
- **Goodput**：全部请求中同时满足 TTFT 和（输出超过一个 token 时）TPOT SLO 的比例；没有首 token 的请求计入失败。

阶段活动窗口包含客户端可观测的排队与网络延迟，不等同于服务端 scheduler 的纯 GPU 计算时间。若输出 token 数无法通过服务端 usage 或本地 tokenizer 可靠确定，依赖该计数的指标会被省略而非估算。

### 服务端 KV Cache 命中率

API round 会通过与 chat-completions 相同鉴权头访问服务端 `/metrics`；正式流式请求使用 `aiohttp` 时，该 Prometheus 采样仍使用 `requests`。对 SGLang 暴露的 `sglang:cache_hit_rate`，报告保留测试窗口内活跃快照的 min/avg/max 与分位数；该指标是服务端直接导出的 gauge，缺少原始 hit/query counter 时不能正确换算为 rate。

对支持 `vllm:prefix_cache_hits` 和 `vllm:prefix_cache_queries` counter 的 vLLM，工具使用请求 round 前后相同 Prometheus label series 的 counter 和实际单调时钟窗口，计算 `hit_rate_per_second = ΣΔhit / window_seconds`、`query_rate_per_second = ΣΔquery / window_seconds`，再以 `hit_rate_per_second / query_rate_per_second` 得到命中率百分比。结果写入 `result.server_metrics.metrics.cache_hit_rate`，其 `aggregation` 为 `round_counter_rate`，并包含 `window_seconds`、每秒 hit/query rate 及原始增量，方便审计。

该 vLLM 比率以缓存 token 查询为单位，不是命中请求数。负向 counter delta 会被视为服务重启或 exporter reset 并排除；没有可配对 series、`Δquery=0` 或无有效观测窗口时不记录命中率。若目标 vLLM 版本未导出上述 counters，报告只保留 KV Cache 使用率，无法推导实际命中率。示例配置中“70% 缓存命中”只描述共享前缀工作负载目标，实际服务端命中率仍取决于 prefix caching、逐出、调度与并发。

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

开发和验证均应在仓库根目录执行，并统一使用虚拟环境中的 Python；不要使用系统 `python3`、bare `pip` 或未固定环境的 `pytest`。

开发完成后建议依次执行：

```zsh
# 编译检查
.venv/bin/python -m compileall benchmark tests

# CLI 和配置检查（不会发送模型或 API 请求）
.venv/bin/python benchmark/benchmark.py --help
.venv/bin/python benchmark/benchmark.py \
  --config examples/benchmark-config.example.json --validate-config
.venv/bin/python benchmark/benchmark.py \
  --config examples/benchmark-config.example.json --list-cases

# 运行全部单元测试
.venv/bin/python -m pytest -q

# 检查 Git 差异中的空白错误
/usr/bin/git diff --check
```

如果已安装并配置 pre-commit，可额外运行：

```zsh
.venv/bin/pre-commit run --all-files
```

不要在默认验证流程中启动 API server、watch 进程或真实 GPU/API benchmark；这类场景应由使用者明确配置后单独执行。
