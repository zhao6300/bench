# AGENTS.md

> **适用范围：** 仓库根目录及其所有子目录。本文规定 Agent 在本项目中开发、修改、测试和交付时必须遵守的规则。

## 项目概览

这是一个可扩展的 LLM 推理性能测试基准工具，支持本地离线推理和 OpenAI 兼容 API，并为更多推理引擎预留扩展空间。

- 实现和项目文档使用中文；面向用户的说明放在 `README.md`。
- 可运行配置只放在 `examples/`。
- Python 代码的注释和 docstring 必须遵循 [Google Python Style Guide §3.8：Comments and Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)。

### 关键目录

| 路径 | 职责 |
| --- | --- |
| `benchmark/benchmark.py` | CLI、场景执行、指标、报告和配置套件。 |
| `benchmark/benchmark_datasets.py` | `text` / `random` 数据集及其长度和前缀语义。 |
| `tests/` | 不依赖 GPU、模型下载或真实服务的单元测试。 |
| `examples/` | 可运行的公开配置示例。 |

## 强制约束

### 环境与依赖

- **不得使用**系统 `python3`、bare `pip` 或 `pip install`。
- 所有环境管理命令必须通过 `uv` 执行；所有 Python 命令必须使用 `.venv/bin/python`。
- 所有命令均在仓库根目录运行。
- `offline` 模式依赖的 `vllm` 和 `torch` 由使用者按其 CUDA、GPU 和系统环境单独安装。

### 流量、成本与安全

- 不得在自动化修改或默认验证流程中启动 watch、server 或真实 API/GPU 基准；这些操作会产生流量和成本。
- 默认不得向非 loopback 的明文 HTTP 地址发送 bearer key；只有用户明确接受风险时才可使用 `allow_insecure_api_key`。
- S3 的 AK/SK、会话令牌和 endpoint 只能通过环境变量配置；不得写入 JSON、报告、示例、测试夹具或日志。
- 通用示例不得包含真实密钥、内部地址或本地绝对模型路径。

### 兼容性与范围

- 保持 CLI 和配置字段向后兼容。新增参数必须提供帮助文本、默认值和校验。
- `benchmark.py` 必须同时支持直接脚本执行和安装后的 `llm-benchmark` 入口。
- GPU、模型和网络相关依赖必须按使用路径延迟导入，并提供可操作的缺失依赖错误。
- 遵循相邻代码的命名、结构、类型标注和格式；除非任务要求或有明确收益，不做无关重构。

## 开发工作流

### 初始化环境

若尚未安装 `uv`：

```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

首次安装和 hook 配置：

```zsh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/common.txt
uv pip install -r requirements/lint.txt
uv pip install -e .
pre-commit install
```

### 实现要求

- 修改功能时按职责拆分模块和函数，复用明确的接口或扩展点；避免继续在 `benchmark.py` 堆叠无关分支。
- 每次代码变更都必须复核 `requirements/common.txt` 和 `README.md`。涉及依赖、安装、CLI、配置或其他用户可见行为时，必须同步更新对应文件。
- 基准长度以 `DatasetBatch` 的实际 token 统计为准；`random_input_len` / `random_output_len` 优先于 `context_len` / `max_tokens`。

### 配置要求

- 保持配置 schema 的 `version: 1`、`defaults`、`cases`、`matrix`、`repeat` 和 `${VAR}` 展开行为。
- 修改 schema 时，必须同步更新 `README.md`、`examples/benchmark-config.example.json` 和测试。
- JSON 使用 2 空格缩进；昂贵用例默认设置为 `"enabled": false`。

## 代码与文档规范

- Docstring 使用 Google 风格。需要说明参数、返回值或异常时，使用 `Args:`、`Returns:` 和 `Raises:` 小节；不要使用 reStructuredText/Sphinx 的 `:param:`、`:return:` 或 `:rtype:` 字段。
- 注释和 docstring 应简短直接，只保留无法从清晰命名和代码结构推断的信息。
- 删除冗余注释，避免用注释重复代码本身。

## 测试与验证

### 测试设计

- 单元测试必须 mock tokenizer、HTTP 和 vLLM；真实 GPU/API 场景只能作为显式 integration 测试，不能进入默认测试路径。
- 每项新增或变更的可观察行为都必须有对应单元测试，至少覆盖正常路径、关键边界和预期错误路径；修复缺陷时先补复现测试。
- 先设计再编写测试：明确模块职责、输入/输出契约、要防止的失败，以及最低成本的捕获层级（优先 unit，其次 integration，最后 e2e）。
- 优先扩展相邻测试文件、`conftest.py` fixture 和已有 helper；没有合适的相邻 suite 时才新建测试文件。
- 通过公开 API 断言可观察行为，并在测试名或 docstring 中表达测试原因；跳过琐碎 wiring，避免不稳定测试。
- 每个测试只验证一个行为，使用触发该行为所需的最小 setup；若测试 diff 明显大于代码变更，应缩小测试范围。

### 验证命令

修改 Python、配置或文档后，按影响范围运行：

```zsh
.venv/bin/python -m compileall benchmark tests
.venv/bin/python benchmark/benchmark.py --help
.venv/bin/python benchmark/benchmark.py \
  --config examples/benchmark-config.example.json --validate-config
.venv/bin/python benchmark/benchmark.py \
  --config examples/benchmark-config.example.json --list-cases
.venv/bin/python -m pytest -q
.venv/bin/pre-commit run --all-files
```

对 JSON 改动额外执行：

```zsh
.venv/bin/python -m json.tool <file>
```

## 交付要求

- 不提交基准报告、锁文件、profile、虚拟环境或本地配置/密钥；这些产物应由 `.gitignore` 排除。
- 交付前确认变更符合本文件的环境、安全、兼容性、文档和测试要求。
