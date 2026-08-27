# Agent Instructions for Benchmark Project

本文件定义 Agent 在本项目中开发、修改、测试和交付代码时**必须遵守**的指令。Agent 开发 Python 代码时，必须遵循 [Google Python Style Guide §3.8：Comments and Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)。

## 项目边界

这是一个面向 vLLM 离线引擎和 OpenAI 兼容 API 的 LLM 推理基准工具。实现与文档使用中文；用户使用说明放在 `README.md`，可运行配置只放在 `examples/`。

- `benchmark/benchmark.py`：CLI、场景执行、指标、报告与配置套件。
- `benchmark/benchmark_datasets.py`：`text` / `random` 数据集及长度、前缀语义。
- `tests/`：不依赖 GPU、模型下载或真实服务的单元测试。

## Development Workflow

**绝不使用系统 `python3`、bare `pip` 或 `pip install`。** 所有环境管理命令必须通过 `uv` 执行，所有 Python 执行必须使用 `.venv/bin/python`。

在仓库根目录运行。若尚未安装 `uv`：

```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

首次安装与 hook 配置：

```zsh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/common.txt
uv pip install -r requirements/lint.txt
uv pip install -e .
pre-commit install
```

修改 Python、配置或文档后，按影响范围执行：

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

不要在自动化修改中启动 watch/server，也不要执行真实 API/GPU 基准；它们会产生流量和成本。`offline` 模式的 `vllm`、`torch` 由使用者按 CUDA、GPU 和系统环境单独安装。

## 实现和配置规则

- 保持 CLI 与配置字段向后兼容。新增参数应包含帮助文本、默认值和校验。
- `benchmark.py` 必须同时支持直接脚本执行和安装后的 `llm-benchmark` 入口。
- GPU、模型和网络相关依赖应按使用路径延迟导入，并提供可操作的缺失依赖错误。
- 每次代码变更都必须复核 `requirements/common.txt` 和 README；涉及依赖、安装、CLI、配置或用户可见行为时必须同步更新对应文件。
- 修改功能时按职责拆分模块和函数，复用明确的接口或扩展点；避免继续堆叠 `benchmark.py` 中不相关的分支逻辑。
- 每项新增或变更的可观察行为都必须有对应单元测试，至少覆盖正常路径、关键边界和预期错误路径；修复缺陷时先补复现测试。
- 基准长度以 `DatasetBatch` 实际 token 统计为准；`random_input_len` / `random_output_len` 优先于 `context_len` / `max_tokens`。
- 保持配置 schema 的 `version: 1`、`defaults`、`cases`、`matrix`、`repeat` 和 `${VAR}` 展开行为。修改 schema 时同步更新 README、`examples/benchmark-config.example.json` 与测试。
- JSON 使用 2 空格缩进；提交的通用示例不得新增真实密钥、内部地址或本地绝对模型路径。昂贵用例默认 `enabled: false`。
- 默认不得向非 loopback 的明文 HTTP 地址发送 bearer key；仅在用户明确接受风险时使用 `allow_insecure_api_key`。
- S3 的 AK/SK、会话令牌和 endpoint 仅通过环境变量配置；不得写入 JSON、报告、示例、测试夹具或日志。

## 代码风格

- 遵循邻近代码的命名、结构、类型标注和格式风格；除非任务要求或存在明确收益，不做无关重构。
- Docstring 使用 [Google Python Style Guide §3.8：Comments and Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) 的 Google 风格；需要参数、返回值或异常说明时使用 `Args:`、`Returns:`、`Raises:` 小节，不使用 reStructuredText/Sphinx 的 `:param:`、`:return:`、`:rtype:` 字段。
- 注释和 docstring 应简短直接，只保留无法从清晰命名和代码结构推断的信息；删除冗余注释，避免用注释重复代码本身。

## 测试和产物

单元测试应 mock tokenizer、HTTP 和 vLLM；真实 GPU/API 场景只能作为显式 integration 测试，不能进入默认测试路径。对 JSON 改动使用 `.venv/bin/python -m json.tool <file>` 校验。

### 新增或修改测试时

- **先设计，再编写。** 先回答：模块的职责是什么、输入/输出契约是什么、要防止什么失败、以及捕获它的最低成本层级是什么（优先 unit，其次 integration，最后 e2e）。
- **先复用，再创建。** 优先扩展附近的测试文件、`conftest.py` fixture 和已有 helper；只有没有合适的相邻 suite 时才新建测试文件。
- **按意图测试行为。** 通过公开 API 断言可观察结果，并在测试名称或 docstring 中表达测试原因；跳过琐碎 wiring，避免引入不稳定测试。
- **保持最小范围。** 每个测试只验证一个行为，使用触发该行为所需的最小 setup；若测试 diff 明显大于代码变更，应缩小测试范围。

不要提交基准报告、锁文件、profile、虚拟环境或本地配置/密钥；这些产物由 `.gitignore` 排除。
