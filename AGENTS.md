# AGENTS.md

## 项目边界

这是一个面向 vLLM 离线引擎和 OpenAI 兼容 API 的 LLM 推理基准工具。实现与文档使用中文；用户使用说明放在 `README.md`，可运行配置只放在 `examples/`。

- `benchmark/benchmark.py`：CLI、场景执行、指标、报告与配置套件。
- `benchmark/benchmark_datasets.py`：`text` / `random` 数据集及长度、前缀语义。
- `tests/`：不依赖 GPU、模型下载或真实服务的单元测试。

## 本地命令

在仓库根目录运行。首次安装：

```zsh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

修改 Python、配置或文档后，在已激活的虚拟环境中按影响范围执行：

```zsh
python -m compileall benchmark tests
python benchmark/benchmark.py --help
python benchmark/benchmark.py \
  --config examples/benchmark-config.example.json --validate-config
python benchmark/benchmark.py \
  --config examples/benchmark-config.example.json --list-cases
python -m pytest -q
```

不要在自动化修改中启动 watch/server，也不要执行真实 API/GPU 基准；它们会产生流量和成本。`offline` 模式的 `vllm`、`torch` 由使用者按 CUDA、GPU 和系统环境单独安装。

## 实现和配置规则

- 保持 CLI 与配置字段向后兼容。新增参数应包含帮助文本、默认值和校验。
- `benchmark.py` 必须同时支持直接脚本执行和安装后的 `llm-benchmark` 入口。
- GPU、模型和网络相关依赖应按使用路径延迟导入，并提供可操作的缺失依赖错误。
- 修改功能时按职责拆分模块和函数，复用明确的接口或扩展点；避免继续堆叠 `benchmark.py` 中不相关的分支逻辑。
- 每项新增或变更的可观察行为都必须有对应单元测试，至少覆盖正常路径、关键边界和预期错误路径；修复缺陷时先补复现测试。
- 基准长度以 `DatasetBatch` 实际 token 统计为准；`random_input_len` / `random_output_len` 优先于 `context_len` / `max_tokens`。
- 保持配置 schema 的 `version: 1`、`defaults`、`cases`、`matrix`、`repeat` 和 `${VAR}` 展开行为。修改 schema 时同步更新 README、`examples/benchmark-config.example.json` 与测试。
- JSON 使用 2 空格缩进；提交的通用示例不得新增真实密钥、内部地址或本地绝对模型路径。昂贵用例默认 `enabled: false`。
- 默认不得向非 loopback 的明文 HTTP 地址发送 bearer key；仅在用户明确接受风险时使用 `allow_insecure_api_key`。

## 测试和产物

单元测试应 mock tokenizer、HTTP 和 vLLM；真实 GPU/API 场景只能作为显式 integration 测试，不能进入默认测试路径。对 JSON 改动使用 `python3 -m json.tool <file>` 校验。

不要提交基准报告、锁文件、profile、虚拟环境或本地配置/密钥；这些产物由 `.gitignore` 排除。
