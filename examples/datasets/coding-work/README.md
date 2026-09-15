# Coding / work 本地数据

本目录保存五类从 Hugging Face 数据集移植的离线工作负载。非公开 Slack webhook 和 API token
已替换为脱敏占位符，`_source.json` 记录来源 slug、revision、许可证（若有），
行数和脱敏后 SHA256。

| 文件 | 用途 |
| --- | --- |
| `openai-humaneval-test.jsonl` | HumanEval 函数补全短入短出场景。 |
| `likaixin-instructcoder-validation.jsonl` | InstructCoder 代码编辑指令场景。 |
| `blazedit-5k-char-train.jsonl` / `blazedit-10k-char-train.jsonl` | 整文件代码改写，覆盖中等和更高输入/输出长度。 |
| `bfcl-v3-simple-live_simple-multiple.jsonl` | BFCL 工具选择与调用触发的生产工作流场景。 |
| `princeton-swe-bench-test.jsonl` | SWE-bench full test split，读取 issue 描述并构造 patch 风格编码工作负载。 |

这些 JSONL 是移植后的标准文本负载，不包含 Hugging Face hub 代码；若需要在文档中完整引用数据来源，请访问各 dataset card。
