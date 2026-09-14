# GSM8K 本地数据

本目录的 JSONL 文件来自 Hugging Face 数据集 [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)，使用 `main` 配置，许可证为 MIT。

- `openai-gsm8k-train.jsonl`：训练 split，7473 条。
- `openai-gsm8k-test.jsonl`：测试 split，1319 条。
- `_source.json`：记录源仓库、revision、许可证和原始 parquet 的 SHA256。

数据保持原始 `question` / `answer` 字段，未改写内容。GSM8K benchmark 可以直接将 `dataset_path` 指向其中一个 JSONL 文件；推荐使用 test split 作为可复现评测源。如果需要在文档中完整引用数据来源，可访问数据集主页查看 dataset card。
