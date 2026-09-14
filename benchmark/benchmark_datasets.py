"""Prompt dataset implementations used by the benchmark runner.

The datasets expose one stable request representation so benchmark scenarios can
change how inputs are generated without changing request execution or metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass
import json
import math
import random as random_module
from pathlib import Path
from typing import Any, Protocol, Sequence
import uuid


class TokenizerLike(Protocol):
    """Subset of the tokenizer API required by benchmark datasets."""

    vocab_size: int
    all_special_ids: Sequence[int]

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str: ...

    def num_special_tokens_to_add(self, *args: Any, **kwargs: Any) -> int: ...


@dataclass(frozen=True)
class SampleRequest:
    """A generated benchmark request before it is sent to the API or vLLM."""

    prompt: str
    prompt_len: int
    expected_output_len: int
    request_id: str


@dataclass(frozen=True)
class DatasetBatch:
    """Generated requests plus prefix-cache-related prompt shape metadata."""

    requests: list[SampleRequest]
    shared_prefix_len: int = 0

    @property
    def prompts(self) -> list[str]:
        return [request.prompt for request in self.requests]

    @property
    def prompt_lens(self) -> list[int]:
        return [request.prompt_len for request in self.requests]

    @property
    def output_lens(self) -> list[int]:
        return [request.expected_output_len for request in self.requests]

    @property
    def unique_prefix_len(self) -> int:
        if not self.requests:
            return 0
        return max(0, self.requests[0].prompt_len - self.shared_prefix_len)


class BenchmarkDataset(ABC):
    """Base class for prompt generators used by benchmark scenarios."""

    def __init__(
        self,
        *,
        random_seed: int | None = None,
        dataset_path: str | None = None,
        disable_shuffle: bool = False,
    ) -> None:
        self.random_seed = random_seed
        self.dataset_path = dataset_path
        self.disable_shuffle = disable_shuffle

    @abstractmethod
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        **kwargs: Any,
    ) -> DatasetBatch:
        """Generate a batch of requests for one benchmark round."""

    @staticmethod
    def _require_path(dataset_name: str | None) -> str:
        """Validate that a local data source is declared."""
        if not dataset_name:
            raise ValueError("dataset_path must be provided")
        return dataset_name

    @staticmethod
    def _tokenizer_sequence(tokenizer: TokenizerLike, text: str) -> list[int]:
        """Encode a non-empty prompt and reject unusable tokenizers."""
        tokens = tokenizer.encode(text)
        if not tokens:
            raise ValueError("tokenizer produced no tokens for a dataset request")
        return tokens

    @staticmethod
    def _sample_rows(
        rows: Sequence[Any],
        num_requests: int,
        seed: int | None,
        no_oversample: bool = False,
        disable_shuffle: bool = False,
    ) -> list[Any]:
        """Sample rows reproducibly, cycling when the source is smaller."""
        if num_requests < 1:
            raise ValueError("num_requests must be positive")
        if not rows:
            return []
        rng = random_module.Random(seed)
        indexes = list(range(len(rows)))
        if not disable_shuffle:
            rng.shuffle(indexes)
        if len(indexes) >= num_requests:
            selected = indexes[:num_requests]
        elif no_oversample:
            selected = indexes
        else:
            selected = list(range(num_requests))
            rng.shuffle(selected)
        return [rows[index] for index in selected]

    @staticmethod
    def _shared_token_prefix(tokenizer: TokenizerLike, prompts: Sequence[str]) -> int:
        """Return the token prefix shared by the first two sampled prompts."""
        if len(prompts) < 2:
            return 0
        first = BenchmarkDataset._tokenizer_sequence(tokenizer, prompts[0])
        second = BenchmarkDataset._tokenizer_sequence(tokenizer, prompts[1])
        limit = min(len(first), len(second))
        for index in range(limit):
            if first[index] != second[index]:
                return index
        return limit

    @staticmethod
    def _load_json_records(path: str) -> list[dict[str, Any]]:
        """Read a small local JSON/JSONL file as a list of dictionaries."""
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            value = value.get("data", value.get("items", value.get("records", [])))
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise ValueError("dataset_path must contain a JSON array of objects")
        return value

    @staticmethod
    def _load_dataset_rows(path: str) -> list[dict[str, str]]:
        """Read CSV headers into string records without an optional dependency."""
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("dataset_path CSV has no header row")
            return [dict(row) for row in reader]


class TextDataset(BenchmarkDataset):
    """The original natural-language filler prompt generator.

    This preserves the prior benchmark behaviour: requests use a fixed Chinese
    filler paragraph, while UUID labels create cache misses outside the shared
    prefix portion.
    """

    _FILLER_TEXT = (
        "在计算机科学领域，大规模语言模型的训练与推理涉及诸多复杂的工程问题。"
        "从数据预处理、分布式训练策略，到推理阶段的显存优化与算力调度，"
        "每一个环节都对最终的系统性能产生深远影响。本段文字仅用于填充上下文长度，"
        "以便测试模型在长序列输入下的预填充（prefill）耗时表现。"
    )

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        *,
        input_len: int,
        output_len: int,
        prefix_ratio: float = 1.0,
        share_prefix: bool = False,
        random_shared_prefix: bool = False,
        run_id: str = "",
        **kwargs: Any,
    ) -> DatasetBatch:
        if input_len < 1:
            raise ValueError("input_len must be positive")
        if output_len < 1:
            raise ValueError("output_len must be positive")
        if not 0.0 <= prefix_ratio <= 1.0:
            raise ValueError("prefix_ratio must be between 0 and 1")

        run_id = run_id or (uuid.uuid4().hex[:8] if random_shared_prefix else "")
        requests = []
        shared_prefix_len = int(input_len * prefix_ratio) if share_prefix else 0
        for index in range(num_requests):
            prompt, actual_len, shared_len, _ = self._build_prompt(
                tokenizer,
                target_tokens=input_len,
                prefix_ratio=prefix_ratio,
                request_index=index,
                share_prefix=share_prefix,
                run_id=run_id,
            )
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=actual_len,
                    expected_output_len=output_len,
                    request_id=f"{request_id_prefix}{index}",
                )
            )
            shared_prefix_len = shared_len
        return DatasetBatch(requests, shared_prefix_len=shared_prefix_len)

    def _build_prompt(
        self,
        tokenizer: TokenizerLike,
        *,
        target_tokens: int,
        prefix_ratio: float,
        request_index: int,
        share_prefix: bool,
        run_id: str,
    ) -> tuple[str, int, int, int]:
        if not share_prefix or prefix_ratio <= 0.0:
            random_prefix = f"【测试唯一标识-{uuid.uuid4().hex[:6]}-{request_index}】"
            prefix_ids = tokenizer.encode(random_prefix)
            body_ids = self._filler_token_ids(tokenizer, target_tokens - len(prefix_ids))
            token_ids = (prefix_ids + body_ids)[:target_tokens]
            return tokenizer.decode(token_ids), len(token_ids), 0, len(token_ids)

        shared_tokens_count = int(target_tokens * prefix_ratio)
        unique_tokens_count = target_tokens - shared_tokens_count
        if shared_tokens_count:
            shared_label = (
                f"【公共固定前缀-Qwen36-Prefill-Bench-{run_id}】"
                if run_id
                else "【公共固定前缀-Qwen36-Prefill-Bench】"
            )
            shared_ids = tokenizer.encode(shared_label)
            shared_part = (shared_ids + self._filler_token_ids(
                tokenizer, shared_tokens_count - len(shared_ids)
            ))[:shared_tokens_count]
        else:
            shared_part = []
        if unique_tokens_count:
            unique_ids = tokenizer.encode(
                f"【独特请求后缀-{uuid.uuid4().hex[:6]}-{request_index}】"
            )
            unique_part = (unique_ids + self._filler_token_ids(
                tokenizer, unique_tokens_count - len(unique_ids)
            ))[:unique_tokens_count]
        else:
            unique_part = []
        token_ids = (shared_part + unique_part)[:target_tokens]
        return tokenizer.decode(token_ids), len(token_ids), len(shared_part), len(unique_part)

    def _filler_token_ids(self, tokenizer: TokenizerLike, needed_len: int) -> list[int]:
        if needed_len <= 0:
            return []
        paragraph_ids = tokenizer.encode(self._FILLER_TEXT)
        if not paragraph_ids:
            raise ValueError("tokenizer produced no tokens for the text benchmark filler")
        repeats = math.ceil(needed_len / len(paragraph_ids))
        # Re-tokenize the repeated text, matching the original implementation
        # and preserving tokenization at paragraph boundaries.
        return tokenizer.encode(self._FILLER_TEXT * repeats)[:needed_len]


def parse_range_ratio(value: str | float | tuple[float, float]) -> tuple[float, float]:
    """Parse one ratio or an ``input,output`` ratio pair and validate it."""
    if isinstance(value, tuple):
        ratios = value
    elif isinstance(value, (float, int)):
        ratios = (float(value), float(value))
    else:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) == 1:
            ratios = (float(parts[0]), float(parts[0]))
        elif len(parts) == 2:
            ratios = (float(parts[0]), float(parts[1]))
        else:
            raise ValueError("range_ratio must be one number or input_ratio,output_ratio")
    if any(not 0.0 <= ratio <= 1.0 for ratio in ratios):
        raise ValueError("range_ratio values must be between 0 and 1")
    return float(ratios[0]), float(ratios[1])


def get_sampling_params(
    rng: Any,
    num_requests: int,
    range_ratio: str | float | tuple[float, float],
    input_len: int,
    output_len: int,
    tokenizer: TokenizerLike,
) -> tuple[Any, Any, Any]:
    """Sample reproducible input/output lengths and token-sequence offsets."""
    if num_requests < 1:
        raise ValueError("num_requests must be positive")
    if input_len < 0 or output_len < 1:
        raise ValueError("input_len must be non-negative and output_len must be positive")
    input_ratio, output_ratio = parse_range_ratio(range_ratio)
    special_tokens = int(tokenizer.num_special_tokens_to_add())
    real_input_len = max(0, input_len - special_tokens)

    def sample_lengths(mean: int, ratio: float, minimum: int = 0) -> Any:
        low = max(minimum, math.floor(mean * (1.0 - ratio)))
        high = max(low, math.ceil(mean * (1.0 + ratio)))
        return rng.integers(low, high + 1, size=num_requests)

    return (
        sample_lengths(real_input_len, input_ratio),
        sample_lengths(output_len, output_ratio, minimum=1),
        rng.integers(0, max(1, int(tokenizer.vocab_size)), size=num_requests),
    )


class RandomDataset(BenchmarkDataset):
    """Synthetic text-only data for reproducible serving/throughput benchmarks.

    Input and output lengths are sampled from integer-uniform ranges around the
    configured means. A prefix is generated once, then each request uses a
    deterministic allowed-token sequence. The final decoded prompt is encoded
    without special tokens and repaired until it exactly matches its target.
    """

    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128
    MAX_LENGTH_REPAIR_ATTEMPTS = 64

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "--dataset random requires numpy; install a pinned numpy version "
                "in the benchmark environment."
            ) from exc
        self._np = np
        # Keep this RNG isolated from Python's and NumPy's process-global state.
        # It advances for every batch so later batches receive fresh suffixes.
        self._rng = np.random.default_rng(self.random_seed)
        # Prefixes are cached on the dataset instance. A benchmark runner reuses
        # one instance per sweep, so a configured seed produces one stable
        # shared prefix while each sample still consumes new suffix randomness.
        self._prefix_cache: dict[int, list[int]] = {}

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: str | float | tuple[float, float] = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        batchsize: int = 1,
        max_loras: int | None = None,
        lora_path: str | None = None,
        lora_assignment: str = "random",
        **kwargs: Any,
    ) -> DatasetBatch:
        del no_oversample, max_loras, lora_path, lora_assignment, kwargs
        input_ratio, _ = parse_range_ratio(range_ratio)
        num_special = int(tokenizer.num_special_tokens_to_add())
        real_input_len = max(0, int(input_len) - num_special)
        min_total_input = int(prefix_len) + math.floor(real_input_len * (1.0 - input_ratio))
        if min_total_input < 1:
            raise ValueError(
                "--random-input-len is too small: with tokenizer special tokens "
                f"{num_special} and input range ratio {input_ratio}, the minimum "
                f"total input tokens (prefix + sampled) is {min_total_input}. "
                "Increase --random-input-len and/or --random-prefix-len, or "
                "decrease --random-range-ratio."
            )
        if batchsize != 1:
            raise ValueError("batched random dataset requests are not supported by this benchmark")

        vocab_size = int(tokenizer.vocab_size)
        prohibited_tokens = set(int(token) for token in tokenizer.all_special_ids)
        allowed_tokens = self._np.array(
            [token for token in range(vocab_size) if token not in prohibited_tokens],
            dtype=int,
        )
        if len(allowed_tokens) == 0:
            raise ValueError("tokenizer has no non-special vocabulary tokens")
        # Resolve the shared prefix before sampling batch-specific values so the
        # prefix is determined only by this dataset instance's seed/tokenizer,
        # rather than by the request count of its first batch.
        prefix_token_ids = self.get_prefix(tokenizer, allowed_tokens, prefix_len)
        input_lens, output_lens, offsets = get_sampling_params(
            self._rng, num_requests, range_ratio, input_len, output_len, tokenizer
        )

        requests = []
        for index in range(num_requests):
            prompt, total_input_len = self.generate_token_sequence(
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                input_len=int(input_lens[index]),
                offset=int(offsets[index]),
                index=index,
                allowed_tokens=allowed_tokens,
            )
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[index]),
                    request_id=f"{request_id_prefix}{index}",
                )
            )
        return DatasetBatch(requests, shared_prefix_len=len(prefix_token_ids))

    def get_prefix(self, tokenizer: TokenizerLike, allowed_tokens: Any, prefix_len: int) -> list[int]:
        if prefix_len <= 0:
            return []
        cached_prefix = self._prefix_cache.get(prefix_len)
        if cached_prefix is not None:
            return cached_prefix
        prefix_tokens = allowed_tokens[
            self._rng.integers(0, len(allowed_tokens), size=prefix_len)
        ].tolist()
        _, adjusted_tokens = self._decode_to_target_len(
            tokenizer, prefix_tokens, prefix_len, allowed_tokens
        )
        self._prefix_cache[prefix_len] = adjusted_tokens
        return adjusted_tokens

    def generate_token_sequence(
        self,
        *,
        tokenizer: TokenizerLike,
        prefix_token_ids: list[int],
        prefix_len: int,
        input_len: int,
        offset: int,
        index: int,
        allowed_tokens: Any,
    ) -> tuple[str, int]:
        inner_seq = allowed_tokens[
            (offset + index + self._np.arange(input_len)) % len(allowed_tokens)
        ].tolist()
        target_len = prefix_len + int(input_len)
        prompt, adjusted_tokens = self._decode_to_target_len(
            tokenizer, prefix_token_ids + inner_seq, target_len, allowed_tokens
        )
        return prompt, len(adjusted_tokens)

    def _decode_to_target_len(
        self,
        tokenizer: TokenizerLike,
        token_sequence: list[int],
        target_token_len: int,
        allowed_tokens: Any,
    ) -> tuple[str, list[int]]:
        """Return a decoded prompt whose final encoding exactly meets the target.

        Args:
            tokenizer: Tokenizer used to encode and decode the prompt text.
            token_sequence: Initial non-special token IDs for the prompt.
            target_token_len: Required final token count without special tokens.
            allowed_tokens: Non-special token IDs available for deterministic padding.

        Returns:
            The submitted prompt text and its final verified token IDs.

        Raises:
            ValueError: If the tokenizer cannot produce the requested length
                within the bounded repair attempts.
        """
        candidate_tokens = list(token_sequence)
        padding_offset = len(candidate_tokens)
        for _ in range(self.MAX_LENGTH_REPAIR_ATTEMPTS):
            prompt = tokenizer.decode(candidate_tokens)
            final_tokens = self._encode_without_special_tokens(tokenizer, prompt)
            token_delta = len(final_tokens) - target_token_len
            if token_delta == 0:
                return prompt, final_tokens
            if token_delta > 0:
                candidate_tokens = final_tokens[:target_token_len]
                continue

            padding_len = -token_delta
            padding_indices = (
                padding_offset + self._np.arange(padding_len)
            ) % len(allowed_tokens)
            candidate_tokens.extend(allowed_tokens[padding_indices].tolist())
            padding_offset += padding_len

        raise ValueError(
            "random prompt tokenization could not reach the requested "
            f"length {target_token_len} after "
            f"{self.MAX_LENGTH_REPAIR_ATTEMPTS} repair attempts"
        )

    @staticmethod
    def _encode_without_special_tokens(tokenizer: TokenizerLike, prompt: str) -> list[int]:
        """Encode prompt text without adding special tokens when supported."""
        try:
            return tokenizer.encode(prompt, add_special_tokens=False)
        except TypeError:
            return tokenizer.encode(prompt)


class SonnetDataset(BenchmarkDataset):
    """Port of the vLLM line-selection Sonnet workload."""
    DEFAULT_PREFIX_LEN = 200
    DEFAULT_INPUT_LEN = 550
    DEFAULT_OUTPUT_LEN = 150

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        path = self._require_path(self.dataset_path)
        with open(path, encoding="utf-8") as handle:
            self.data = [line for line in handle.readlines() if line.strip()]
        if not self.data:
            raise ValueError("sonnet dataset_path contains no non-empty lines")

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        *,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> DatasetBatch:
        del no_oversample
        if num_requests < 1:
            raise ValueError("num_requests must be positive")
        if prefix_len < 0 or input_len < 1 or output_len < 1:
            raise ValueError("sonnet lengths must be non-negative and positive where required")
        tokenized_lines = [
            self._tokenizer_sequence(tokenizer, line) for line in self.data
        ]
        average_len = sum(len(tokens) for tokens in tokenized_lines) / len(tokenized_lines)
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_offset = len(self._tokenizer_sequence(tokenizer, base_prompt))
        if input_len <= base_offset:
            raise ValueError(
                "sonnet input_len must be higher than the base prompt length"
            )
        num_input_lines = max(round((input_len - base_offset) / average_len), 1)
        num_prefix_lines = max(round((prefix_len - base_offset) / average_len), 0)
        prefix_lines = self.data[:num_prefix_lines]
        rng = random_module.Random(self.random_seed)
        requests = []
        attempts = 0
        request_index = 0
        while len(requests) < num_requests:
            extra_count = max(num_input_lines - num_prefix_lines, 1)
            extra_lines = [rng.choice(self.data) for _ in range(extra_count)]
            prompt = base_prompt
            for line in prefix_lines + extra_lines:
                prompt += line if line.endswith("\n") else f"{line}\n"
            prompt_len = len(self._tokenizer_sequence(tokenizer, prompt))
            if prompt_len <= input_len:
                requests.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                        request_id=f"{request_id_prefix}{request_index}",
                    )
                )
                request_index += 1
            attempts += 1
            if attempts > 64 * num_requests + 64:
                break
        if not requests:
            raise ValueError("sonnet tokenizer could not reach the requested input length")
        common_len = self._shared_token_prefix(
            tokenizer, prompts=[request.prompt for request in requests]
        ) if prefix_lines else 0
        return DatasetBatch(requests, shared_prefix_len=common_len)


def _conversation_parts(entry: dict[str, Any]) -> tuple[str, str]:
    """Read the first two conversation turns from a ShareGPT record."""
    conversations = entry.get("conversations", [])
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("sharegpt records must contain at least two turns")
    prompt = conversations[0].get("value")
    completion = conversations[1].get("value")
    if not isinstance(prompt, str) or not isinstance(completion, str) or not prompt:
        raise ValueError("sharegpt conversation turns must be non-empty strings")
    return prompt, completion


class ShareGptDataset(BenchmarkDataset):
    """Local adapter for ShareGPT-style JSON and JSONL conversation files."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        path = self._require_path(self.dataset_path)
        self.data = self._load_json_records(path)
        self.data = [
            entry for entry in self.data
            if "conversations" in entry and len(entry["conversations"]) >= 2
        ]
        if self.data and not self.disable_shuffle:
            random_module.Random(self.random_seed).shuffle(self.data)

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        *,
        output_len: int | None = None,
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> DatasetBatch:
        if output_len is not None and output_len < 1:
            raise ValueError("sharegpt output_len must be positive when set")
        rows = self._sample_rows(
            self.data,
            num_requests,
            self.random_seed,
            no_oversample=no_oversample,
            disable_shuffle=self.disable_shuffle,
        )
        requests = []
        for index, entry in enumerate(rows):
            prompt, completion = _conversation_parts(entry)
            prompt_len = len(self._tokenizer_sequence(tokenizer, prompt))
            native_output_len = len(tokenizer.encode(completion))
            resolved_output_len = output_len if output_len is not None else native_output_len
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=resolved_output_len,
                    request_id=f"{request_id_prefix}{index}",
                )
            )
        return DatasetBatch(requests)


class BurstGptDataset(BenchmarkDataset):
    """Local trace adapter for the GPT-4 subset of BurstGPT."""
    INPUT_KEYS = ("Request tokens", "Input tokens", "input_tokens", "input_length")
    OUTPUT_KEYS = ("Response tokens", "Output tokens", "output_tokens", "output_length")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        path = self._require_path(self.dataset_path)
        rows = self._load_dataset_rows(path)
        filtered = []
        for row in rows:
            if row.get("Model") != "GPT-4":
                continue
            input_len = self._numeric(row, self.INPUT_KEYS)
            output_len = self._numeric(row, self.OUTPUT_KEYS)
            if input_len < 0 or output_len < 1:
                continue
            filtered.append((input_len, output_len))
        self.data = filtered

    @staticmethod
    def _numeric(row: dict[str, str], keys: tuple[str, ...]) -> int | None:
        for key in keys:
            value = row.get(key)
            if value:
                try:
                    return int(value)
                except ValueError as exc:
                    raise ValueError(f"burstgpt {key!r} must contain integers") from exc
        return None

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        *,
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> DatasetBatch:
        rows = self._sample_rows(
            self.data,
            num_requests,
            self.random_seed,
            no_oversample=no_oversample,
            disable_shuffle=self.disable_shuffle,
        )
        requests = []
        for index, (input_len, output_len) in enumerate(rows):
            token_ids = [(index + offset) % tokenizer.vocab_size for offset in range(input_len)]
            requests.append(
                SampleRequest(
                    prompt=tokenizer.decode(token_ids),
                    prompt_len=input_len,
                    expected_output_len=output_len,
                    request_id=f"{request_id_prefix}{index}",
                )
            )
        return DatasetBatch(requests)


class HuggingFaceDataset(BenchmarkDataset):
    """Offline adapter for a locally exported HuggingFace records file."""

    PROMPT_KEYS = ("prompt", "input", "question", "query")
    OUTPUT_KEYS = ("completion", "response", "answer", "output")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        path = self._require_path(self.dataset_path)
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            self.data = self._load_json_records(path)
        elif suffix in {".jsonl", ".ndjson"}:
            self.data = self._load_jsonl_records(path)
        else:
            self.data = self._load_dataset_rows(path)

    @staticmethod
    def _load_jsonl_records(path: str) -> list[dict[str, Any]]:
        records = []
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on JSONL line {line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError("dataset_path JSONL must contain objects")
                records.append(value)
        return records

    @staticmethod
    def _pick(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        return next(
            (record[key] for key in keys if key in record and isinstance(record[key], str)),
            None,
        )

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        *,
        output_len: int | None = None,
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> DatasetBatch:
        if output_len is not None and output_len < 1:
            raise ValueError("hf output_len must be positive when set")
        rows = self._sample_rows(
            self.data,
            num_requests,
            self.random_seed,
            no_oversample=no_oversample,
            disable_shuffle=self.disable_shuffle,
        )
        requests = []
        for index, record in enumerate(rows):
            prompt = self._pick(record, self.PROMPT_KEYS)
            completion = self._pick(record, self.OUTPUT_KEYS)
            if not prompt:
                raise ValueError("hf records require a text prompt field")
            if output_len is None:
                if not completion:
                    raise ValueError("hf records require a text completion field")
                native_output_len = len(tokenizer.encode(completion))
                resolved_output_len = native_output_len
            else:
                resolved_output_len = output_len
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=len(tokenizer.encode(prompt)),
                    expected_output_len=resolved_output_len,
                    request_id=f"{request_id_prefix}{index}",
                )
            )
        return DatasetBatch(requests)


class Gsm8kDataset(BenchmarkDataset):
    """Copy GSM8K problems to target length with round-isolated prompts."""

    DEFAULT_INPUT_LEN = 2048
    DEFAULT_OUTPUT_LEN = 256
    DEFAULT_ROUND_PREFIX_LEN = 32

    PROMPT_KEYS = ("question", "question_text", "problem")
    ANSWER_KEYS = ("answer", "output", "response", "completion")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        path = self._require_path(self.dataset_path)
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            self.data = self._load_json_records(path)
        elif suffix in {".jsonl", ".ndjson"}:
            self.data = self._load_jsonl_records(path)
        else:
            self.data = self._load_dataset_rows(path)
        if not self.data:
            raise ValueError("gsm8k dataset_path contains no records")

    @staticmethod
    def _load_jsonl_records(path: str) -> list[dict[str, Any]]:
        """Read a small local JSONL file without optional dependencies."""
        records = []
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on JSONL line {line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError("dataset_path JSONL must contain objects")
                records.append(value)
        return records

    @staticmethod
    def _pick(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        return next(
            (record[key] for key in keys if key in record and isinstance(record[key], str)),
            None,
        )

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        *,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int | None = None,
        prefix_len: int = DEFAULT_ROUND_PREFIX_LEN,
        no_oversample: bool = False,
        run_id: str = "",
        **kwargs: Any,
    ) -> DatasetBatch:
        if num_requests < 1:
            raise ValueError("num_requests must be positive")
        if input_len < 1 or prefix_len < 0:
            raise ValueError("gsm8k input_len must be positive and prefix_len non-negative")
        if output_len is not None and output_len < 1:
            raise ValueError("gsm8k output_len must be positive when set")
        if prefix_len >= input_len:
            raise ValueError("gsm8k round prefix must be shorter than input_len")

        round_text = f"【GSM8K-ROUND-{run_id or uuid.uuid4().hex[:8]}】"
        round_ids = self._tokenizer_sequence(tokenizer, round_text)
        if prefix_len and len(round_ids) > prefix_len:
            raise ValueError("gsm8k round prefix is too short for the round marker")
        prefix_ids = (round_ids * ((prefix_len // len(round_ids)) + 1))[:prefix_len]
        requests = []
        rows = self._sample_rows(
            self.data,
            num_requests,
            self.random_seed,
            no_oversample=no_oversample,
            disable_shuffle=self.disable_shuffle,
        )
        body_target = input_len - len(prefix_ids)
        for index, record in enumerate(rows):
            question = self._pick(record, self.PROMPT_KEYS)
            if not question:
                raise ValueError("gsm8k records require a question field")
            answer = self._pick(record, self.ANSWER_KEYS)
            resolved_output_len = (
                output_len if output_len is not None else len(tokenizer.encode(answer or "\n"))
            )
            source_text = f"Question: {question}\nAnswer: {answer or ''}\n"
            source_ids = self._tokenizer_sequence(tokenizer, source_text)
            repeats = max(1, math.ceil(body_target / len(source_ids)))
            body_ids = (source_ids * repeats)[:body_target]
            prompt_ids = prefix_ids + body_ids
            prompt = tokenizer.decode(prompt_ids)
            actual_ids = self._tokenizer_sequence(tokenizer, prompt)
            # The text is sent to the backend, so only a decodable prompt is accepted.
            if len(actual_ids) != len(prompt_ids):
                prompt_ids = actual_ids
                prompt_len = len(actual_ids)
            else:
                prompt_len = len(prompt_ids)
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=resolved_output_len,
                    request_id=f"{request_id_prefix}{index}",
                )
            )
        return DatasetBatch(requests, shared_prefix_len=len(prefix_ids))


def create_dataset(
    name: str,
    *,
    random_seed: int | None = None,
    dataset_path: str | None = None,
    disable_shuffle: bool = False,
) -> BenchmarkDataset:
    """Create a named dataset without coupling generators to argparse."""
    path_args = {
        "random_seed": random_seed,
        "dataset_path": dataset_path,
        "disable_shuffle": disable_shuffle,
    }
    if name == "text":
        return TextDataset(random_seed=random_seed)
    if name == "random":
        return RandomDataset(random_seed=random_seed)
    if name == "sonnet":
        return SonnetDataset(**path_args)
    if name == "sharegpt":
        return ShareGptDataset(**path_args)
    if name == "burstgpt":
        return BurstGptDataset(**path_args)
    if name == "hf":
        return HuggingFaceDataset(**path_args)
    if name == "gsm8k":
        return Gsm8kDataset(**path_args)
    raise ValueError(
        f"unknown benchmark dataset {name!r}; expected "
        "'text', 'random', 'sonnet', 'sharegpt', 'burstgpt', 'hf' or 'gsm8k'"
    )
