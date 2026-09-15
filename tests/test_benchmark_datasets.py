from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark.benchmark import build_request_batch
from benchmark.benchmark_datasets import (
    BenchmarkDataset,
    BfclDataset,
    BlazeditDataset,
    BurstGptDataset,
    Gsm8kDataset,
    HuggingFaceDataset,
    HumanEvalDataset,
    InstructCoderDataset,
    RandomDataset,
    ShareGptDataset,
    SonnetDataset,
    TextDataset,
    create_dataset,
    parse_range_ratio,
)


class FakeTokenizer:
    """Small reversible tokenizer for dataset tests without model downloads."""

    vocab_size = 64
    all_special_ids = (0, 1)

    def encode(self, text: str, **kwargs: object) -> list[int]:
        del kwargs
        return [
            ord(char) - 0xE000
            if 0xE000 <= ord(char) < 0xE000 + self.vocab_size
            else ord(char)
            for char in text
        ]

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        del kwargs
        return "".join(chr(0xE000 + token_id) for token_id in token_ids)

    def num_special_tokens_to_add(self, *args: object, **kwargs: object) -> int:
        del args, kwargs
        return 0


class TrailingTokenDroppingTokenizer(FakeTokenizer):
    """Tokenizer that loses the final token whenever decoded text is encoded."""

    def encode(self, text: str, **kwargs: object) -> list[int]:
        return super().encode(text, **kwargs)[:-1]


class EmptyTokenizer(FakeTokenizer):
    """Tokenizer that cannot encode any text, for repair failure coverage."""

    def encode(self, text: str, **kwargs: object) -> list[int]:
        del text, kwargs
        return []


class LineTokenizer(FakeTokenizer):
    """Tokenizer that emits one token per text line for Sonnet length tests."""

    def encode(self, text: str, **kwargs: object) -> list[int]:
        del kwargs
        return list(range(len(text.splitlines())))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.25, (0.25, 0.25)),
        ("0.1", (0.1, 0.1)),
        ("0.1, 0.2", (0.1, 0.2)),
        ((0.3, 0.4), (0.3, 0.4)),
    ],
)
def test_parse_range_ratio_accepts_single_and_pair_values(
    value: object, expected: tuple[float, float]
) -> None:
    assert parse_range_ratio(value) == expected


@pytest.mark.parametrize("value", ["0.1,0.2,0.3", "-0.1", "1.1", (0.2, 1.5)])
def test_parse_range_ratio_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        parse_range_ratio(value)


def test_text_dataset_reports_shared_and_unique_prompt_lengths() -> None:
    batch = TextDataset().sample(
        FakeTokenizer(),
        num_requests=2,
        request_id_prefix="text-",
        input_len=24,
        output_len=5,
        share_prefix=True,
        prefix_ratio=0.5,
        run_id="stable-run",
    )

    assert batch.prompt_lens == [24, 24]
    assert batch.output_lens == [5, 5]
    assert batch.shared_prefix_len == 12
    assert batch.unique_prefix_len == 12
    assert [request.request_id for request in batch.requests] == ["text-0", "text-1"]
    assert batch.prompts[0][:12] == batch.prompts[1][:12]
    assert batch.prompts[0][12:] != batch.prompts[1][12:]


def test_text_dataset_validates_lengths_and_prefix_ratio() -> None:
    dataset = TextDataset()
    tokenizer = FakeTokenizer()

    with pytest.raises(ValueError, match="input_len"):
        dataset.sample(tokenizer, num_requests=1, input_len=0, output_len=1)
    with pytest.raises(ValueError, match="output_len"):
        dataset.sample(tokenizer, num_requests=1, input_len=1, output_len=0)
    with pytest.raises(ValueError, match="prefix_ratio"):
        dataset.sample(
            tokenizer, num_requests=1, input_len=1, output_len=1, prefix_ratio=1.1
        )


def test_random_dataset_preserves_cached_shared_prefix_and_lengths() -> None:
    tokenizer = FakeTokenizer()
    dataset = RandomDataset(random_seed=7)

    first_batch = dataset.sample(
        tokenizer,
        num_requests=2,
        request_id_prefix="first-",
        prefix_len=4,
        input_len=8,
        output_len=3,
        range_ratio=0,
    )
    second_batch = dataset.sample(
        tokenizer,
        num_requests=1,
        request_id_prefix="second-",
        prefix_len=4,
        input_len=8,
        output_len=3,
        range_ratio=0,
    )

    assert first_batch.prompt_lens == [12, 12]
    assert first_batch.output_lens == [3, 3]
    assert first_batch.shared_prefix_len == 4
    assert second_batch.shared_prefix_len == 4
    assert (
        tokenizer.encode(first_batch.prompts[0])[:4]
        == tokenizer.encode(second_batch.prompts[0])[:4]
    )
    assert [request.request_id for request in first_batch.requests] == [
        "first-0",
        "first-1",
    ]


def test_random_dataset_repairs_non_reversible_tokenizer_lengths() -> None:
    tokenizer = TrailingTokenDroppingTokenizer()
    batch = RandomDataset(random_seed=7).sample(
        tokenizer,
        num_requests=2,
        prefix_len=2,
        input_len=6,
        output_len=3,
        range_ratio=0,
    )

    assert batch.prompt_lens == [8, 8]
    assert batch.shared_prefix_len == 2
    assert all(
        len(tokenizer.encode(request.prompt, add_special_tokens=False))
        == request.prompt_len
        for request in batch.requests
    )


def test_random_dataset_rejects_unrepairable_tokenizer_length() -> None:
    with pytest.raises(ValueError, match="could not reach the requested length 1"):
        RandomDataset(random_seed=7).sample(
            EmptyTokenizer(),
            num_requests=1,
            input_len=1,
            output_len=1,
            range_ratio=0,
        )


def test_create_dataset_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown benchmark dataset"):
        create_dataset("unsupported")


def test_dataset_output_len_applies_to_new_local_datasets(tmp_path) -> None:
    path = tmp_path / "humaneval.jsonl"
    path.write_text(
        '{"prompt":"write code","canonical_solution":"solution"}\n',
        encoding="utf-8",
    )
    args = SimpleNamespace(
        dataset="humaneval",
        dataset_path=str(path),
        context_len=128,
        max_tokens=128,
        random_seed=None,
        seed=1,
        no_oversample=False,
        disable_shuffle=True,
        dataset_output_len=3,
        share_prefix=False,
        prefix_ratio=1.0,
    )

    batch = build_request_batch(args, FakeTokenizer(), num_requests=1)

    assert batch.output_lens == [3]


def test_dataset_specific_output_len_stays_backward_compatible(tmp_path) -> None:
    path = tmp_path / "humaneval.jsonl"
    path.write_text(
        '{"prompt":"write code","canonical_solution":"solution"}\n',
        encoding="utf-8",
    )
    args = SimpleNamespace(
        dataset="humaneval",
        dataset_path=str(path),
        context_len=128,
        max_tokens=128,
        random_seed=None,
        seed=1,
        no_oversample=False,
        disable_shuffle=True,
        dataset_output_len=3,
        humaneval_output_len=5,
        share_prefix=False,
        prefix_ratio=1.0,
    )

    batch = build_request_batch(args, FakeTokenizer(), num_requests=1)

    assert batch.output_lens == [5]


def test_sonnet_dataset_port_uses_prefix_and_native_lengths(tmp_path) -> None:
    path = tmp_path / "sonnet.txt"
    path.write_text("one\n", encoding="utf-8")
    dataset = SonnetDataset(dataset_path=str(path), random_seed=9)

    batch = dataset.sample(
        LineTokenizer(),
        num_requests=2,
        request_id_prefix="sonnet-",
        prefix_len=60,
        input_len=70,
        output_len=4,
    )

    assert batch.prompt_lens == [12, 12]
    assert batch.output_lens == [4, 4]
    assert batch.requests[0].request_id == "sonnet-0"
    assert batch.prompts[0] == batch.prompts[1]
    assert batch.prompts[0].startswith("Pick as many lines as you can")
    assert batch.shared_prefix_len == 12


def test_create_dataset_builds_local_ported_datasets(tmp_path) -> None:
    sonnet_path = tmp_path / "sonnet.txt"
    sonnet_path.write_text("line\n", encoding="utf-8")
    sharegpt_path = tmp_path / "sharegpt.json"
    sharegpt_path.write_text(
        '[{"conversations":[{"value":"hello"},{"value":"answer"}]}]',
        encoding="utf-8",
    )
    burst_path = tmp_path / "burstgpt.csv"
    burst_path.write_text(
        "Model,Request tokens,Response tokens\nGPT-4,7,2\nGPT-5,5,1\n",
        encoding="utf-8",
    )
    hf_path = tmp_path / "records.jsonl"
    hf_path.write_text('{"question":"question","answer":"answers"}\n', encoding="utf-8")
    local_port_path = tmp_path / "records.ndjson"
    local_port_path.write_text('{"prompt":"code"}\n', encoding="utf-8")

    assert isinstance(
        create_dataset("sonnet", dataset_path=str(sonnet_path)), SonnetDataset
    )
    assert isinstance(
        create_dataset("sharegpt", dataset_path=str(sharegpt_path)), ShareGptDataset
    )
    assert isinstance(
        create_dataset("burstgpt", dataset_path=str(burst_path)), BurstGptDataset
    )
    assert isinstance(
        create_dataset("hf", dataset_path=str(hf_path)), HuggingFaceDataset
    )
    assert isinstance(
        create_dataset("humaneval", dataset_path=str(local_port_path)), HumanEvalDataset
    )
    assert isinstance(
        create_dataset("instructcoder", dataset_path=str(local_port_path)),
        InstructCoderDataset,
    )
    assert isinstance(
        create_dataset("blazedit", dataset_path=str(local_port_path)), BlazeditDataset
    )
    assert isinstance(
        create_dataset("bfcl", dataset_path=str(local_port_path)), BfclDataset
    )


def test_humaneval_dataset_expands_prompt_to_requested_input_length(tmp_path) -> None:
    """Humaneval adapter can pad prompts to fixed input token length."""
    path = tmp_path / "humaneval.jsonl"
    path.write_text(
        '{"prompt":"write code","canonical_solution":"solution"}\n',
        encoding="utf-8",
    )

    batch = HumanEvalDataset(dataset_path=str(path), disable_shuffle=True).sample(
        FakeTokenizer(),
        num_requests=1,
        input_len=12,
        output_len=8,
    )

    assert batch.prompt_lens == [12]
    assert batch.output_lens == [8]


def test_sharegpt_dataset_uses_completion_length_in_hierarchy(tmp_path) -> None:
    path = tmp_path / "sharegpt.json"
    path.write_text(
        '[{"conversations":[{"value":"hello"},{"value":"answers"}]}]',
        encoding="utf-8",
    )

    batch = ShareGptDataset(dataset_path=str(path), disable_shuffle=True).sample(
        FakeTokenizer(),
        num_requests=1,
        request_id_prefix="sharegpt-",
    )

    assert batch.prompt_lens == [5]
    assert batch.output_lens == [7]
    assert batch.requests[0].request_id == "sharegpt-0"


def test_humaneval_dataset_uses_prompt_and_native_output_length(tmp_path) -> None:
    path = tmp_path / "humaneval.jsonl"
    path.write_text(
        '{"prompt":"solve this","canonical_solution":"solution"}\n',
        encoding="utf-8",
    )

    batch = HumanEvalDataset(dataset_path=str(path), disable_shuffle=True).sample(
        FakeTokenizer(),
        num_requests=1,
        request_id_prefix="humaneval-",
    )

    assert batch.prompt_lens == [10]
    assert batch.output_lens == [8]
    assert batch.requests[0].request_id == "humaneval-0"
    assert batch.prompts[0][0] == "solve this"


def test_instructcoder_dataset_formats_editing_prompt(tmp_path) -> None:
    path = tmp_path / "instructcoder.jsonl"
    path.write_text(
        '{"input":"original code","instruction":"add tests"}\n',
        encoding="utf-8",
    )

    batch = InstructCoderDataset(dataset_path=str(path), disable_shuffle=True).sample(
        FakeTokenizer(),
        num_requests=1,
        output_len=7,
    )

    assert batch.prompt_lens == [78]
    assert batch.output_lens == [7]
    assert "original code" in batch.prompts[0][0]
    assert "add tests" in batch.prompts[0][0]


def test_blazedit_dataset_filters_edit_distance(tmp_path) -> None:
    path = tmp_path / "blazedit.jsonl"
    path.write_text(
        '{"code":"file body","change_request":"rename old","norm_distance":0.5}\n',
        encoding="utf-8",
    )

    batch = BlazeditDataset(dataset_path=str(path), disable_shuffle=True).sample(
        FakeTokenizer(),
        num_requests=1,
        output_len=11,
        min_distance=0.4,
        max_distance=0.6,
    )

    assert batch.output_lens == [11]
    assert "file body" in batch.prompts[0][0]
    assert "rename old" in batch.prompts[0][0]


def test_bfcl_dataset_translates_function_schema(tmp_path) -> None:
    path = tmp_path / "bfcl.jsonl"
    path.write_text(
        '{"question":[[{"role":"user","content":"Find area"}]],'
        '"function":[{"name":"circle_area","parameters":{"type":"dict"}}]}',
        encoding="utf-8",
    )

    batch = BfclDataset(dataset_path=str(path), disable_shuffle=True).sample(
        FakeTokenizer(),
        num_requests=1,
        output_len=9,
    )

    assert batch.output_lens == [9]
    assert "Find area" in batch.prompts[0][0]
    assert '"type":"object"' in batch.prompts[0][0]


def test_burstgpt_dataset_filters_rows_and_synthesizes_tokens(tmp_path) -> None:
    path = tmp_path / "burstgpt.csv"
    path.write_text(
        "Model,Request tokens,Response tokens\nGPT-4,6,3\nGPT-4,0,1\n",
        encoding="utf-8",
    )

    batch = BurstGptDataset(dataset_path=str(path), random_seed=3).sample(
        FakeTokenizer(),
        num_requests=2,
        request_id_prefix="burstgpt-",
    )

    assert len(batch.requests) == 2
    assert sorted(batch.output_lens) == [1, 3]
    assert all(request.request_id.startswith("burstgpt-") for request in batch.requests)


def test_hf_dataset_reads_offline_records_and_output_override(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"prompt":"hello","completion":"answers"}\n'
        '{"question":"world","answer":"answers"}\n',
        encoding="utf-8",
    )
    dataset = HuggingFaceDataset(dataset_path=str(path), disable_shuffle=True)

    batch = dataset.sample(
        FakeTokenizer(),
        num_requests=2,
        request_id_prefix="hf-",
        output_len=1,
    )

    assert batch.prompt_lens == [5, 5]
    assert batch.output_lens == [1, 1]


def test_sharegpt_dataset_supports_original_order_and_no_oversample(tmp_path) -> None:
    path = tmp_path / "sharegpt.json"
    path.write_text(
        "["
        + ",".join(
            '{"conversations":[{"value":"prompt"},{"value":"answer"}]}'
            for _ in range(3)
        )
        + "]",
        encoding="utf-8",
    )
    dataset = ShareGptDataset(dataset_path=str(path), disable_shuffle=True)

    batch = dataset.sample(FakeTokenizer(), num_requests=5, no_oversample=True)

    assert len(batch.requests) == 3


def test_gsm8k_dataset_copies_records_and_isolates_rounds(tmp_path) -> None:
    path = tmp_path / "gsm8k.jsonl"
    path.write_text(
        '{"question":"hello math","answer":"2"}\n'
        '{"question":"second math","answer":"3"}\n',
        encoding="utf-8",
    )
    dataset = Gsm8kDataset(dataset_path=str(path), random_seed=17)

    first = dataset.sample(
        FakeTokenizer(),
        num_requests=1,
        input_len=64,
        output_len=128,
        prefix_len=32,
        run_id="round-one",
    )
    second = dataset.sample(
        FakeTokenizer(),
        num_requests=1,
        input_len=64,
        output_len=7,
        prefix_len=32,
        run_id="round-two",
    )
    repeated = dataset.sample(
        FakeTokenizer(),
        num_requests=1,
        input_len=64,
        output_len=128,
        prefix_len=32,
        run_id="round-one",
    )

    assert first.prompt_lens == [64]
    assert first.output_lens == [128]
    assert second.output_lens == [7]
    assert first.prompts != second.prompts
    assert first.prompts == repeated.prompts
    assert first.shared_prefix_len == 32


def test_gsm8k_dataset_supports_shared_prefix_ratio(tmp_path) -> None:
    path = tmp_path / "gsm8k.jsonl"
    path.write_text(
        '{"question":"hello math","answer":"2"}\n'
        '{"question":"second math","answer":"3"}\n',
        encoding="utf-8",
    )
    dataset = Gsm8kDataset(dataset_path=str(path), random_seed=17)

    batch = dataset.sample(
        FakeTokenizer(),
        num_requests=2,
        input_len=100,
        output_len=8,
        prefix_len=32,
        shared_prefix_ratio=0.7,
        run_id="same-round",
    )

    assert batch.prompt_lens == [100, 100]
    assert batch.shared_prefix_len == 70
    assert batch.prompts[0][:70] == batch.prompts[1][:70]
    assert batch.prompts[0][70:] != batch.prompts[1][70:]


def test_gsm8k_dataset_maps_cli_and_run_id_into_batch(tmp_path) -> None:
    path = tmp_path / "gsm8k.jsonl"
    path.write_text(
        '{"question":"hello math","answer":"2"}\n',
        encoding="utf-8",
    )
    args = SimpleNamespace(
        dataset="gsm8k",
        context_len=None,
        max_tokens=16,
        random_seed=11,
        seed=11,
        dataset_path=str(path),
        disable_shuffle=False,
        gsm8k_input_len=64,
        gsm8k_output_len=16,
        gsm8k_round_prefix_len=32,
        gsm8k_shared_prefix_ratio=0.7,
        no_oversample=True,
    )

    first = build_request_batch(
        args,
        FakeTokenizer(),
        num_requests=1,
        run_id="round-one",
    )
    second = build_request_batch(
        args,
        FakeTokenizer(),
        num_requests=1,
        run_id="round-two",
    )

    assert first.prompt_lens == [64]
    assert first.output_lens == [16]
    assert first.prompts != second.prompts
    assert first.shared_prefix_len == 44


def test_gsm8k_dataset_seed_selects_contiguous_repeatable_offset(tmp_path) -> None:
    """GSM8K sampling selects a seed-derived offset and contiguous records."""
    path = tmp_path / "gsm8k.jsonl"
    path.write_text(
        "\n".join(
            f'{{"question":"q{index}","answer":"{index}"}}' for index in range(12)
        ),
        encoding="utf-8",
    )
    rows = Gsm8kDataset(dataset_path=str(path), random_seed=17).data
    selected = BenchmarkDataset._sample_contiguous_rows(rows, num_requests=5, seed=17)

    assert [row["question"] for row in selected] == ["q8", "q9", "q10", "q11", "q0"]
