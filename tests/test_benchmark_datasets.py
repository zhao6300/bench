from __future__ import annotations

import pytest

from benchmark.benchmark_datasets import (
    RandomDataset,
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
            ord(char) - 0xE000 if 0xE000 <= ord(char) < 0xE000 + self.vocab_size else ord(char)
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.25, (0.25, 0.25)),
        ("0.1", (0.1, 0.1)),
        ("0.1, 0.2", (0.1, 0.2)),
        ((0.3, 0.4), (0.3, 0.4)),
    ],
)
def test_parse_range_ratio_accepts_single_and_pair_values(value: object, expected: tuple[float, float]) -> None:
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
        dataset.sample(tokenizer, num_requests=1, input_len=1, output_len=1, prefix_ratio=1.1)


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
    assert tokenizer.encode(first_batch.prompts[0])[:4] == tokenizer.encode(second_batch.prompts[0])[:4]
    assert [request.request_id for request in first_batch.requests] == ["first-0", "first-1"]


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
        len(tokenizer.encode(request.prompt, add_special_tokens=False)) == request.prompt_len
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
