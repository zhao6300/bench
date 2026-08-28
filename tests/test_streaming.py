from __future__ import annotations

import asyncio
import json

from benchmark import benchmark as benchmark_module
from benchmark.api_transport import finalize_stream_result
from benchmark.streaming import aiter_sse_data, iter_sse_data


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int | None = None):
        assert chunk_size is None
        yield from self._chunks


def test_iter_sse_data_reassembles_split_utf8_and_ignores_comments() -> None:
    payload = json.dumps(
        {"choices": [{"delta": {"content": "你好"}}]},
        ensure_ascii=False,
    ).encode("utf-8")
    split_at = payload.index("你".encode("utf-8")) + 1
    chunks = [
        b": keep-alive\n\n",
        b"data: " + payload[:split_at],
        payload[split_at:] + b"\n\n",
        b"data: [DONE]",
    ]

    assert list(iter_sse_data(chunks)) == [payload.decode("utf-8"), "[DONE]"]


def test_iter_sse_data_waits_for_a_complete_json_event() -> None:
    chunks = [b'data: {"choices":', b' [{"delta": {"content": "ok"}}]}']

    assert list(iter_sse_data(chunks)) == ['{"choices": [{"delta": {"content": "ok"}}]}']


def test_aiter_sse_data_matches_sync_utf8_and_comment_handling() -> None:
    payload = json.dumps(
        {"choices": [{"delta": {"content": "你好"}}]},
        ensure_ascii=False,
    ).encode("utf-8")
    split_at = payload.index("你".encode("utf-8")) + 1

    async def chunks():
        yield b": keep-alive\r\n\r\n"
        yield b"data: " + payload[:split_at]
        yield payload[split_at:] + b"\r"
        yield b"\n\r\n"
        yield b"data: [DONE]"

    async def collect() -> list[str]:
        return [data async for data in aiter_sse_data(chunks())]

    assert asyncio.run(collect()) == [payload.decode("utf-8"), "[DONE]"]


def test_send_single_api_request_uses_chunked_sse_stream(monkeypatch) -> None:
    content_event = json.dumps(
        {"choices": [{"delta": {"content": "你好"}}]},
        ensure_ascii=False,
    ).encode("utf-8")
    usage_event = json.dumps(
        {"usage": {"completion_tokens": 2, "prompt_tokens": 3}}
    ).encode("utf-8")
    response = _FakeResponse(
        [
            b"data: " + content_event[:30],
            content_event[30:] + b"\n\n",
            b"data: " + usage_event + b"\n\n",
            b"data: [DONE]",
        ]
    )
    monkeypatch.setattr(benchmark_module.requests, "post", lambda *_args, **_kwargs: response)

    result = benchmark_module.send_single_api_request(
        req_id=1,
        prompt="prompt",
        url="http://localhost:8000/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        model="test-model",
        max_tokens=2,
    )

    assert result["error"] is None
    assert result["first_token_text"] == "你好"
    assert result["prompt_tokens"] == 3
    assert result["output_tokens"] is None

    finalize_stream_result(result, None)

    assert result["output_tokens"] == 2
    assert result["output_token_source"] == "server_usage"
