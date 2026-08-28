"""Unit tests for the optional OpenAI-compatible aiohttp transport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from benchmark import api_transport
from benchmark import benchmark as benchmark_module
from benchmark.api_transport import finalize_stream_result
from benchmark.benchmark import BenchmarkConfigError, _build_case_args, build_parser


class _FakeAsyncContent:
    """Expose aiohttp's asynchronous response-content iteration contract."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeAsyncResponse:
    """Provide the response operations used by the async transport."""

    def __init__(self, chunks: list[bytes], error: Exception | None = None) -> None:
        self.content = _FakeAsyncContent(chunks)
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error


class _FakeAsyncRequestContext:
    """Implement the async context manager returned by aiohttp.ClientSession.post."""

    def __init__(self, response: _FakeAsyncResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeAsyncResponse:
        return self._response

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeAsyncSession:
    """Capture async request arguments and return a predefined stream response."""

    def __init__(self, response: _FakeAsyncResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _FakeAsyncRequestContext:
        self.calls.append({"url": url, **kwargs})
        return _FakeAsyncRequestContext(self._response)


class _FallbackTokenizer:
    """Count a known complete response as three local output tokens."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.calls.append((text, add_special_tokens))
        return [0, 1, 2] if text == "fallback" else [0, 1]


def _run(coroutine: Any) -> Any:
    """Run one async transport request without pytest async plugins."""
    return asyncio.run(coroutine)


def _content_event(text: str) -> bytes:
    return json.dumps({"choices": [{"delta": {"content": text}}]}).encode()


def test_aiohttp_transport_sends_shared_payload_and_uses_server_usage(monkeypatch) -> None:
    response = _FakeAsyncResponse([
        b"data: " + _content_event("first") + b"\n\n",
        b"data: " + _content_event("next") + b"\n\n",
        b'data: {"usage": {"completion_tokens": 3, "prompt_tokens": 4}}\n\n',
        b"data: [DONE]",
    ])
    session = _FakeAsyncSession(response)
    tokenizer = _FallbackTokenizer()
    ticks = iter([0.0, 1.0, 1.4, 2.0])
    monkeypatch.setattr(api_transport.time, "perf_counter", lambda: next(ticks))

    result = _run(api_transport.send_aiohttp_chat_request(
        7,
        "prompt",
        "http://localhost:8000/v1/chat/completions",
        {"Authorization": "Bearer test-key"},
        "test-model",
        9,
        session,
        tokenizer,
        False,
    ))

    assert session.calls == [{
        "url": "http://localhost:8000/v1/chat/completions",
        "json": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "prompt"}],
            "max_tokens": 9,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": False,
        },
        "headers": {"Authorization": "Bearer test-key"},
    }]
    assert result["error"] is None
    assert result["first_token_text"] == "first"
    assert result["prompt_tokens"] == 4
    assert result["output_tokens"] is None
    assert tokenizer.calls == []

    finalize_stream_result(result, tokenizer)

    assert result["output_tokens"] == 3
    assert result["output_token_source"] == "server_usage"
    assert tokenizer.calls == [("next", False)]
    assert result["estimated_itl_samples"] == pytest.approx([0.2, 0.2])
    assert result["tpot"] == pytest.approx(0.2)
    assert not any(key.startswith("_stream_") for key in result)


def test_aiohttp_transport_tokenizes_complete_output_once_after_stream() -> None:
    session = _FakeAsyncSession(_FakeAsyncResponse([
        b"data: " + _content_event("fall") + b"\n\n",
        b"data: " + _content_event("back") + b"\n\n",
        b"data: [DONE]",
    ]))
    tokenizer = _FallbackTokenizer()

    result = _run(api_transport.send_aiohttp_chat_request(
        1, "prompt", "http://localhost/chat/completions", {}, "model", 3,
        session, tokenizer, True,
    ))

    assert result["error"] is None
    assert result["output_tokens"] is None
    assert tokenizer.calls == []

    finalize_stream_result(result, tokenizer)

    assert result["output_tokens"] == 3
    assert result["output_token_source"] == "local_tokenizer"
    assert tokenizer.calls == [("fallback", False), ("back", False)]


def test_aiohttp_transport_returns_failed_record_without_content() -> None:
    session = _FakeAsyncSession(_FakeAsyncResponse([
        b'data: {"usage": {"prompt_tokens": 4}}\n\n',
        b"data: [DONE]",
    ]))

    result = _run(api_transport.send_aiohttp_chat_request(
        1, "prompt", "http://localhost/chat/completions", {}, "model", 3, session,
    ))

    assert result["error"] == "未收到首个 token"
    assert result["ttft"] is None
    assert result["prompt_tokens"] == 4
    assert result["output_tokens"] is None


def test_aiohttp_transport_returns_failed_record_for_http_error() -> None:
    session = _FakeAsyncSession(_FakeAsyncResponse([], RuntimeError("service unavailable")))

    result = _run(api_transport.send_aiohttp_chat_request(
        1, "prompt", "http://localhost/chat/completions", {}, "model", 3, session,
    ))

    assert result["error"] == "service unavailable"
    assert result["ttft"] is None
    assert result["output_token_source"] == "unavailable"


def test_api_transport_parser_and_suite_config_accept_aiohttp() -> None:
    assert build_parser().parse_args([]).api_transport == "requests"

    args, _ = _build_case_args(
        {"mode": "api", "api_transport": "aiohttp"},
        {"name": "async-transport"},
        {},
    )

    assert args.api_transport == "aiohttp"


def test_suite_config_rejects_invalid_api_transport() -> None:
    with pytest.raises(BenchmarkConfigError, match="api_transport"):
        _build_case_args(
            {"mode": "api", "api_transport": "unsupported"},
            {"name": "invalid-transport"},
            {},
        )


def test_suite_config_rejects_remote_plaintext_bearer_key() -> None:
    with pytest.raises(BenchmarkConfigError, match="refusing to send api_key"):
        _build_case_args(
            {
                "mode": "api",
                "api_base": "http://api.example.test/v1",
                "api_key": "test-key",
            },
            {"name": "insecure-key"},
            {},
        )


def test_api_round_dispatches_aiohttp_records(monkeypatch) -> None:
    observed: dict[str, Any] = {}
    progress_snapshots: list[dict[str, Any]] = []

    class _Reporter:
        def round_started(self, _total: int, _concurrency: int) -> None:
            pass

        def request_finished(self, snapshot: dict[str, Any]) -> None:
            progress_snapshots.append(snapshot)

        def round_finished(self) -> None:
            pass

    def fake_aiohttp_runner(
        prompts: list[str],
        url: str,
        headers: dict[str, str],
        model: str,
        max_tokens: list[int],
        concurrency: int,
        tokenizer: Any,
        ignore_eos: bool,
        on_complete: Any,
    ) -> None:
        observed.update({
            "prompts": prompts,
            "url": url,
            "headers": headers,
            "model": model,
            "max_tokens": max_tokens,
            "concurrency": concurrency,
            "ignore_eos": ignore_eos,
        })
        for req_id in range(len(prompts)):
            on_complete({
                "req_id": req_id,
                "error": None,
                "ttft": 0.01,
                "tpot": 0.01,
                "request_start_timestamp": float(req_id),
                "first_token_timestamp": float(req_id) + 0.01,
                "last_token_timestamp": float(req_id) + 0.02,
                "first_token_text": "ok",
                "prompt_tokens": 1,
                "output_tokens": 2,
                "output_token_source": "server_usage",
                "total_time": 0.02,
                "estimated_itl_samples": [0.01],
            })

    def fake_finalize(records: list[dict[str, Any]], tokenizer: Any) -> None:
        observed["finalized_req_ids"] = [record["req_id"] for record in records]

    monkeypatch.setattr(benchmark_module, "run_aiohttp_chat_requests", fake_aiohttp_runner)
    monkeypatch.setattr(benchmark_module, "finalize_stream_results", fake_finalize)
    monkeypatch.setattr(benchmark_module, "query_gpu_metrics", lambda *_args, **_kwargs: None)

    metrics = benchmark_module.run_api_benchmark_round(
        ["first", "second"], [1, 1], "http://localhost/v1/chat/completions",
        {"Content-Type": "application/json"}, "model", [2, 2], 2,
        api_transport="aiohttp", progress_reporter=_Reporter(),
    )

    assert observed == {
        "prompts": ["first", "second"],
        "url": "http://localhost/v1/chat/completions",
        "headers": {"Content-Type": "application/json"},
        "model": "model",
        "max_tokens": [2, 2],
        "concurrency": 2,
        "ignore_eos": True,
        "finalized_req_ids": [0, 1],
    }
    assert metrics["successful"] == 2
    assert [
        (snapshot["completed"], snapshot["total"], snapshot["succeeded"],
         snapshot["failed"], snapshot["last_ttft"], snapshot["request_id"],
         snapshot["error"])
        for snapshot in progress_snapshots
    ] == [
        (1, 2, 1, 0, 0.01, 0, None),
        (2, 2, 2, 0, 0.01, 1, None),
    ]
    assert all(snapshot["elapsed_seconds"] >= 0 for snapshot in progress_snapshots)


def test_progress_parser_defaults_to_auto_and_is_not_suite_parameter() -> None:
    """Keep progress rendering as a CLI-only concern outside suite JSON schema."""
    parser = build_parser()

    assert parser.parse_args([]).progress == "auto"
    assert parser.parse_args(["--progress", "plain"]).progress == "plain"
    assert parser.parse_args(["--progress", "rich"]).progress == "rich"
    assert parser.parse_args(["--progress", "off"]).progress == "off"

    with pytest.raises(BenchmarkConfigError, match="unknown benchmark parameters: progress"):
        _build_case_args(
            {"mode": "api", "progress": "off"},
            {"name": "progress-is-cli-only"},
            {},
        )
