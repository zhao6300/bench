"""Unit tests for the optional OpenAI-compatible aiohttp transport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from itertools import pairwise
from typing import Any

import pytest

from benchmark import api_transport
from benchmark import benchmark as benchmark_module
from benchmark.api_transport import finalize_stream_result
from benchmark.benchmark import (
    BenchmarkConfigError,
    _aggregate_spec_decode_metrics,
    _build_case_args,
    build_parser,
    load_suite_config,
    run_configured_suite,
)


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


def test_aiohttp_request_collects_speculative_decoding_metrics() -> None:
    """Per-request speculative metrics survive stream finalization."""
    speculative_metrics: dict[str, object] = {
        "mean_acceptance_length": 2.0,
        "draft_acceptance_rate": 0.5,
        "acceptance_histogram": [1, 0, 1],
        "num_spec_steps": 2,
        "num_accepted_draft_tokens": 1,
        "num_draft_tokens": 2,
        "num_spec_tokens": 2,
    }
    response = _FakeAsyncResponse([
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
        (
            'data: {"usage":{"completion_tokens":1,"prompt_tokens":2},'
            f'"metrics":{{"speculative_decoding":{json.dumps(speculative_metrics)}}}}}\n\n'
        ).encode("utf-8"),
        b"data: [DONE]",
    ])
    session = _FakeAsyncSession(response)

    result = _run(api_transport.send_aiohttp_chat_request(
        0,
        "prompt",
        "http://localhost:8000/v1/chat/completions",
        {},
        "model",
        4,
        session,
    ))
    finalize_stream_result(result, None)

    assert result["speculative_metrics"] == speculative_metrics


def test_spec_decode_metrics_aggregate_sum_and_rates() -> None:
    """Round metrics sum per-request counters and derive acceptance rates."""
    metrics = _aggregate_spec_decode_metrics(
        [
            {
                "speculative_metrics": {
                    "num_spec_steps": 2,
                    "num_accepted_draft_tokens": 1,
                    "num_draft_tokens": 3,
                    "num_spec_tokens": 3,
                    "acceptance_histogram": [1, 0, 1],
                },
            },
            {
                "speculative_metrics": {
                    "num_spec_steps": 3,
                    "num_accepted_draft_tokens": 2,
                    "num_draft_tokens": 3,
                    "num_spec_tokens": 3,
                    "acceptance_histogram": [0, 2, 0],
                },
            },
        ],
        successful_count=2,
    )

    assert metrics == {
        "source": "vllm_metrics.speculative_decoding",
        "aggregation": "sum",
        "request_count": 2,
        "missing_request_count": 0,
        "num_spec_steps": 5,
        "num_draft_tokens": 6,
        "num_accepted_draft_tokens": 3,
        "num_spec_tokens": 3,
        "draft_acceptance_rate": 0.5,
        "mean_acceptance_length": 0.6,
        "acceptance_histogram": [1, 2, 1],
    }
    assert metrics.get("per_position") is None


def test_spec_decode_metrics_aggregates_detailed_per_position_rates() -> None:
    """Detailed vLLM arrays produce each draft position's acceptance rate."""
    metrics = _aggregate_spec_decode_metrics(
        [
            {
                "speculative_metrics": {
                    "num_spec_steps": 3,
                    "num_accepted_draft_tokens": 2,
                    "num_draft_tokens": 3,
                    "num_spec_tokens": 3,
                    "per_step_accepted": [2, 0, 0],
                    "per_step_drafted": [2, 1, 0],
                },
            },
        ],
        successful_count=1,
    )

    assert metrics["per_position"] == [
        {
            "position": 1,
            "num_accepted_draft_tokens": 1,
            "num_draft_tokens": 2,
            "draft_acceptance_rate": 0.5,
        },
        {
            "position": 2,
            "num_accepted_draft_tokens": 1,
            "num_draft_tokens": 1,
            "draft_acceptance_rate": 1.0,
        },
    ]


def test_spec_decode_metrics_position_count_follows_each_server_method() -> None:
    """MTP and DSpark positions are inferred from their own reported arrays."""
    metrics = _aggregate_spec_decode_metrics(
        [
            {
                "speculative_metrics": {
                    "num_spec_steps": 2,
                    "num_accepted_draft_tokens": 3,
                    "num_draft_tokens": 6,
                    "num_spec_tokens": 3,
                    "per_step_accepted": [2, 1, 0],
                    "per_step_drafted": [2, 2, 2],
                },
            },
            {
                "speculative_metrics": {
                    "num_spec_steps": 1,
                    "num_accepted_draft_tokens": 1,
                    "num_draft_tokens": 7,
                    "num_spec_tokens": 7,
                    "per_step_accepted": [1, 0, 0, 0, 0, 0, 0],
                    "per_step_drafted": [7, 7, 7, 7, 7, 7, 7],
                },
            },
        ],
        successful_count=2,
    )

    assert metrics["num_spec_tokens"] == 7
    assert metrics["per_position_request_count"] == 2
    assert metrics["per_position"] == [
        {
            "position": 1,
            "num_accepted_draft_tokens": 3,
            "num_draft_tokens": 10,
            "draft_acceptance_rate": 0.3,
        },
        {
            "position": 2,
            "num_accepted_draft_tokens": 1,
            "num_draft_tokens": 10,
            "draft_acceptance_rate": 0.1,
        },
        {
            "position": 3,
            "num_accepted_draft_tokens": 0,
            "num_draft_tokens": 7,
            "draft_acceptance_rate": 0.0,
        },
        {
            "position": 4,
            "num_accepted_draft_tokens": 0,
            "num_draft_tokens": 7,
            "draft_acceptance_rate": 0.0,
        },
        {
            "position": 5,
            "num_accepted_draft_tokens": 0,
            "num_draft_tokens": 7,
            "draft_acceptance_rate": 0.0,
        },
        {
            "position": 6,
            "num_accepted_draft_tokens": 0,
            "num_draft_tokens": 7,
            "draft_acceptance_rate": 0.0,
        },
        {
            "position": 7,
            "num_accepted_draft_tokens": 0,
            "num_draft_tokens": 7,
            "draft_acceptance_rate": 0.0,
        },
    ]


def test_spec_decode_metrics_derives_per_position_for_full_draft_summary() -> None:
    """A full-draft summary exposes every server-reported position exactly."""
    metrics = _aggregate_spec_decode_metrics(
        [
            {
                "speculative_metrics": {
                    "num_spec_steps": 2,
                    "num_accepted_draft_tokens": 12,
                    "num_draft_tokens": 14,
                    "num_spec_tokens": 7,
                    "acceptance_histogram": [0, 0, 0, 0, 0, 0, 2, 0],
                },
            },
        ],
        successful_count=1,
    )

    assert metrics["per_position"] == [
        {
            "position": position,
            "num_accepted_draft_tokens": 2 if position < 7 else 0,
            "num_draft_tokens": 2,
            "draft_acceptance_rate": 1.0 if position < 7 else 0.0,
        }
        for position in range(1, 8)
    ]


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
    parser = build_parser()
    assert parser.parse_args([]).api_transport == "requests"
    assert parser.parse_args([]).api_timeout_seconds == 600.0
    assert parser.parse_args([]).target_rps is None
    assert parser.parse_args([]).rate_schedule == "fixed"
    assert parser.parse_args([]).rate_burstiness is None
    assert parser.parse_args([]).rate_ramp_up_strategy == "none"
    assert parser.parse_args([]).rate_ramp_up_start_rps is None
    assert parser.parse_args([]).rate_ramp_up_end_rps is None
    assert parser.parse_args([]).rate_overload_policy == "no-catch-up"
    assert not hasattr(parser.parse_args([]), "rate_burst")
    assert parser.parse_args(["--rate-burst", "3"])._legacy_rate_burst == 3
    assert parser.parse_args(["--target-rps", "2.5"]).target_rps == 2.5

    args, _ = _build_case_args(
        {
            "mode": "api",
            "api_transport": "aiohttp",
            "api_timeout_seconds": 12.5,
            "requests": {
                "target_rps": None,
                "rate_schedule": "gamma",
                "rate_burstiness": 0.5,
                "rate_ramp_up_strategy": "linear",
                "rate_ramp_up_start_rps": 1.0,
                "rate_ramp_up_end_rps": 2.0,
                "rate_overload_policy": "drop",
                "rate_burst": 3,
            },
        },
        {"name": "async-transport"},
        {},
    )

    assert args.api_transport == "aiohttp"
    assert args.api_timeout_seconds == 12.5
    assert args.target_rps is None
    assert args.rate_schedule == "gamma"
    assert args.rate_burstiness == 0.5
    assert args.rate_ramp_up_strategy == "linear"
    assert args.rate_ramp_up_start_rps == 1.0
    assert args.rate_ramp_up_end_rps == 2.0
    assert args.rate_overload_policy == "drop"
    assert args._legacy_rate_burst == 3


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_suite_config_rejects_invalid_target_rps(value: object) -> None:
    """拒绝会产生无意义或不可调度速率的配置值。"""
    with pytest.raises(BenchmarkConfigError, match="target_rps"):
        _build_case_args(
            {"mode": "api", "target_rps": value},
            {"name": "invalid-target-rps"},
            {},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rate_schedule", "unsupported", "rate_schedule"),
        ("rate_burstiness", 1.0, "rate_burstiness"),
        ("rate_schedule", "gamma", "rate_burstiness"),
        ("rate_burstiness", 0, "rate_burstiness"),
        ("rate_ramp_up_strategy", "unsupported", "rate_ramp_up_strategy"),
        ("rate_ramp_up_start_rps", 1.0, "rate_ramp_up_start_rps"),
    ],
)
def test_suite_config_rejects_invalid_rate_control(
    field: str,
    value: object,
    message: str,
) -> None:
    """拒绝无效的速率调度策略与 burst 上限。"""
    with pytest.raises(BenchmarkConfigError, match=message):
        _build_case_args(
            {"mode": "api", field: value},
            {"name": "invalid-rate-control"},
            {},
        )


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_suite_config_rejects_invalid_api_timeout_seconds(value: object) -> None:
    with pytest.raises(BenchmarkConfigError, match="api_timeout_seconds"):
        _build_case_args(
            {"mode": "api", "api_timeout_seconds": value},
            {"name": "invalid-api-timeout"},
            {},
        )


def test_suite_config_rejects_invalid_api_transport() -> None:
    with pytest.raises(BenchmarkConfigError, match="api_transport"):
        _build_case_args(
            {"mode": "api", "api_transport": "unsupported"},
            {"name": "invalid-transport"},
            {},
        )


def test_suite_config_rejects_ramp_outside_api_mode() -> None:
    """ramp 是 API 请求发生器能力，offline 模式必须拒绝。"""
    with pytest.raises(BenchmarkConfigError, match="rate_ramp_up_strategy"):
        _build_case_args(
            {
                "mode": "offline",
                "rate_ramp_up_strategy": "linear",
                "rate_ramp_up_start_rps": 1.0,
                "rate_ramp_up_end_rps": 2.0,
            },
            {"name": "offline-ramp"},
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
        *,
        timeout_seconds: float,
        target_rps: float | None,
        rate_schedule: str,
        rate_burstiness: float | None,
        rate_ramp_up_strategy: str,
        rate_ramp_up_start_rps: float | None,
        rate_ramp_up_end_rps: float | None,
        rate_overload_policy: str,
        rate_seed: int | None,
        arrival_due_times: list[float],
        on_admitted: Any,
        on_dropped: Any,
    ) -> None:
        observed.update({
            "prompts": prompts,
            "url": url,
            "headers": headers,
            "model": model,
            "max_tokens": max_tokens,
            "concurrency": concurrency,
            "ignore_eos": ignore_eos,
            "timeout_seconds": timeout_seconds,
            "target_rps": target_rps,
            "rate_schedule": rate_schedule,
            "rate_burstiness": rate_burstiness,
            "rate_ramp_up_strategy": rate_ramp_up_strategy,
            "rate_ramp_up_start_rps": rate_ramp_up_start_rps,
            "rate_ramp_up_end_rps": rate_ramp_up_end_rps,
            "rate_overload_policy": rate_overload_policy,
            "rate_seed": rate_seed,
            "arrival_due_times": arrival_due_times,
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
        api_transport="aiohttp",
        api_timeout_seconds=12.5,
        progress_reporter=_Reporter(),
    )

    assert observed == {
        "prompts": ["first", "second"],
        "url": "http://localhost/v1/chat/completions",
        "headers": {"Content-Type": "application/json"},
        "model": "model",
        "max_tokens": [2, 2],
        "concurrency": 2,
        "ignore_eos": True,
        "timeout_seconds": 12.5,
        "target_rps": None,
        "rate_schedule": "fixed",
        "rate_burstiness": None,
        "rate_ramp_up_strategy": "none",
        "rate_ramp_up_start_rps": None,
        "rate_ramp_up_end_rps": None,
        "rate_overload_policy": "no-catch-up",
        "rate_seed": None,
        "arrival_due_times": [0.0, 0.0],
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


def test_progress_parser_defaults_to_off_and_is_not_suite_parameter() -> None:
    """Keep progress opt-in and outside the suite JSON schema."""
    parser = build_parser()

    assert parser.parse_args([]).progress == "off"
    assert parser.parse_args(["--progress", "plain"]).progress == "plain"
    assert parser.parse_args(["--progress", "rich"]).progress == "rich"
    assert parser.parse_args(["--progress", "off"]).progress == "off"

    with pytest.raises(BenchmarkConfigError, match="unknown benchmark parameters: progress"):
        _build_case_args(
            {"mode": "api", "progress": "off"},
            {"name": "progress-is-cli-only"},
            {},
        )


class _FailingResponse:
    """构造包含敏感文本、但不得写入日志的错误响应。"""

    status_code = 401

    def __enter__(self) -> "_FailingResponse":  # noqa: UP037
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        raise RuntimeError("Authorization Bearer secret-token")


class _FailingRequestsClient:
    """返回一个确定性的失败 HTTP 响应。"""

    def post(self, _url: str, **_kwargs: Any) -> _FailingResponse:
        return _FailingResponse()


def test_debug_parser_and_root_config(tmp_path) -> None:
    """debug 可由 CLI 或 suite 根字段启用，不能作为单个 case 参数。"""
    parser = build_parser()

    assert parser.parse_args([]).debug is False
    assert parser.parse_args(["--debug"]).debug is True

    config_path = tmp_path / "debug-suite.json"
    config_path.write_text(
        json.dumps({"version": 1, "debug": True, "cases": [{"name": "smoke"}]}),
        encoding="utf-8",
    )
    assert load_suite_config(str(config_path))["debug"] is True

    config_path.write_text(
        json.dumps({"version": 1, "debug": "yes", "cases": [{"name": "smoke"}]}),
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkConfigError, match="debug must be true or false"):
        load_suite_config(str(config_path))

    with pytest.raises(BenchmarkConfigError, match="unknown benchmark parameters: debug"):
        _build_case_args(
            {"mode": "api", "debug": True},
            {"name": "debug-is-not-a-case-parameter"},
            {},
        )


def test_configured_suite_enables_root_debug_logging(tmp_path, capsys) -> None:
    """suite 根级 debug 应在执行前启用进程日志。"""
    from benchmark.logging_utils import configure_logging, get_logger

    config_path = tmp_path / "debug-suite.json"
    config_path.write_text(
        json.dumps({
            "version": 1,
            "debug": True,
            "defaults": {"mode": "api"},
            "cases": [{"name": "smoke"}],
        }),
        encoding="utf-8",
    )
    args = build_parser().parse_args([
        "--config", str(config_path), "--validate-config",
    ])

    try:
        assert run_configured_suite(str(config_path), args) == 0
        get_logger("test").debug("configured debug enabled")
    finally:
        configure_logging(False)

    assert "configured debug enabled" in capsys.readouterr().err


def test_debug_transport_log_excludes_url_credentials_and_exception_text(monkeypatch, capsys) -> None:
    """确认请求元数据可诊断，同时日志不包含密钥或原始 HTTP 错误。"""
    from benchmark.logging_utils import configure_logging

    configure_logging(True)
    try:
        result = api_transport.send_requests_chat_request(
            7,
            "private prompt",
            "https://user:password@example.com/v1/chat/completions?api_key=top-secret",
            {"Authorization": "Bearer top-secret"},
            "model",
            4,
            _FailingRequestsClient(),
        )
    finally:
        configure_logging(False)

    debug_output = capsys.readouterr().err
    assert result["error"] == "Authorization Bearer secret-token"
    assert "req_id=7" in debug_output
    assert "example.com/v1/chat/completions" in debug_output
    assert "password" not in debug_output
    assert "top-secret" not in debug_output
    assert "secret-token" not in debug_output


def test_aiohttp_planned_arrivals_queue_behind_concurrency_limit(monkeypatch) -> None:
    """计划到达时间不因满并发而重排，真实 HTTP 请求在 semaphore 后排队。"""
    started_at: list[float] = []
    admissions: list[tuple[int, float, float, float]] = []
    completed: list[int] = []

    class _FakeClientSession:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *_args: object) -> None:
            return None

    class _FakeAiohttp:
        class ClientTimeout:
            def __init__(self, **_kwargs: object) -> None:
                pass

        class TCPConnector:
            def __init__(self, **_kwargs: object) -> None:
                pass

        @staticmethod
        def ClientSession(**_kwargs: object) -> _FakeClientSession:
            return _FakeClientSession()

    async def slow_send(req_id: int, *_args: object, **_kwargs: object) -> dict[str, object]:
        started_at.append(api_transport.time.perf_counter())
        await asyncio.sleep(0.03)
        return {"req_id": req_id, "ttft": 0.001}

    monkeypatch.setattr(api_transport, "_require_aiohttp", lambda: _FakeAiohttp)
    monkeypatch.setattr(api_transport, "send_aiohttp_chat_request", slow_send)

    api_transport.run_aiohttp_chat_requests(
        ["first", "second", "third"],
        "http://localhost/v1/chat/completions",
        {},
        "model",
        [1, 1, 1],
        1,
        None,
        True,
        lambda result: completed.append(int(result["req_id"])),
        target_rps=100.0,
        arrival_due_times=[0.0, 0.01, 0.02],
        on_admitted=lambda req_id, timestamp, rate_wait, concurrency_wait: admissions.append(
            (req_id, timestamp, rate_wait, concurrency_wait)
        ),
    )

    intervals = [later - earlier for earlier, later in pairwise(started_at)]
    assert completed == [0, 1, 2]
    assert len(admissions) == 3
    assert min(intervals) >= 0.025
    assert admissions[1][3] >= 0.015
    assert admissions[2][3] >= 0.03


def test_aiohttp_drop_policy_skips_http_dispatch_when_concurrency_is_full(
    monkeypatch,
) -> None:
    """drop 策略在槽位耗尽时只回调丢弃记录，不创建 HTTP 请求。"""
    started: list[int] = []
    completed: list[int] = []
    dropped: list[int] = []

    class _FakeClientSession:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *_args: object) -> None:
            return None

    class _FakeAiohttp:
        class ClientTimeout:
            def __init__(self, **_kwargs: object) -> None:
                pass

        class TCPConnector:
            def __init__(self, **_kwargs: object) -> None:
                pass

        @staticmethod
        def ClientSession(**_kwargs: object) -> _FakeClientSession:
            return _FakeClientSession()

    async def slow_send(req_id: int, *_args: object, **_kwargs: object) -> dict[str, object]:
        started.append(req_id)
        await asyncio.sleep(0.05)
        return {"req_id": req_id, "ttft": 0.001}

    monkeypatch.setattr(api_transport, "_require_aiohttp", lambda: _FakeAiohttp)
    monkeypatch.setattr(api_transport, "send_aiohttp_chat_request", slow_send)

    api_transport.run_aiohttp_chat_requests(
        ["first", "second", "third"],
        "http://localhost/v1/chat/completions",
        {},
        "model",
        [1, 1, 1],
        1,
        None,
        True,
        lambda result: completed.append(int(result["req_id"])),
        target_rps=1000.0,
        rate_overload_policy="drop",
        arrival_due_times=[0.0, 0.001, 0.002],
        on_dropped=dropped.append,
    )

    assert started == [0]
    assert completed == [0]
    assert dropped == [1, 2]
