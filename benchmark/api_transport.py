"""Transport adapters for OpenAI-compatible streaming chat benchmarks."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

try:
    from .logging_utils import exception_type, get_logger, safe_endpoint
    from .request_schedule import build_arrival_plan
    from .streaming import aiter_sse_data, iter_sse_data
except ImportError:
    from logging_utils import exception_type, get_logger, safe_endpoint
    from request_schedule import build_arrival_plan
    from streaming import aiter_sse_data, iter_sse_data


LOGGER = get_logger("api")


def build_chat_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    """Build the shared OpenAI-compatible streaming chat payload.

    Args:
        model: The server model name.
        prompt: The user message content.
        max_tokens: Requested maximum completion tokens.
        ignore_eos: Whether the server should ignore EOS.

    Returns:
        The request JSON payload used by every chat transport.
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos,
    }


def _failure_result(req_id: int | None, error: str, started_at: float) -> RequestResult:
    """Create the established failed request record.

    Args:
        req_id: The request identifier.
        error: Human-readable failure reason.
        started_at: The monotonic request start time.

    Returns:
        A failed record compatible with benchmark metric aggregation.
    """
    return {
        "req_id": req_id,
        "error": error,
        "ttft": None,
        "tpot": None,
        "request_start_timestamp": started_at,
        "first_token_timestamp": None,
        "last_token_timestamp": None,
        "first_token_text": "",
        "prompt_tokens": None,
        "output_tokens": None,
        "output_token_source": "unavailable",
        "total_time": time.perf_counter() - started_at,
        "estimated_itl_samples": [],
    }


class _StreamRecordBuilder:
    """Collect stream timestamps and text without tokenizing in the I/O path."""

    def __init__(self, req_id: int) -> None:
        self.req_id = req_id
        self.started_at = time.perf_counter()
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None
        self.first_token_text = ""
        self.content_events: list[tuple[str, float]] = []
        self.full_text = ""
        self.usage_completion_tokens: int | None = None
        self.usage_prompt_tokens: int | None = None

    def consume(self, data: str) -> None:
        """Consume one SSE data payload without blocking on tokenizer work.

        Args:
            data: The payload from a `data:` SSE event.
        """
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return

        usage = chunk.get("usage")
        if usage:
            if usage.get("completion_tokens") is not None:
                self.usage_completion_tokens = usage["completion_tokens"]
            if usage.get("prompt_tokens") is not None:
                self.usage_prompt_tokens = usage["prompt_tokens"]

        choices = chunk.get("choices", [])
        if not choices:
            return
        delta = choices[0].get("delta", {})
        text = (
            delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("reasoning")
        )
        if not text:
            return

        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
            self.first_token_text = text
        self.last_token_time = now
        self.content_events.append((text, now))
        self.full_text += text

    def finish(self) -> RequestResult:
        """Build an unfinalized record for post-round token processing.

        Returns:
            A request record with private stream data for finalization after all
            network streams in the round have completed.
        """
        if self.first_token_time is None:
            result = _failure_result(
                self.req_id,
                "未收到首个 token",
                self.started_at,
            )
            result["prompt_tokens"] = self.usage_prompt_tokens
            return result

        return {
            "req_id": self.req_id,
            "error": None,
            "ttft": self.first_token_time - self.started_at,
            "tpot": None,
            "request_start_timestamp": self.started_at,
            "first_token_timestamp": self.first_token_time,
            "last_token_timestamp": self.last_token_time,
            "first_token_text": self.first_token_text,
            "prompt_tokens": self.usage_prompt_tokens,
            "output_tokens": None,
            "output_token_source": "unavailable",
            "total_time": time.perf_counter() - self.started_at,
            "estimated_itl_samples": [],
            "_stream_content_events": self.content_events,
            "_stream_full_text": self.full_text,
            "_stream_usage_completion_tokens": self.usage_completion_tokens,
        }


def _count_tokens(tokenizer: Any, text: str) -> int:
    """Count text tokens with support for tokenizer-compatible interfaces.

    Args:
        tokenizer: The tokenizer used for local fallback counting.
        text: Non-empty streamed text.

    Returns:
        The number of locally encoded tokens, with a minimum of one.
    """
    try:
        return max(1, len(tokenizer.encode(text, add_special_tokens=False)))
    except (TypeError, ValueError):
        return max(1, len(tokenizer.encode(text)))


def finalize_stream_result(result: RequestResult, tokenizer: Any | None) -> None:
    """Finalize token-derived metrics after a benchmark round's streams end.

    Args:
        result: A mutable request record emitted by a stream transport.
        tokenizer: Optional tokenizer for local output and chunk token counting.
    """
    content_events = result.pop("_stream_content_events", None)
    full_text = result.pop("_stream_full_text", "")
    usage_completion_tokens = result.pop("_stream_usage_completion_tokens", None)
    if not content_events:
        return

    output_tokens: int | None = None
    output_token_source = "unavailable"
    if (
        isinstance(usage_completion_tokens, int)
        and usage_completion_tokens >= 0
    ):
        output_tokens = usage_completion_tokens
        output_token_source = "server_usage"
    elif tokenizer is not None and full_text:
        output_tokens = _count_tokens(tokenizer, full_text)
        output_token_source = "local_tokenizer"

    estimated_itl_samples = []
    previous_received_at = content_events[0][1]
    for text, received_at in content_events[1:]:
        chunk_tokens = _count_tokens(tokenizer, text) if tokenizer is not None else 1
        estimated_itl = (received_at - previous_received_at) / chunk_tokens
        estimated_itl_samples.extend([estimated_itl] * chunk_tokens)
        previous_received_at = received_at

    first_token_time = result["first_token_timestamp"]
    last_token_time = result["last_token_timestamp"]
    tpot = None
    if (
        output_tokens is not None
        and output_tokens > 1
        and last_token_time is not None
        and last_token_time > first_token_time
    ):
        tpot = (last_token_time - first_token_time) / (output_tokens - 1)

    result["output_tokens"] = output_tokens
    result["output_token_source"] = output_token_source
    result["tpot"] = tpot
    result["estimated_itl_samples"] = estimated_itl_samples


def finalize_stream_results(results: list[RequestResult], tokenizer: Any | None) -> None:
    """Finalize all request records after a round's network activity ends.

    Args:
        results: Mutable request records collected for one benchmark round.
        tokenizer: Optional tokenizer for local output and chunk token counting.
    """
    for result in results:
        finalize_stream_result(result, tokenizer)


def send_requests_chat_request(
    req_id: int,
    prompt: str,
    url: str,
    headers: dict[str, str],
    model: str,
    max_tokens: int,
    requests_client: Any,
    tokenizer: Any | None = None,
    ignore_eos: bool = True,
    timeout_seconds: float = 600.0,
) -> RequestResult:
    """Send one streaming chat request through requests.

    Args:
        req_id: The request identifier.
        prompt: The user message content.
        url: The chat completions URL.
        headers: Prevalidated request headers.
        model: The server model name.
        max_tokens: Requested maximum completion tokens.
        requests_client: The requests-compatible module or session.
        tokenizer: Optional tokenizer for output-token fallback.
        ignore_eos: Whether the server should ignore EOS.
        timeout_seconds: Timeout applied to the streaming HTTP request.

    Returns:
        A benchmark request record.
    """
    record = _StreamRecordBuilder(req_id)
    status_code = None
    LOGGER.debug(
        "requests request started req_id=%s endpoint=%s max_tokens=%s",
        req_id,
        safe_endpoint(url),
        max_tokens,
    )
    try:
        with requests_client.post(
            url,
            json=build_chat_payload(model, prompt, max_tokens, ignore_eos),
            headers=headers,
            stream=True,
            timeout=timeout_seconds,
        ) as response:
            status_code = getattr(response, "status_code", None)
            response.raise_for_status()
            for data in iter_sse_data(response.iter_content(chunk_size=None)):
                if data == "[DONE]":
                    break
                record.consume(data)
    except Exception as exc:
        LOGGER.debug(
            "requests request failed req_id=%s status=%s exception=%s",
            req_id,
            status_code,
            exception_type(exc),
        )
        return _failure_result(req_id, str(exc), record.started_at)
    result = record.finish()
    LOGGER.debug(
        "requests request finished req_id=%s status=%s ttft=%s total_time=%s",
        req_id,
        status_code,
        result.get("ttft"),
        result.get("total_time"),
    )
    return result


async def send_aiohttp_chat_request(
    req_id: int,
    prompt: str,
    url: str,
    headers: dict[str, str],
    model: str,
    max_tokens: int,
    session: Any,
    tokenizer: Any | None = None,
    ignore_eos: bool = True,
) -> RequestResult:
    """Send one streaming chat request through an aiohttp-compatible session.

    Args:
        req_id: The request identifier.
        prompt: The user message content.
        url: The chat completions URL.
        headers: Prevalidated request headers.
        model: The server model name.
        max_tokens: Requested maximum completion tokens.
        session: An aiohttp client session.
        tokenizer: Optional tokenizer for output-token fallback.
        ignore_eos: Whether the server should ignore EOS.

    Returns:
        A benchmark request record.
    """
    record = _StreamRecordBuilder(req_id)
    status_code = None
    LOGGER.debug(
        "aiohttp request started req_id=%s endpoint=%s max_tokens=%s",
        req_id,
        safe_endpoint(url),
        max_tokens,
    )
    try:
        async with session.post(
            url,
            json=build_chat_payload(model, prompt, max_tokens, ignore_eos),
            headers=headers,
        ) as response:
            status_code = getattr(response, "status", None)
            response.raise_for_status()
            async for data in aiter_sse_data(response.content.iter_any()):
                if data == "[DONE]":
                    break
                record.consume(data)
    except Exception as exc:
        LOGGER.debug(
            "aiohttp request failed req_id=%s status=%s exception=%s",
            req_id,
            status_code,
            exception_type(exc),
        )
        return _failure_result(req_id, str(exc), record.started_at)
    result = record.finish()
    LOGGER.debug(
        "aiohttp request finished req_id=%s status=%s ttft=%s total_time=%s",
        req_id,
        status_code,
        result.get("ttft"),
        result.get("total_time"),
    )
    return result


def _require_aiohttp() -> Any:
    """Load aiohttp only when the asynchronous transport is selected.

    Returns:
        The imported aiohttp module.

    Raises:
        RuntimeError: If aiohttp is not installed.
    """
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError(
            "api_transport=aiohttp requires aiohttp; install the API dependencies "
            "with uv pip install -r requirements/common.txt"
        ) from exc
    return aiohttp


def run_aiohttp_chat_requests(
    prompts: list[str],
    url: str,
    headers: dict[str, str],
    model: str,
    max_tokens: list[int],
    concurrency: int,
    tokenizer: Any | None,
    ignore_eos: bool,
    on_complete: Callable[[RequestResult], None],
    *,
    timeout_seconds: float = 600.0,
    target_rps: float | None = None,
    rate_schedule: str = "fixed",
    rate_burstiness: float | None = None,
    rate_ramp_up_strategy: str = "none",
    rate_ramp_up_start_rps: float | None = None,
    rate_ramp_up_end_rps: float | None = None,
    rate_overload_policy: str = "no-catch-up",
    rate_seed: int | None = None,
    arrival_due_times: list[float] | None = None,
    on_admitted: Callable[[int, float, float, float], None] | None = None,
    on_dropped: Callable[[int], None] | None = None,
) -> None:
    """通过共享 aiohttp session 运行一轮并发 chat 请求。

    Args:
        prompts: 每条请求的 Prompt 文本。
        url: chat completions URL。
        headers: 已校验的请求头。
        model: 服务端模型名。
        max_tokens: 每条请求的输出 token 上限。
        concurrency: 最大在途请求数和连接池上限。
        tokenizer: 可选的输出 token 回退 tokenizer。
        ignore_eos: 是否要求服务端忽略 EOS。
        on_complete: 每条完成记录的回调。
        timeout_seconds: 每条流式 HTTP 请求的总超时。
        target_rps: 可选的目标请求准入速率。
        rate_schedule: 到达间隔分布，可为 fixed、poisson 或 gamma。
        rate_burstiness: gamma 调度的 shape 参数；无穷大表示固定间隔。
        rate_ramp_up_strategy: 按请求序号插值的速率爬升策略。
        rate_ramp_up_start_rps: ramp 的起始请求速率。
        rate_ramp_up_end_rps: ramp 的结束请求速率。
        rate_overload_policy: 已计划到达而并发满时的处理策略。
        rate_seed: 随机到达过程使用的可复现随机种子。
        arrival_due_times: 由调用方预生成的相对到达时间。
        on_admitted: 记录真实 HTTP 准入时间和排队等待的回调。
        on_dropped: 请求在 HTTP 发送前被丢弃时的回调。

    Raises:
        RuntimeError: aiohttp 不可用或当前已有运行中的事件循环。
    """
    aiohttp = _require_aiohttp()
    LOGGER.debug(
        "aiohttp round started endpoint=%s requests=%d concurrency=%d",
        safe_endpoint(url),
        len(prompts),
        concurrency,
    )

    scheduled_due_times = arrival_due_times

    async def run() -> None:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        connector = aiohttp.TCPConnector(limit=concurrency)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async def send_one(req_id: int, prompt: str) -> RequestResult:
                return await send_aiohttp_chat_request(
                    req_id,
                    prompt,
                    url,
                    headers,
                    model,
                    max_tokens[req_id],
                    session,
                    tokenizer,
                    ignore_eos,
                )

            if scheduled_due_times is None:
                arrival_plan = build_arrival_plan(
                    len(prompts),
                    target_rps=target_rps,
                    rate_schedule=rate_schedule,
                    rate_burstiness=rate_burstiness,
                    rate_seed=rate_seed,
                    rate_ramp_up_strategy=rate_ramp_up_strategy,
                    rate_ramp_up_start_rps=rate_ramp_up_start_rps,
                    rate_ramp_up_end_rps=rate_ramp_up_end_rps,
                )
                due_times = arrival_plan.due_times_seconds
                arrival_mode = arrival_plan.schedule
            else:
                if len(scheduled_due_times) != len(prompts):
                    raise ValueError("arrival_due_times must match prompts length")
                due_times = scheduled_due_times
                arrival_mode = "unbounded" if target_rps is None else rate_schedule

            semaphore = asyncio.Semaphore(concurrency)

            async def send_with_limit(
                req_id: int,
                prompt: str,
                arrived_at: float,
            ) -> RequestResult:
                async with semaphore:
                    admitted_at = time.perf_counter()
                    if on_admitted is not None:
                        on_admitted(req_id, admitted_at, 0.0, admitted_at - arrived_at)
                    return await send_one(req_id, prompt)

            tasks: list[asyncio.Task[RequestResult]] = []
            schedule_started_at = time.perf_counter()
            for req_id, prompt in enumerate(prompts):
                due_at = schedule_started_at + due_times[req_id]
                remaining = due_at - time.perf_counter()
                if remaining > 0:
                    await asyncio.sleep(remaining)
                arrived_at = time.perf_counter()
                active_tasks = sum(not task.done() for task in tasks)
                if (
                    arrival_mode != "unbounded"
                    and rate_overload_policy == "drop"
                    and active_tasks >= concurrency
                ):
                    if on_dropped is not None:
                        on_dropped(req_id)
                    continue
                tasks.append(
                    asyncio.create_task(send_with_limit(req_id, prompt, arrived_at))
                )

            for task in asyncio.as_completed(tasks):
                on_complete(await task)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(run())
    else:
        raise RuntimeError(
            "api_transport=aiohttp requires a synchronous benchmark entry point"
        )
