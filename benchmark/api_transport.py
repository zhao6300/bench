"""Transport adapters for OpenAI-compatible streaming chat benchmarks."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

try:
    from .streaming import aiter_sse_data, iter_sse_data
except ImportError:
    from streaming import aiter_sse_data, iter_sse_data


RequestResult = dict[str, Any]


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

    Returns:
        A benchmark request record.
    """
    record = _StreamRecordBuilder(req_id)
    try:
        with requests_client.post(
            url,
            json=build_chat_payload(model, prompt, max_tokens, ignore_eos),
            headers=headers,
            stream=True,
            timeout=600,
        ) as response:
            response.raise_for_status()
            for data in iter_sse_data(response.iter_content(chunk_size=None)):
                if data == "[DONE]":
                    break
                record.consume(data)
    except Exception as exc:
        return _failure_result(req_id, str(exc), record.started_at)
    return record.finish()


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
    try:
        async with session.post(
            url,
            json=build_chat_payload(model, prompt, max_tokens, ignore_eos),
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for data in aiter_sse_data(response.content.iter_any()):
                if data == "[DONE]":
                    break
                record.consume(data)
    except Exception as exc:
        return _failure_result(req_id, str(exc), record.started_at)
    return record.finish()


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
) -> None:
    """Run one concurrent chat round through a shared aiohttp session.

    Args:
        prompts: Prompt text for each request.
        url: The chat completions URL.
        headers: Prevalidated request headers.
        model: The server model name.
        max_tokens: Per-request completion limits.
        concurrency: Maximum in-flight requests and connector limit.
        tokenizer: Optional tokenizer for output-token fallback.
        ignore_eos: Whether the server should ignore EOS.
        on_complete: Callback invoked once for each completed record.

    Raises:
        RuntimeError: If aiohttp is unavailable or an event loop is already running.
    """
    aiohttp = _require_aiohttp()

    async def run() -> None:
        timeout = aiohttp.ClientTimeout(total=600)
        connector = aiohttp.TCPConnector(limit=concurrency)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            semaphore = asyncio.Semaphore(concurrency)

            async def send_one(req_id: int, prompt: str) -> RequestResult:
                async with semaphore:
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

            tasks = [
                asyncio.create_task(send_one(req_id, prompt))
                for req_id, prompt in enumerate(prompts)
            ]
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
