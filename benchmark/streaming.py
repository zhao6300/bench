"""Parse streaming server-sent events without optional HTTP dependencies."""

from __future__ import annotations

import codecs
import json
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator


class SSEEventDecoder:
    """Decode byte chunks into complete server-sent event frames."""

    def __init__(self) -> None:
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")()

    def add_chunk(self, chunk: bytes) -> list[str]:
        """Add a response chunk and return complete event frames.

        Args:
            chunk: A raw response body fragment.

        Returns:
            Complete SSE event frames without trailing separators.
        """
        self._buffer += self._decoder.decode(chunk)
        self._buffer = self._buffer.replace("\r\n", "\n")
        events = []
        while "\n\n" in self._buffer:
            event, self._buffer = self._buffer.split("\n\n", 1)
            event = event.strip()
            if event:
                events.append(event)

        if self._buffer.startswith("data: "):
            data = self._buffer.removeprefix("data: ").strip()
            if data == "[DONE]":
                events.append(self._buffer.strip())
                self._buffer = ""
            elif data:
                try:
                    json.loads(data)
                except json.JSONDecodeError:
                    pass
                else:
                    events.append(self._buffer.strip())
                    self._buffer = ""
        return events


def iter_sse_data(chunks: Iterable[bytes]) -> Iterator[str]:
    """Yield data payloads from arbitrary SSE response byte chunks.

    Args:
        chunks: Raw byte chunks from a streaming HTTP response.

    Yields:
        The trimmed payload of each `data:` SSE event, including `[DONE]`.
    """
    decoder = SSEEventDecoder()
    for chunk in chunks:
        if not chunk:
            continue
        for event in decoder.add_chunk(chunk):
            if event.startswith(":") or not event.startswith("data:"):
                continue
            yield event.removeprefix("data:").strip()


async def aiter_sse_data(chunks: AsyncIterable[bytes]) -> AsyncIterator[str]:
    """Asynchronously yield data payloads from SSE response byte chunks.

    Args:
        chunks: Raw byte chunks from an asynchronous streaming HTTP response.

    Yields:
        The trimmed payload of each `data:` SSE event, including `[DONE]`.
    """
    decoder = SSEEventDecoder()
    async for chunk in chunks:
        if not chunk:
            continue
        for event in decoder.add_chunk(chunk):
            if event.startswith(":") or not event.startswith("data:"):
                continue
            yield event.removeprefix("data:").strip()
