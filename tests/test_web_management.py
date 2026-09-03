"""Unit tests for the web dashboard push reporter.

The dashboard is an optional best-effort monitoring aid: configuration errors surface through the documented
WebPushConfigError, while network push failures must never propagate back into the benchmark.
"""

from __future__ import annotations

import json

import pytest

from benchmark.progress import OffProgressReporter
from benchmark.web_management import (
    WebPushConfigError,
    WebPushProgressReporter,
    post_event,
    resolve_web_target,
)


class _FakeResponse:
    """Minimal read-only stand-in returned by the mocked urlopen call."""

    def __init__(self) -> None:
        self.read = lambda: b"{}"


def _capture_urlopen(captured: list) -> callable:
    """Return a urlopen stand-in that records parsed request bodies."""

    def fake_urlopen(request, timeout=None):
        captured.append(json.loads(request.data))
        return _FakeResponse()

    return fake_urlopen


def test_resolve_web_target_disabled_when_both_options_absent() -> None:
    """Without a port, pushing stays disabled and no URL is built."""
    assert resolve_web_target(None, None) is None


def test_resolve_web_target_defaults_host_to_loopback() -> None:
    """A port enables pushing on the loopback ingest endpoint by default."""
    assert resolve_web_target(None, 3000) == "http://127.0.0.1:3000/ingest"


def test_resolve_web_target_keeps_explicit_host() -> None:
    """An explicit host is honored verbatim in the ingest URL."""
    assert resolve_web_target("localhost", 3000) == "http://localhost:3000/ingest"


def test_resolve_web_target_requires_port_when_host_given() -> None:
    """A host without a port is a user configuration error."""
    with pytest.raises(WebPushConfigError):
        resolve_web_target("127.0.0.1", None)


@pytest.mark.parametrize("host,port", [("127.0.0.1", 0), ("127.0.0.1", 70000)])
def test_resolve_web_target_rejects_invalid_port(host, port) -> None:
    """Out-of-range ports are rejected with an actionable error."""
    with pytest.raises(WebPushConfigError):
        resolve_web_target(host, port)


def test_resolve_web_target_rejects_non_int_port() -> None:
    """A non-integer port value is rejected rather than silently coerced."""
    with pytest.raises(WebPushConfigError):
        resolve_web_target(None, "3000")


def test_post_event_posts_json_body(monkeypatch) -> None:
    """A valid payload is POSTed once as JSON with the request drained."""
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        "urllib.request.urlopen", _capture_urlopen(captured), raising=True
    )
    monkeypatch.delenv("LLM_BENCHMARK_WEB_TOKEN", raising=False)

    post_event("http://127.0.0.1:3000/ingest", {"event": "event", "message": "x"}, None)

    assert captured == [{"event": "event", "message": "x"}]


def test_post_event_swallows_connection_failure(monkeypatch) -> None:
    """Network errors never propagate to the caller (best-effort pushing)."""

    def boom(request, timeout=None):
        raise RuntimeError("dashboard offline")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    post_event("http://127.0.0.1:3000/ingest", {"event": "event"}, None)


def test_reporter_inherits_base_terminal_behaviour(monkeypatch) -> None:
    """The wrapper reports the wrapped reporter's captures_runtime_output value."""
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse()
    )
    reporter = WebPushProgressReporter(
        OffProgressReporter(), "http://127.0.0.1:3000/ingest"
    )
    assert reporter.captures_runtime_output is False


def test_reporter_publishes_lifecycle_events(monkeypatch) -> None:
    """Lifecycle events are forwarded to the dashboard in arrival order."""
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        "urllib.request.urlopen", _capture_urlopen(captured), raising=True
    )
    reporter = WebPushProgressReporter(
        OffProgressReporter(), "http://127.0.0.1:3000/ingest"
    )

    reporter.case_started("smoke", "single", 1, 2)
    reporter.request_finished({
        "completed": 1, "total": 2, "succeeded": 1, "failed": 0,
        "elapsed_seconds": 0.5, "last_ttft": 0.4, "error": None,
    })
    reporter.close()

    assert [event["event"] for event in captured] == [
        "case_started", "request_finished", "dashboard_closed",
    ]
    assert captured[0]["case_name"] == "smoke"
