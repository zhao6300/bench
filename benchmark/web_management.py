"""Push benchmark lifecycle events to a local Node.js web dashboard.

Web pushes are best-effort: failures never fail the benchmark, because the dashboard is an
optional monitoring aid rather than part of the measurement path.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from typing import Any

try:
    from .progress import ProgressReporter
except ImportError:  # pragma: no cover - exercised via direct script execution
    from progress import ProgressReporter


class WebPushConfigError(RuntimeError):
    """Raised when the requested web dashboard target is invalid."""


def resolve_web_target(host: str | None, port: int | None) -> str | None:
    """Return the loopback ingest target URL, or None when web pushing is disabled.

    Args:
        host: ``--web-host`` value, defaulting to the loopback address when None.
        port: ``--web-port`` value required to enable the dashboard.

    Returns:
        The ingest URL (``http://<host>:<port>/ingest``) when pushing is enabled,
        otherwise None.

    Raises:
        WebPushConfigError: If ``--web-port`` is missing but ``--web-host`` was
            given, the port is outside the valid range, or the value is not an int.
    """
    if host is None and port is None:
        return None
    effective_host = host or "127.0.0.1"
    if port is None:
        raise WebPushConfigError("--web-port is required when --web-host is provided")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        raise WebPushConfigError("--web-port must be an integer between 1 and 65535")
    return f"http://{effective_host}:{port}/ingest"


def _web_token() -> str | None:
    """Return the optional dashboard bearer token from the environment."""
    return os.environ.get("LLM_BENCHMARK_WEB_TOKEN") or None


def post_event(target: str, payload: dict[str, Any], token: str | None) -> None:
    """POST a JSON event payload to a local dashboard, swallowing connection errors.

    Args:
        target: Ingest URL to receive the event.
        payload: JSON-serializable event body forwarded to the dashboard.
        token: Optional bearer token sent as an Authorization header.
    """
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(target, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=2.0) as response:
            response.read()  # Drain and release the connection.
    except Exception:
        pass  # Web pushes are best-effort; they never affect benchmark results.


class WebPushProgressReporter(ProgressReporter):
    """Mirror progress events to a local Node.js dashboard through a wrapped reporter.

    The wrapped reporter controls terminal output behavior (Rich dashboard, plain lines, or
    nothing); every lifecycle event is additionally forwarded to the configured dashboard.
    Web pushes are best-effort and never raise, matching their monitoring-only role.

    ``captures_runtime_output`` is intentionally set from the wrapped reporter rather than
    declared here so that outermost callers observe the wrapped reporter's behavior.
    """

    def __init__(self, base: ProgressReporter, target: str) -> None:
        self._base = base
        self._target: str = target
        self._token = _web_token()
        self._lock = threading.Lock()
        self.captures_runtime_output = bool(
            getattr(base, "captures_runtime_output", False)
        )

    def _publish(self, payload: dict[str, Any]) -> None:
        with self._lock:
            post_event(self._target, payload, self._token)

    def case_started(
        self, case_name: str, scenario: str, position: int, total_cases: int
    ) -> None:
        self._base.case_started(case_name, scenario, position, total_cases)
        self._publish({
            "event": "case_started",
            "case_name": case_name,
            "scenario": scenario,
            "position": position,
            "total_cases": total_cases,
            "message": f"case {position}/{total_cases} 开始: {case_name} ({scenario})",
        })

    def case_finished(self, status: str, completed_cases: int, total_cases: int) -> None:
        self._base.case_finished(status, completed_cases, total_cases)
        self._publish({
            "event": "case_finished",
            "status": status,
            "completed_cases": completed_cases,
            "total_cases": total_cases,
            "message": f"case {completed_cases}/{total_cases} 状态: {status}",
        })

    def round_started(self, total_requests: int, concurrency: int) -> None:
        self._base.round_started(total_requests, concurrency)
        self._publish({
            "event": "round_started",
            "round_total": total_requests,
            "concurrency": concurrency,
            "message": f"round 开始: requests={total_requests}，concurrency={concurrency}",
        })

    def request_finished(self, snapshot: dict[str, Any]) -> None:
        self._base.request_finished(snapshot)
        error = snapshot.get("error")
        if error:
            message = (
                "请求失败: "
                f"req_id={snapshot.get('request_id', 'unknown')} 原因: {error}"
            )
        else:
            ttft_text = ""
            ttft = snapshot.get("last_ttft")
            if isinstance(ttft, (int, float)):
                ttft_text = f"，最新 TTFT={ttft:.3f}s"
            message = (
                "进度: "
                f"{snapshot.get('completed', 0)}/{snapshot.get('total', 0)}，"
                f"成功={snapshot.get('succeeded', 0)}，"
                f"失败={snapshot.get('failed', 0)}，"
                f"耗时={snapshot.get('elapsed_seconds', 0.0):.1f}s{ttft_text}"
            )
        self._publish({
            "event": "request_finished",
            **snapshot,
            "message": message,
        })

    def round_finished(self) -> None:
        self._base.round_finished()
        self._publish({"event": "round_finished", "message": "round 完成"})

    def stage_started(self, stage: str, detail: str | None = None) -> None:
        self._base.stage_started(stage, detail)
        suffix = f"：{detail}" if detail else ""
        self._publish({
            "event": "stage_started",
            "stage": stage,
            "detail": detail,
            "message": f"stage {stage} 开始{suffix}",
        })

    def stage_finished(self, stage: str, detail: str | None = None) -> None:
        self._base.stage_finished(stage, detail)
        suffix = f"：{detail}" if detail else ""
        self._publish({
            "event": "stage_finished",
            "stage": stage,
            "detail": detail,
            "message": f"stage {stage} 完成{suffix}",
        })

    def event(self, message: str) -> None:
        self._base.event(message)
        self._publish({"event": "event", "message": message})

    def show_final_results(
        self, report: dict[str, Any], report_location: str | None = None
    ) -> None:
        self._base.show_final_results(report, report_location)
        self._publish({
            "event": "final_results",
            "report": report,
            "report_location": report_location,
        })

    def close(self) -> None:
        try:
            self._base.close()
        finally:
            self._publish({"event": "dashboard_closed"})
