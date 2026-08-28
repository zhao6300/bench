"""Terminal progress reporters for benchmark execution."""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, TextIO


class ProgressDependencyError(RuntimeError):
    """Raised when an explicitly requested progress renderer is unavailable."""


class ProgressReporter:
    """Receive benchmark lifecycle events without affecting benchmark results."""

    def case_started(
        self, case_name: str, scenario: str, position: int, total_cases: int
    ) -> None:
        """Record that a benchmark case started."""

    def case_finished(self, status: str, completed_cases: int, total_cases: int) -> None:
        """Record that a benchmark case reached a terminal status."""

    def round_started(self, total_requests: int, concurrency: int) -> None:
        """Record that one request-bearing benchmark round started."""

    def request_finished(self, snapshot: dict[str, Any]) -> None:
        """Record completion of one benchmark request."""

    def round_finished(self) -> None:
        """Record that the current benchmark round finished."""

    def close(self) -> None:
        """Release terminal resources held by the reporter."""


class OffProgressReporter(ProgressReporter):
    """Suppress all real-time progress output."""


class PlainProgressReporter(ProgressReporter):
    """Render portable line-oriented progress updates."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def case_started(
        self, case_name: str, scenario: str, position: int, total_cases: int
    ) -> None:
        self._write(f"进度: case {position}/{total_cases} 开始: {case_name} ({scenario})")

    def case_finished(self, status: str, completed_cases: int, total_cases: int) -> None:
        self._write(f"进度: case {completed_cases}/{total_cases} 状态: {status}")

    def round_started(self, total_requests: int, concurrency: int) -> None:
        self._write(
            f"进度: round 开始，requests={total_requests}，concurrency={concurrency}"
        )

    def request_finished(self, snapshot: dict[str, Any]) -> None:
        error = snapshot.get("error")
        if error:
            self._write(
                "请求失败: "
                f"req_id={snapshot.get('request_id', 'unknown')} 原因: {error}"
            )
        ttft = snapshot.get("last_ttft")
        ttft_text = f"，最新 TTFT={ttft:.3f}s" if isinstance(ttft, (int, float)) else ""
        self._write(
            "进度: "
            f"{snapshot['completed']}/{snapshot['total']}，成功={snapshot['succeeded']}，"
            f"失败={snapshot['failed']}，耗时={snapshot['elapsed_seconds']:.1f}s{ttft_text}"
        )

    def round_finished(self) -> None:
        self._write("进度: round 完成")

    def _write(self, message: str) -> None:
        with self._lock:
            print(message, file=self._stream, flush=True)


class RichProgressReporter(ProgressReporter):
    """Render a non-interactive Rich live status panel."""

    def __init__(self, stream: TextIO) -> None:
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.table import Table
        except ImportError as exc:
            raise ProgressDependencyError(
                "--progress rich requires the 'rich' package; install the benchmark "
                "package with its standard dependencies."
            ) from exc

        self._Console = Console
        self._Live = Live
        self._Table = Table
        self._console = Console(file=stream, force_terminal=stream.isatty())
        self._lock = threading.Lock()
        self._live: Any | None = None
        self._case_name = "等待开始"
        self._scenario = "-"
        self._case_position = 0
        self._total_cases = 0
        self._round_total = 0
        self._concurrency = 0
        self._completed = 0
        self._succeeded = 0
        self._failed = 0
        self._elapsed_seconds = 0.0
        self._last_ttft: float | None = None
        self._last_error: str | None = None

    def case_started(
        self, case_name: str, scenario: str, position: int, total_cases: int
    ) -> None:
        with self._lock:
            self._case_name = case_name
            self._scenario = scenario
            self._case_position = position
            self._total_cases = total_cases
            self._ensure_live()
            self._refresh()

    def case_finished(self, status: str, completed_cases: int, total_cases: int) -> None:
        with self._lock:
            self._case_position = completed_cases
            self._total_cases = total_cases
            self._last_error = self._last_error or f"case 状态: {status}"
            self._refresh()

    def round_started(self, total_requests: int, concurrency: int) -> None:
        with self._lock:
            self._round_total = total_requests
            self._concurrency = concurrency
            self._completed = 0
            self._succeeded = 0
            self._failed = 0
            self._elapsed_seconds = 0.0
            self._last_ttft = None
            self._last_error = None
            self._ensure_live()
            self._refresh()

    def request_finished(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._completed = snapshot["completed"]
            self._succeeded = snapshot["succeeded"]
            self._failed = snapshot["failed"]
            self._elapsed_seconds = snapshot["elapsed_seconds"]
            self._last_ttft = snapshot.get("last_ttft")
            self._last_error = snapshot.get("error")
            self._refresh()

    def round_finished(self) -> None:
        with self._lock:
            self._refresh()

    def close(self) -> None:
        with self._lock:
            if self._live is not None:
                self._live.stop()
                self._live = None

    def _ensure_live(self) -> None:
        if self._live is None:
            self._live = self._Live(
                self._render(), console=self._console, refresh_per_second=8, transient=True
            )
            self._live.start()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render(), refresh=True)

    def _render(self) -> Any:
        table = self._Table(title="LLM Benchmark 进度", show_header=False)
        table.add_column("字段", style="cyan", no_wrap=True)
        table.add_column("状态")
        table.add_row("Case", f"{self._case_position}/{self._total_cases}: {self._case_name}")
        table.add_row("Scenario", self._scenario)
        if self._round_total:
            table.add_row(
                "Round",
                f"{self._completed}/{self._round_total}，并发={self._concurrency}，"
                f"成功={self._succeeded}，失败={self._failed}，耗时={self._elapsed_seconds:.1f}s",
            )
        if self._last_ttft is not None:
            table.add_row("最新 TTFT", f"{self._last_ttft:.3f}s")
        if self._last_error:
            table.add_row("最近状态", self._last_error)
        return table


def resolve_progress_mode(mode: str, stream: TextIO = sys.stderr) -> str:
    """Resolve auto progress mode from terminal capabilities.

    Args:
        mode: Requested ``auto``, ``plain``, ``rich``, or ``off`` mode.
        stream: Output stream whose terminal capabilities are inspected.

    Returns:
        The effective ``plain``, ``rich``, or ``off`` progress mode.

    Raises:
        ValueError: If mode is not a supported progress mode.
    """
    if mode not in {"auto", "plain", "rich", "off"}:
        raise ValueError(f"unsupported progress mode: {mode!r}")
    if mode != "auto":
        return mode
    is_tty = getattr(stream, "isatty", lambda: False)()
    return "rich" if is_tty and os.environ.get("TERM", "").lower() != "dumb" else "plain"


def create_progress_reporter(
    mode: str, stream: TextIO = sys.stderr
) -> ProgressReporter:
    """Create the progress reporter selected by a CLI progress mode.

    Args:
        mode: Requested ``auto``, ``plain``, ``rich``, or ``off`` mode.
        stream: Stream for real-time status output. Defaults to standard error.

    Returns:
        A reporter instance. ``auto`` uses Rich for interactive terminals and
        line-oriented output otherwise.

    Raises:
        ProgressDependencyError: If ``rich`` is explicitly selected without
            the Rich dependency being installed.
    """
    resolved_mode = resolve_progress_mode(mode, stream)
    if resolved_mode == "off":
        return OffProgressReporter()
    if resolved_mode == "plain":
        return PlainProgressReporter(stream)
    return RichProgressReporter(stream)
