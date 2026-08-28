"""Terminal progress reporters for benchmark execution."""

from __future__ import annotations

import os
import sys
import threading
import time
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

    def stage_started(self, stage: str, detail: str | None = None) -> None:
        """Record that a setup or warmup stage started."""

    def stage_finished(self, stage: str, detail: str | None = None) -> None:
        """Record that a setup or warmup stage finished."""

    def event(self, message: str) -> None:
        """Record an informational benchmark event."""

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


def _load_rich_dashboard_components() -> dict[str, Any]:
    """Import Rich lazily so plain progress needs no UI dependency at import time."""
    try:
        from rich.console import Console
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError as exc:
        raise ProgressDependencyError(
            "--progress rich requires the 'rich' package; install the benchmark "
            "package with its standard dependencies."
        ) from exc
    return {
        "Console": Console,
        "Layout": Layout,
        "Live": Live,
        "Panel": Panel,
        "Table": Table,
        "Text": Text,
    }


class RichProgressReporter(ProgressReporter):
    """Render a full-screen, non-interactive Rich benchmark dashboard."""

    def __init__(self, stream: TextIO) -> None:
        components = _load_rich_dashboard_components()
        self._Layout = components["Layout"]
        self._Panel = components["Panel"]
        self._Table = components["Table"]
        self._Text = components["Text"]
        self._console = components["Console"](
            file=stream,
            force_terminal=stream.isatty(),
        )
        self._Live = components["Live"]
        self._lock = threading.Lock()
        self._live: Any | None = None
        self._started_perf = time.perf_counter()
        self._case_name = "等待开始"
        self._scenario = "-"
        self._case_status = "pending"
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
        self._events: list[str] = []

    def case_started(
        self, case_name: str, scenario: str, position: int, total_cases: int
    ) -> None:
        with self._lock:
            self._case_name = case_name
            self._scenario = scenario
            self._case_status = "running"
            self._case_position = position
            self._total_cases = total_cases
            self._record_event(f"case {position}/{total_cases} 开始: {case_name}")
            self._ensure_live()
            self._refresh()

    def case_finished(self, status: str, completed_cases: int, total_cases: int) -> None:
        with self._lock:
            self._case_status = status
            self._case_position = completed_cases
            self._total_cases = total_cases
            self._record_event(f"case {completed_cases}/{total_cases} 状态: {status}")
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
            self._record_event(
                f"round 开始: requests={total_requests}，concurrency={concurrency}"
            )
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
            if self._last_error:
                self._record_event(
                    "请求失败: "
                    f"req_id={snapshot.get('request_id', 'unknown')} {self._last_error}"
                )
            self._refresh()

    def round_finished(self) -> None:
        with self._lock:
            self._record_event(
                f"round 完成: 成功={self._succeeded}，失败={self._failed}"
            )
            self._refresh()

    def stage_started(self, stage: str, detail: str | None = None) -> None:
        """Show that a setup or warmup stage started."""
        self._record_stage_event("开始", stage, detail)

    def stage_finished(self, stage: str, detail: str | None = None) -> None:
        """Show that a setup or warmup stage finished."""
        self._record_stage_event("完成", stage, detail)

    def event(self, message: str) -> None:
        """Append a non-request event to the dashboard."""
        with self._lock:
            self._record_event(message)
            self._ensure_live()
            self._refresh()

    def close(self) -> None:
        with self._lock:
            if self._live is not None:
                self._live.stop()
                self._live = None

    def _ensure_live(self) -> None:
        if self._live is None:
            self._live = self._Live(
                self._render(),
                console=self._console,
                refresh_per_second=8,
                screen=True,
                transient=True,
            )
            self._live.start()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render(), refresh=True)

    def _record_stage_event(
        self, state: str, stage: str, detail: str | None
    ) -> None:
        with self._lock:
            suffix = f"：{detail}" if detail else ""
            self._record_event(f"阶段{state}: {stage}{suffix}")
            self._ensure_live()
            self._refresh()

    def _record_event(self, message: str) -> None:
        self._events.append(message)
        del self._events[:-8]

    def _render(self) -> Any:
        terminal_width = self._console.size.width
        layout = self._Layout(name="root")
        layout.split_column(
            self._Layout(name="header", size=3),
            self._Layout(name="main", ratio=1),
            self._Layout(name="events", size=10),
            self._Layout(name="footer", size=3),
        )
        if terminal_width >= 100:
            layout["main"].split_row(
                self._Layout(name="suite"),
                self._Layout(name="round"),
            )
        else:
            layout["main"].split_column(
                self._Layout(name="suite"),
                self._Layout(name="round"),
            )

        layout["header"].update(self._header_panel())
        layout["suite"].update(self._suite_panel())
        layout["round"].update(self._round_panel())
        layout["events"].update(self._events_panel())
        layout["footer"].update(self._footer_panel())
        return layout

    def _header_panel(self) -> Any:
        elapsed = time.perf_counter() - self._started_perf
        title = self._Text(
            f"LLM Benchmark Dashboard  ·  运行 {elapsed:.1f}s  ·  非交互监控",
            justify="center",
            style="bold white",
        )
        return self._Panel(title, style="blue", padding=(0, 1))

    def _suite_panel(self) -> Any:
        table = self._status_table("Suite / Case")
        table.add_row("进度", f"{self._case_position}/{self._total_cases}")
        table.add_row("当前 case", self._case_name)
        table.add_row("场景", self._scenario)
        table.add_row("状态", self._case_status)
        return self._Panel(table, title="Suite / Case", border_style="cyan")

    def _round_panel(self) -> Any:
        table = self._status_table("当前 Round")
        table.add_row("请求", f"{self._completed}/{self._round_total}")
        table.add_row("并发", str(self._concurrency))
        table.add_row("成功 / 失败", f"{self._succeeded} / {self._failed}")
        table.add_row("Round 耗时", f"{self._elapsed_seconds:.1f}s")
        table.add_row(
            "最新 TTFT",
            f"{self._last_ttft:.3f}s" if self._last_ttft is not None else "等待首 token",
        )
        return self._Panel(table, title="当前 Round", border_style="green")

    def _events_panel(self) -> Any:
        table = self._Table.grid(expand=True)
        table.add_column(overflow="fold")
        for event in self._events or ["等待 benchmark 事件..."]:
            table.add_row(event)
        return self._Panel(table, title="最近事件", border_style="yellow")

    def _footer_panel(self) -> Any:
        status = self._last_error or "正常运行"
        return self._Panel(
            f"实时刷新 8 Hz  ·  最近状态: {status}",
            title="运行信息",
            border_style="magenta",
            padding=(0, 1),
        )

    def _status_table(self, _title: str) -> Any:
        table = self._Table.grid(padding=(0, 1), expand=True)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(overflow="fold")
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
        A reporter instance. ``auto`` uses the full-screen Rich dashboard for
        interactive terminals and line-oriented output otherwise.

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
