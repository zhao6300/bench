"""Terminal progress reporters for benchmark execution."""

from __future__ import annotations

import math
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

    def show_final_results(
        self, report: dict[str, Any], report_location: str | None = None
    ) -> None:
        """Optionally render final benchmark results before closing."""

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
    """Render a full-screen Rich dashboard and interactive final result view."""

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
        self._final_report: dict[str, Any] | None = None
        self._final_report_location: str | None = None

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

    def show_final_results(
        self, report: dict[str, Any], report_location: str | None = None
    ) -> None:
        """Show a final result table until the user exits the dashboard.

        Args:
            report: Fully finalized benchmark report used to populate the table.
            report_location: Optional persisted JSON report location.
        """
        with self._lock:
            self._final_report = report
            self._final_report_location = report_location
            self._ensure_live()
            self._refresh()
        self._wait_for_final_exit()

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

    def _wait_for_final_exit(self) -> None:
        """Wait for Q or Ctrl-C when a terminal input stream is available."""
        try:
            while True:
                key = self._read_final_key()
                if key is None or key.lower() == "q":
                    return
        except KeyboardInterrupt:
            return

    @staticmethod
    def _read_final_key() -> str | None:
        """Read one cbreak-mode terminal key without requiring Enter."""
        stream = sys.stdin
        if not getattr(stream, "isatty", lambda: False)():
            return None
        try:
            import termios
            import tty
        except ImportError:
            return None
        try:
            file_descriptor = stream.fileno()
            original_settings = termios.tcgetattr(file_descriptor)
        except (AttributeError, OSError, termios.error):
            return None
        try:
            tty.setcbreak(file_descriptor)
            return stream.read(1)
        finally:
            termios.tcsetattr(file_descriptor, termios.TCSADRAIN, original_settings)

    def _render(self) -> Any:
        if self._final_report is not None:
            return self._render_final_results()
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

    def _render_final_results(self) -> Any:
        """Build the completion view shown after benchmark results are persisted."""
        summary = self._final_report.get("summary", {}) if self._final_report else {}
        layout = self._Layout(name="root")
        layout.split_column(
            self._Layout(name="header", size=3),
            self._Layout(name="results", ratio=1),
            self._Layout(name="footer", size=3),
        )
        header = self._Text(
            "LLM Benchmark 最终结果",
            justify="center",
            style="bold white",
        )
        layout["header"].update(self._Panel(header, style="green", padding=(0, 1)))
        layout["results"].update(self._final_results_panel(summary))
        report_text = (
            f"JSON report: {self._final_report_location}"
            if self._final_report_location
            else "JSON report: 未输出"
        )
        footer = f"{report_text}  ·  按 Q 退出  ·  Ctrl-C 退出"
        layout["footer"].update(
            self._Panel(footer, title="结果已完成", border_style="magenta", padding=(0, 1))
        )
        return layout

    def _final_results_panel(self, summary: dict[str, Any]) -> Any:
        """Return a width-adaptive table for all completed benchmark cases."""
        profile = self._final_table_profile()
        table = self._Table(expand=True, show_lines=True, padding=(0, 1))
        for name, options in self._final_table_columns(profile):
            table.add_column(name, **options)
        for record in (self._final_report or {}).get("cases", []):
            if isinstance(record, dict):
                table.add_row(*self._final_result_row(record, profile))
        if not table.rows:
            table.add_row(*(["-"] * (len(self._final_table_columns(profile)) - 1)), "无可显示的用例")
        title = (
            "最终结果 · "
            f"通过 {summary.get('passed', 0)}  失败 {summary.get('failed', 0)}  "
            f"中断 {summary.get('interrupted', 0)}  跳过 {summary.get('skipped', 0)}"
        )
        return self._Panel(table, title=title, border_style="green")

    def _final_table_profile(self) -> str:
        """Select final-table detail based on the current terminal width."""
        width = self._console.size.width
        if width < 100:
            return "narrow"
        if width < 160:
            return "medium"
        return "wide"

    @staticmethod
    def _final_table_columns(profile: str) -> list[tuple[str, dict[str, Any]]]:
        """Return column definitions for one final-table width profile."""
        if profile == "narrow":
            return [
                ("Case", {"style": "cyan", "overflow": "fold"}),
                ("场景 / 状态", {"style": "blue", "overflow": "fold"}),
                ("请求 / 负载", {"justify": "right", "overflow": "fold"}),
                ("核心结果", {"justify": "right", "overflow": "fold"}),
                ("说明", {"overflow": "fold"}),
            ]
        if profile == "medium":
            return [
                ("Case", {"style": "cyan", "overflow": "fold"}),
                ("场景 / 状态", {"style": "blue", "overflow": "fold"}),
                ("请求", {"justify": "right", "overflow": "fold"}),
                ("负载", {"justify": "right", "overflow": "fold"}),
                ("延迟", {"justify": "right", "overflow": "fold"}),
                ("吞吐", {"justify": "right", "overflow": "fold"}),
                ("QPS / Goodput", {"justify": "right", "overflow": "fold"}),
                ("Cache / GPU", {"justify": "right", "overflow": "fold"}),
                ("场景结果", {"overflow": "fold"}),
                ("说明", {"overflow": "fold"}),
            ]
        return [
            ("Case", {"style": "cyan", "overflow": "fold"}),
            ("场景 / 状态", {"style": "blue", "overflow": "fold"}),
            ("并发 / 请求", {"justify": "right", "overflow": "fold"}),
            ("成功 / 失败", {"justify": "right", "overflow": "fold"}),
            ("负载 P / O / 共享", {"justify": "right", "overflow": "fold"}),
            ("TTFT avg / P50 / P99", {"justify": "right", "overflow": "fold"}),
            ("TPOT avg / P50 / P99", {"justify": "right", "overflow": "fold"}),
            ("E2E avg / P99", {"justify": "right", "overflow": "fold"}),
            ("吞吐 P / Pre / Dec / All", {"justify": "right", "overflow": "fold"}),
            ("QPS / Goodput", {"justify": "right", "overflow": "fold"}),
            ("Cache / GPU", {"justify": "right", "overflow": "fold"}),
            ("场景结果", {"overflow": "fold"}),
            ("说明", {"overflow": "fold"}),
        ]

    def _final_result_row(self, record: dict[str, Any], profile: str) -> tuple[str, ...]:
        """Format one finalized case record for a final-table width profile."""
        result = record.get("result")
        result = result if isinstance(result, dict) else {}
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            metrics = result.get("selected_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        scenario = str(record.get("scenario") or result.get("scenario") or "-")
        name = str(record.get("name", "-"))
        status = str(record.get("status", "-"))
        message = self._final_result_message(record)
        request_summary = self._final_request_summary(metrics)
        workload_summary = self._final_workload_summary(result, metrics)
        latency_summary = self._final_latency_summary(metrics)
        throughput_summary = self._final_throughput_summary(result, metrics)
        qps_goodput_summary = self._final_qps_goodput_summary(metrics)
        server_summary = self._final_server_summary(metrics)
        scenario_summary = self._final_scenario_summary(result, scenario)

        if profile == "narrow":
            return (
                name,
                f"{scenario}\n{status}",
                self._final_join_lines(request_summary, workload_summary),
                self._final_join_lines(
                    latency_summary,
                    throughput_summary,
                    qps_goodput_summary,
                    server_summary,
                    scenario_summary,
                ),
                message,
            )
        if profile == "medium":
            return (
                name,
                f"{scenario}\n{status}",
                request_summary,
                workload_summary,
                latency_summary,
                throughput_summary,
                qps_goodput_summary,
                server_summary,
                scenario_summary,
                message,
            )
        return (
            name,
            f"{scenario}\n{status}",
            self._final_concurrency_requests(metrics),
            self._final_success_failure_rate(metrics),
            workload_summary,
            self._final_latency_values(metrics, "ttft"),
            self._final_latency_values(metrics, "tpot"),
            self._final_latency_values(metrics, "e2e"),
            throughput_summary,
            qps_goodput_summary,
            server_summary,
            scenario_summary,
            message,
        )

    def _final_request_summary(self, metrics: dict[str, Any]) -> str:
        """Format request and concurrency aggregates for compact layouts."""
        return self._final_join_lines(
            self._final_concurrency_requests(metrics),
            self._final_success_failure_rate(metrics),
        )

    def _final_concurrency_requests(self, metrics: dict[str, Any]) -> str:
        """Format configured concurrency and aggregate request count."""
        parts = []
        concurrency = self._format_integer(metrics.get("concurrency"))
        total = self._format_integer(metrics.get("total_requests"))
        if concurrency != "-":
            parts.append(f"并发 {concurrency}")
        if total != "-":
            parts.append(f"请求 {total}")
        wall_time = self._format_seconds(metrics.get("wall_time"))
        if wall_time != "-":
            parts.append(f"耗时 {wall_time}")
        return " · ".join(parts) if parts else "-"

    def _final_success_failure_rate(self, metrics: dict[str, Any]) -> str:
        """Format success, failure, and failure-rate aggregates."""
        success_failure = self._format_success_failure(metrics)
        failure_rate = self._format_percent(metrics.get("failure_rate"))
        if success_failure == "-":
            return "-"
        return (
            f"成功/失败 {success_failure} · 失败 {failure_rate}"
            if failure_rate != "-"
            else f"成功/失败 {success_failure}"
        )

    def _final_workload_summary(
        self, result: dict[str, Any], metrics: dict[str, Any]
    ) -> str:
        """Format request-shape and measured token aggregates when available."""
        workload = result.get("selected_workload")
        if not isinstance(workload, dict):
            workload = result.get("workload")
        if isinstance(workload, dict) and isinstance(workload.get("summary"), dict):
            workload = workload["summary"]
        workload = workload if isinstance(workload, dict) else {}
        prompt = self._final_stat_value(workload.get("prompt_tokens"))
        output = self._final_stat_value(workload.get("requested_output_tokens"))
        shared = self._final_stat_value(workload.get("shared_prefix_tokens"))
        shape_parts = []
        if prompt != "-":
            shape_parts.append(f"P {prompt}")
        if output != "-":
            shape_parts.append(f"O {output}")
        if shared != "-":
            shape_parts.append(f"共享 {shared}")
        actual_prompt = self._format_token_count(metrics.get("total_prompt_tokens"))
        actual_generated = self._format_token_count(metrics.get("total_generated_tokens"))
        actual_parts = []
        if actual_prompt != "-":
            actual_parts.append(f"P {actual_prompt}")
        if actual_generated != "-":
            actual_parts.append(f"G {actual_generated}")
        return self._final_join_lines(
            " / ".join(shape_parts) if shape_parts else "-",
            f"总计 {' / '.join(actual_parts)}" if actual_parts else "-",
        )

    def _final_latency_summary(self, metrics: dict[str, Any]) -> str:
        """Format the primary latency aggregates for compact layouts."""
        return self._final_join_lines(
            self._final_latency_values(metrics, "ttft", compact=True),
            self._final_latency_values(metrics, "tpot", compact=True),
            self._final_latency_values(metrics, "e2e", compact=True),
            self._final_labeled_value(
                "请求均耗", self._format_milliseconds(metrics.get("avg_request_time"))
            ),
        )

    def _final_latency_values(
        self, metrics: dict[str, Any], metric: str, compact: bool = False
    ) -> str:
        """Format latency percentiles in milliseconds with explicit labels."""
        metric_keys = {
            "ttft": ("TTFT", "avg_ttft", "p50_ttft", "p99_ttft"),
            "tpot": ("TPOT", "avg_tpot", "p50_tpot", "p99_tpot"),
            "e2e": ("E2E", "avg_total_time", "p50_e2e", "p99_e2e"),
        }
        label, average_key, p50_key, p99_key = metric_keys[metric]
        average = self._format_milliseconds(metrics.get(average_key))
        p50 = self._format_milliseconds(metrics.get(p50_key))
        p99 = self._format_milliseconds(metrics.get(p99_key))
        if average == p50 == p99 == "-":
            return "-"
        if compact:
            return f"{label} avg/P99 {average}/{p99}"
        return f"{average} / {p50} / {p99}"

    def _final_throughput_summary(
        self, result: dict[str, Any], metrics: dict[str, Any]
    ) -> str:
        """Format all available phase and overall throughput aggregates."""
        labels = (
            ("P", "prompt_throughput"),
            ("Pre", "prefill_throughput"),
            ("Dec", "decode_throughput"),
            ("All", "overall_throughput"),
        )
        parts = [
            f"{label} {self._format_throughput(metrics.get(key))}"
            for label, key in labels
            if self._format_throughput(metrics.get(key)) != "-"
        ]
        if parts:
            return "\n".join(parts)
        best = self._format_throughput(result.get("best_throughput"))
        if best != "-":
            return f"峰值 {best}"
        prefill = result.get("prefill")
        decode = result.get("decode")
        if isinstance(prefill, dict) and isinstance(decode, dict):
            return self._final_join_lines(
                self._final_labeled_value(
                    "Pre", self._format_throughput(prefill.get("prefill_throughput"))
                ),
                self._final_labeled_value(
                    "Dec", self._format_throughput(decode.get("decode_throughput"))
                ),
            )
        return "-"

    def _final_qps_goodput_summary(self, metrics: dict[str, Any]) -> str:
        """Format completed-request rate and SLO compliance aggregates."""
        return self._final_join_lines(
            self._final_labeled_value("QPS", self._format_qps(metrics.get("qps"))),
            self._final_labeled_value(
                "Goodput", self._format_percent(metrics.get("goodput_pct"))
            ),
            self._final_labeled_value(
                "GP QPS", self._format_qps(metrics.get("goodput_qps"))
            ),
            self._final_labeled_value(
                "SLO TTFT≤", self._format_milliseconds(metrics.get("slo_ttft"))
            ),
            self._final_labeled_value(
                "SLO TPOT≤", self._format_milliseconds(metrics.get("slo_tpot"))
            ),
        )

    def _final_server_summary(self, metrics: dict[str, Any]) -> str:
        """Format cache, GPU, and queue observations from server metrics."""
        server_metrics = metrics.get("server_metrics")
        if not isinstance(server_metrics, dict):
            server_metrics = metrics.get("server_metrics_peak")
        raw_metrics = server_metrics.get("metrics") if isinstance(server_metrics, dict) else {}
        raw_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        cache = self._format_percent(self._final_stat_number(raw_metrics.get("cache_hit_rate")))
        gpu = self._format_percent(self._final_stat_number(raw_metrics.get("gpu_cache_usage_pct")))
        cpu = self._format_percent(self._final_stat_number(raw_metrics.get("cpu_cache_usage_pct")))
        running = self._format_number(self._final_stat_number(raw_metrics.get("running_requests")), 1)
        waiting = self._format_number(self._final_stat_number(raw_metrics.get("waiting_requests")), 1)
        return self._final_join_lines(
            self._final_labeled_value("Cache", cache),
            self._final_labeled_value("GPU", gpu),
            self._final_labeled_value("CPU", cpu),
            (
                f"运行/等待 {running}/{waiting}"
                if running != "-" or waiting != "-"
                else "-"
            ),
        )

    def _final_scenario_summary(self, result: dict[str, Any], scenario: str) -> str:
        """Format scenario-specific aggregates without inferring missing rounds."""
        if scenario == "sweep":
            return self._final_join_lines(
                self._final_labeled_value("指标", self._format_text(result.get("metric"))),
                self._final_labeled_value(
                    "最佳并发", self._format_integer(result.get("best_concurrency"))
                ),
                self._final_labeled_value(
                    "峰值", self._format_throughput(result.get("best_throughput"))
                ),
            )
        if scenario == "slo-capacity-search":
            boundary = result.get("refined_failure_boundary")
            if not isinstance(boundary, int):
                boundary = result.get("failure_boundary")
            return self._final_join_lines(
                self._final_labeled_value(
                    "最大通过", self._format_integer(result.get("max_passing_concurrency"))
                ),
                self._final_labeled_value(
                    "确认", self._format_integer(result.get("confirmed_concurrency"))
                ),
                self._final_labeled_value("失败边界", self._format_integer(boundary)),
                self._final_labeled_value(
                    "要求 Goodput", self._format_percent(result.get("required_goodput_pct"))
                ),
                self._final_labeled_value(
                    "最大失败", self._format_percent_from_ratio(result.get("max_failure_rate"))
                ),
            )
        if scenario == "pd-ratio":
            analysis = result.get("analysis")
            analysis = analysis if isinstance(analysis, dict) else {}
            ratio = analysis.get("recommended_ratio")
            ratio = ratio if isinstance(ratio, dict) else {}
            prefill = self._format_integer(ratio.get("prefill"))
            decode = self._format_integer(ratio.get("decode"))
            recommendation = analysis.get("recommended")
            recommendation_text = (
                "推荐分离" if recommendation is True
                else "不建议分离" if recommendation is False
                else "-"
            )
            ratio_text = f"建议 P:D {prefill}:{decode}" if prefill != "-" and decode != "-" else "-"
            return self._final_join_lines(
                ratio_text,
                self._final_labeled_value(
                    "Prefill 占比", self._format_percent(analysis.get("prefill_share_pct"))
                ),
                self._final_labeled_value(
                    "Decode 占比", self._format_percent(analysis.get("decode_share_pct"))
                ),
                self._final_labeled_value("建议", recommendation_text),
            )
        return "-"

    def _final_throughput(
        self, result: dict[str, Any], metrics: dict[str, Any]
    ) -> str:
        """Select a concise throughput value for backward-compatible callers."""
        summary = self._final_throughput_summary(result, metrics)
        return summary.split("\n", maxsplit=1)[0] if summary != "-" else "-"

    @staticmethod
    def _final_join_lines(*values: str) -> str:
        """Join non-empty table values on separate lines, or return a placeholder."""
        meaningful = [value for value in values if value and value != "-"]
        return "\n".join(meaningful) if meaningful else "-"

    @staticmethod
    def _final_labeled_value(label: str, value: str) -> str:
        """Add a label only when a table value is available."""
        return f"{label} {value}" if value != "-" else "-"

    @staticmethod
    def _format_integer(value: Any) -> str:
        """Format an integer-like result value for a table cell."""
        return str(value) if isinstance(value, int) and not isinstance(value, bool) else "-"

    def _format_success_failure(self, metrics: dict[str, Any]) -> str:
        """Format request success and failure totals for a table cell."""
        successful = self._format_integer(metrics.get("successful"))
        failed = self._format_integer(metrics.get("failed"))
        return f"{successful}/{failed}" if successful != "-" or failed != "-" else "-"

    @staticmethod
    def _is_number(value: Any) -> bool:
        """Return whether a value is a finite numeric metric, excluding booleans."""
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    @classmethod
    def _format_number(cls, value: Any, precision: int) -> str:
        """Format a numeric measurement for a table cell."""
        return f"{value:.{precision}f}" if cls._is_number(value) else "-"

    @classmethod
    def _format_seconds(cls, value: Any) -> str:
        """Format a duration metric in seconds."""
        return f"{value:.2f} s" if cls._is_number(value) else "-"

    @classmethod
    def _format_milliseconds(cls, value: Any) -> str:
        """Format a seconds metric as milliseconds."""
        return f"{value * 1000:.1f} ms" if cls._is_number(value) else "-"

    @classmethod
    def _format_throughput(cls, value: Any) -> str:
        """Format a token-throughput metric with its unit."""
        return f"{value:.1f} tok/s" if cls._is_number(value) else "-"

    @classmethod
    def _format_qps(cls, value: Any) -> str:
        """Format a request-rate metric with its unit."""
        return f"{value:.2f} req/s" if cls._is_number(value) else "-"

    @classmethod
    def _format_percent(cls, value: Any) -> str:
        """Format a percentage metric already expressed on a 0–100 scale."""
        return f"{value:.1f}%" if cls._is_number(value) else "-"

    @classmethod
    def _format_percent_from_ratio(cls, value: Any) -> str:
        """Format a fraction metric as a percentage."""
        return f"{value * 100:.1f}%" if cls._is_number(value) else "-"

    @classmethod
    def _format_token_count(cls, value: Any) -> str:
        """Format a token count compactly while retaining its token unit."""
        if not cls._is_number(value):
            return "-"
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M tok"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.1f}K tok"
        return f"{value:.0f} tok"

    @classmethod
    def _final_stat_value(cls, value: Any) -> str:
        """Format a workload statistic's average value as a token count."""
        if isinstance(value, dict):
            value = value.get("avg")
        return cls._format_token_count(value)

    @classmethod
    def _final_stat_number(cls, value: Any) -> float | int | None:
        """Extract an average, maximum, or scalar number from server statistics."""
        if isinstance(value, dict):
            for key in ("avg", "max"):
                candidate = value.get(key)
                if cls._is_number(candidate):
                    return candidate
            return None
        return value if cls._is_number(value) else None

    @staticmethod
    def _format_text(value: Any) -> str:
        """Format a non-empty textual result value for a table cell."""
        return str(value) if isinstance(value, str) and value else "-"

    @staticmethod
    def _final_result_message(record: dict[str, Any]) -> str:
        """Extract a concise failure, quality, or skip message for a case."""
        error = record.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        quality_failures = record.get("quality_failures")
        if isinstance(quality_failures, list) and quality_failures:
            return "; ".join(str(item) for item in quality_failures)
        return str(record.get("skip_reason") or "-")

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
