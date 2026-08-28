"""Unit tests for terminal progress reporters."""

from __future__ import annotations

import io
import json

from benchmark import benchmark as benchmark_module
from benchmark.benchmark import build_parser
from benchmark.progress import (
    OffProgressReporter,
    PlainProgressReporter,
    resolve_progress_mode,
)


class _TerminalStream(io.StringIO):
    """String buffer with configurable terminal capability."""

    def __init__(self, is_terminal: bool) -> None:
        super().__init__()
        self._is_terminal = is_terminal

    def isatty(self) -> bool:
        return self._is_terminal


def test_auto_progress_uses_rich_only_for_supported_terminal(monkeypatch) -> None:
    """Keep redirected and dumb-terminal output readable as plain text."""
    monkeypatch.setenv("TERM", "xterm-256color")

    assert resolve_progress_mode("auto", _TerminalStream(True)) == "rich"
    assert resolve_progress_mode("auto", _TerminalStream(False)) == "plain"

    monkeypatch.setenv("TERM", "dumb")
    assert resolve_progress_mode("auto", _TerminalStream(True)) == "plain"


def test_plain_progress_renders_request_status_and_failure() -> None:
    """Expose request counts, timing, and errors without terminal control codes."""
    stream = _TerminalStream(False)
    reporter = PlainProgressReporter(stream)

    reporter.case_started("smoke", "single", 1, 2)
    reporter.round_started(2, 1)
    reporter.request_finished({
        "completed": 1,
        "total": 2,
        "succeeded": 0,
        "failed": 1,
        "elapsed_seconds": 0.5,
        "last_ttft": None,
        "request_id": 0,
        "error": "timeout",
    })
    reporter.case_finished("failed", 1, 2)

    output = stream.getvalue()
    assert "case 1/2 开始: smoke (single)" in output
    assert "requests=2，concurrency=1" in output
    assert "请求失败: req_id=0 原因: timeout" in output
    assert "进度: 1/2，成功=0，失败=1，耗时=0.5s" in output
    assert "case 1/2 状态: failed" in output


def test_off_progress_suppresses_lifecycle_output(capsys) -> None:
    """Allow CI and callers to disable only real-time status rendering."""
    reporter = OffProgressReporter()

    reporter.case_started("smoke", "single", 1, 1)
    reporter.round_started(1, 1)
    reporter.request_finished({
        "completed": 1,
        "total": 1,
        "succeeded": 1,
        "failed": 0,
        "elapsed_seconds": 0.1,
        "last_ttft": 0.1,
        "request_id": 0,
        "error": None,
    })
    reporter.round_finished()
    reporter.case_finished("passed", 1, 1)
    reporter.close()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


class _SuiteReporter:
    """Capture suite lifecycle events without rendering a terminal UI."""

    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def case_started(
        self, case_name: str, scenario: str, position: int, total_cases: int
    ) -> None:
        self.events.append(("started", case_name, scenario, position, total_cases))

    def case_finished(self, status: str, completed_cases: int, total_cases: int) -> None:
        self.events.append(("finished", status, completed_cases, total_cases))

    def round_started(self, _total_requests: int, _concurrency: int) -> None:
        pass

    def request_finished(self, _snapshot: dict[str, object]) -> None:
        pass

    def round_finished(self) -> None:
        pass

    def close(self) -> None:
        self.events.append(("closed",))


def test_configured_suite_reports_case_lifecycle_without_network(
    monkeypatch, tmp_path
) -> None:
    """Emit suite events around the shared executor without changing reports."""
    config_path = tmp_path / "suite.json"
    report_path = tmp_path / "report.json"
    config_path.write_text(
        json.dumps({
            "version": 1,
            "name": "progress-suite",
            "defaults": {"mode": "offline", "model": "placeholder"},
            "cases": [{"name": "smoke"}],
        }),
        encoding="utf-8",
    )
    reporter = _SuiteReporter()
    monkeypatch.setattr(
        benchmark_module, "create_progress_reporter", lambda _mode: reporter
    )
    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda _args, _scenario: {
            "scenario": "single",
            "metrics": {
                "total_requests": 1,
                "successful": 1,
                "failed": 0,
                "failure_rate": 0.0,
            },
        },
    )
    cli_args = build_parser().parse_args([
        "--config", str(config_path), "--report", str(report_path), "--no-resume",
    ])

    assert benchmark_module.run_configured_suite(str(config_path), cli_args) == 0
    assert reporter.events == [
        ("started", "smoke", "single", 1, 1),
        ("finished", "passed", 1, 1),
        ("closed",),
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["cases"][0]["status"] == "passed"
