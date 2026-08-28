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


class _FakeConsole:
    """Capture console configuration without importing Rich."""

    def __init__(self, **kwargs: object) -> None:
        from types import SimpleNamespace

        self.kwargs = kwargs
        self.size = SimpleNamespace(width=120)


class _FakeLayout:
    """Minimal Rich Layout replacement for dashboard unit tests."""

    def __init__(self, name: str | None = None, **_kwargs: object) -> None:
        self.name = name
        self.children: list[_FakeLayout] = []
        self.content: object | None = None

    def split_column(self, *children: "_FakeLayout") -> None:
        self.children = list(children)

    def split_row(self, *children: "_FakeLayout") -> None:
        self.children = list(children)

    def __getitem__(self, name: str) -> "_FakeLayout":
        if self.name == name:
            return self
        for child in self.children:
            try:
                return child[name]
            except KeyError:
                continue
        raise KeyError(name)

    def update(self, content: object) -> None:
        self.content = content


class _FakeTable:
    """Record dashboard rows without Rich rendering behavior."""

    def __init__(self) -> None:
        self.rows: list[tuple[object, ...]] = []

    @classmethod
    def grid(cls, **_kwargs: object) -> "_FakeTable":
        return cls()

    def add_column(self, **_kwargs: object) -> None:
        pass

    def add_row(self, *values: object) -> None:
        self.rows.append(values)


class _FakePanel:
    """Store panel content and title for dashboard assertions."""

    def __init__(self, content: object, **kwargs: object) -> None:
        self.content = content
        self.kwargs = kwargs


class _FakeText(str):
    """Accept Rich Text keyword arguments in dashboard tests."""

    def __new__(cls, value: str, **_kwargs: object) -> "_FakeText":
        return super().__new__(cls, value)


class _FakeLive:
    """Capture alternate-screen lifecycle configuration."""

    instances: list["_FakeLive"] = []

    def __init__(self, renderable: object, **kwargs: object) -> None:
        self.renderable = renderable
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.updates: list[object] = []
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def update(self, renderable: object, **_kwargs: object) -> None:
        self.updates.append(renderable)


def test_rich_progress_uses_full_screen_dashboard(monkeypatch) -> None:
    """Use an alternate-screen dashboard with suite, round, and event panels."""
    from benchmark import progress as progress_module

    _FakeLive.instances.clear()
    monkeypatch.setattr(
        progress_module,
        "_load_rich_dashboard_components",
        lambda: {
            "Console": _FakeConsole,
            "Layout": _FakeLayout,
            "Live": _FakeLive,
            "Panel": _FakePanel,
            "Table": _FakeTable,
            "Text": _FakeText,
        },
    )
    reporter = progress_module.RichProgressReporter(_TerminalStream(True))

    reporter.case_started("smoke", "single", 1, 2)
    reporter.round_started(4, 2)
    reporter.request_finished({
        "completed": 1,
        "total": 4,
        "succeeded": 0,
        "failed": 1,
        "elapsed_seconds": 0.2,
        "last_ttft": None,
        "request_id": 0,
        "error": "timeout",
    })

    live = _FakeLive.instances[0]
    assert live.started is True
    assert live.kwargs["screen"] is True
    assert live.kwargs["transient"] is True
    dashboard = live.updates[-1]
    assert isinstance(dashboard["suite"].content, _FakePanel)
    assert isinstance(dashboard["round"].content, _FakePanel)
    assert isinstance(dashboard["events"].content, _FakePanel)
    assert isinstance(dashboard["footer"].content, _FakePanel)
    assert any("请求失败" in row[0] for row in dashboard["events"].content.content.rows)

    reporter.close()
    assert live.stopped is True
