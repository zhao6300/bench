"""Unit tests for terminal progress reporters."""

from __future__ import annotations

import io
import json

import pytest

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
    assert "加载 tokenizer" not in output
    assert "跳过预热请求" not in output


def test_plain_progress_ignores_stage_and_information_events() -> None:
    """Keep portable plain output unchanged when dashboard-only events are emitted."""
    stream = _TerminalStream(False)
    reporter = PlainProgressReporter(stream)

    reporter.stage_started("加载 tokenizer", "placeholder")
    reporter.stage_finished("加载 tokenizer")
    reporter.event("跳过预热请求")
    reporter.show_final_results({"summary": {}, "cases": []})

    assert stream.getvalue() == ""


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
    reporter.show_final_results({"summary": {}, "cases": []})
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


class _FinalSuiteReporter(_SuiteReporter):
    """Capture final-result callbacks in addition to suite lifecycle events."""

    def show_final_results(
        self, report: dict[str, object], report_location: str | None
    ) -> None:
        summary = report["summary"]
        assert isinstance(summary, dict)
        self.events.append(("final", summary["passed"], report_location))


def test_optional_stage_events_support_legacy_reporters() -> None:
    """Do not require suite reporters to implement dashboard-only callbacks."""
    reporter = _SuiteReporter()
    args = type("Args", (), {"_progress_reporter": reporter})()

    benchmark_module._emit_progress_event(args, "stage_started", "加载 tokenizer")
    benchmark_module._emit_progress_event(args, "stage_finished", "加载 tokenizer")
    benchmark_module._emit_progress_event(args, "event", "跳过预热请求")

    assert reporter.events == []


def test_optional_final_results_support_legacy_reporters() -> None:
    """Preserve suite execution for reporters without a final-result hook."""
    reporter = _SuiteReporter()

    benchmark_module._show_final_progress_results(reporter, {"cases": []}, None)

    assert reporter.events == []


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
    reporter = _FinalSuiteReporter()
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
        ("final", 1, str(report_path)),
        ("closed",),
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["cases"][0]["status"] == "passed"


class _FakeConsole:
    """Capture console configuration without importing Rich."""

    width = 120

    def __init__(self, **kwargs: object) -> None:
        from types import SimpleNamespace

        self.kwargs = kwargs
        self.size = SimpleNamespace(width=self.width)


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
    """Record dashboard rows and columns without Rich rendering behavior."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.columns: list[tuple[object, ...]] = []
        self.rows: list[tuple[object, ...]] = []

    @classmethod
    def grid(cls, **_kwargs: object) -> "_FakeTable":
        return cls()

    def add_column(self, *args: object, **_kwargs: object) -> None:
        self.columns.append(args)

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

    reporter.stage_started("加载 tokenizer", "placeholder")
    reporter.stage_finished("加载 tokenizer")
    reporter.event("跳过预热请求")
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
    assert any("阶段开始: 加载 tokenizer：placeholder" in row[0] for row in dashboard["events"].content.content.rows)
    assert any("阶段完成: 加载 tokenizer" in row[0] for row in dashboard["events"].content.content.rows)
    assert any("跳过预热请求" in row[0] for row in dashboard["events"].content.content.rows)

    final_report = {
        "summary": {"passed": 1, "failed": 1, "interrupted": 0, "skipped": 0},
        "cases": [
            {
                "name": "smoke",
                "scenario": "single",
                "status": "passed",
                "result": {
                    "metrics": {
                        "total_requests": 4,
                        "successful": 4,
                        "failed": 0,
                        "avg_ttft": 0.2,
                        "overall_throughput": 100.0,
                        "qps": 2.5,
                    },
                },
            },
            {
                "name": "failed-case",
                "scenario": "single",
                "status": "failed",
                "error": {"message": "quality gate failed"},
            },
        ],
    }
    monkeypatch.setattr(reporter, "_read_final_key", lambda: "q")
    reporter.show_final_results(final_report, "/tmp/final-report.json")

    final_dashboard = live.updates[-1]
    overview_table = final_dashboard["overview"].content.content
    details_table = final_dashboard["details"].content.content
    assert [column[0] for column in overview_table.columns] == [
        "用例 / 场景", "状态", "负载", "核心指标", "结论",
    ]
    assert any(
        "smoke" in row[0] and "整体吞吐 100.0 个 token/秒" in row[3]
        for row in overview_table.rows
    )
    assert any(
        "failed-case" in row[0] and "quality gate failed" in row[4]
        for row in overview_table.rows
    )
    assert any(
        row[0] == "延迟" and "TTFT：平均 200.0 毫秒" in row[1]
        for row in details_table.rows
    )
    assert "↑/↓ 或 j/k" in final_dashboard["footer"].content.content
    monkeypatch.setattr(
        reporter,
        "_read_final_key",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    reporter.show_final_results(final_report)

    reporter.close()
    assert live.stopped is True


def test_rich_final_results_adapt_to_terminal_width(monkeypatch) -> None:
    """Show all available aggregates without making narrow result tables unreadable."""
    from benchmark import progress as progress_module

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
    report = {
        "summary": {"passed": 4, "failed": 1, "interrupted": 0, "skipped": 0},
        "cases": [
            {
                "name": "api",
                "scenario": "single",
                "status": "passed",
                "result": {
                    "metrics": {
                        "concurrency": 4,
                        "total_requests": 8,
                        "successful": 7,
                        "failed": 1,
                        "failure_rate": 0.125,
                        "wall_time": 2.0,
                        "avg_ttft": 0.2,
                        "p50_ttft": 0.15,
                        "p90_ttft": 0.3,
                        "p99_ttft": 0.5,
                        "avg_tpot": 0.04,
                        "p50_tpot": 0.03,
                        "p90_tpot": 0.06,
                        "p99_tpot": 0.08,
                        "avg_total_time": 1.2,
                        "p50_e2e": 1.0,
                        "p90_e2e": 1.5,
                        "p99_e2e": 2.0,
                        "prompt_throughput": 300.0,
                        "prefill_throughput": 250.0,
                        "decode_throughput": 100.0,
                        "overall_throughput": 350.0,
                        "qps": 3.5,
                        "goodput_pct": 87.5,
                        "goodput_qps": 3.0,
                        "slo_ttft": 0.3,
                        "slo_tpot": 0.05,
                        "total_prompt_tokens": 32000,
                        "total_generated_tokens": 16000,
                        "server_metrics": {
                            "metrics": {
                                "cache_hit_rate": {"avg": 70.0},
                                "gpu_cache_usage_pct": {"avg": 85.0},
                                "cpu_cache_usage_pct": {"max": 5.0},
                                "running_requests": {"avg": 3.5},
                                "waiting_requests": {"max": 1.0},
                            },
                        },
                    },
                    "workload": {
                        "prompt_tokens": {"avg": 4000},
                        "requested_output_tokens": {"avg": 2000},
                        "shared_prefix_tokens": 1000,
                    },
                },
            },
            {
                "name": "offline",
                "scenario": "offline",
                "status": "passed",
                "result": {
                    "metrics": {
                        "concurrency": 2,
                        "total_requests": 4,
                        "total_prompt_tokens": 8000,
                        "total_generated_tokens": 2000,
                        "overall_throughput": 100.0,
                    },
                    "workload": {
                        "prompt_tokens": {"avg": 2000},
                        "requested_output_tokens": {"avg": 500},
                        "shared_prefix_tokens": 0,
                    },
                },
            },
            {
                "name": "sweep",
                "scenario": "sweep",
                "status": "passed",
                "result": {
                    "metric": "prefill_throughput",
                    "best_concurrency": 16,
                    "best_throughput": 1200.0,
                },
            },
            {
                "name": "slo",
                "scenario": "slo-capacity-search",
                "status": "passed",
                "result": {
                    "max_passing_concurrency": 8,
                    "confirmed_concurrency": 8,
                    "refined_failure_boundary": 9,
                    "required_goodput_pct": 95.0,
                    "max_failure_rate": 0.05,
                    "selected_metrics": {
                        "concurrency": 8,
                        "total_requests": 16,
                        "successful": 16,
                        "failed": 0,
                        "failure_rate": 0.0,
                        "goodput_pct": 100.0,
                    },
                },
            },
            {
                "name": "pd",
                "scenario": "pd-ratio",
                "status": "failed",
                "result": {
                    "prefill": {"prefill_throughput": 600.0},
                    "decode": {"decode_throughput": 300.0},
                    "analysis": {
                        "recommended": True,
                        "recommended_ratio": {"prefill": 2, "decode": 3},
                        "prefill_share_pct": 40.0,
                        "decode_share_pct": 60.0,
                    },
                },
                "quality_failures": ["需要复核部署 SLO"],
            },
            {
                "name": "slo-unavailable",
                "scenario": "slo-capacity-search",
                "status": "failed",
                "result": {},
            },
        ],
    }
    reporter._final_report = report

    reporter._console.size.width = 80
    narrow_dashboard = reporter._render_final_results()
    narrow_table = narrow_dashboard["overview"].content.content
    narrow_details = narrow_dashboard["details"].content.content
    assert [column[0] for column in narrow_table.columns] == [
        "用例 / 状态", "核心指标", "结论",
    ]
    assert len(narrow_table.rows) == 2
    assert "整体吞吐 350.0 个 token/秒" in narrow_table.rows[0][1]
    assert any(
        row[0] == "请求 / 负载"
        and "输入 4,000 个 token" in row[1]
        and "实际输入 32,000 个 token" in row[1]
        for row in narrow_details.rows
    )
    assert any(
        row[0] == "延迟"
        and "TTFT：平均 200.0 毫秒 / P50 150.0 毫秒" in row[1]
        and "P90 300.0 毫秒" in row[1]
        and "P99 500.0 毫秒" in row[1]
        and "TPOT：平均 40.0 毫秒" in row[1]
        and "E2E：平均 1200.0 毫秒" in row[1]
        for row in narrow_details.rows
    )
    assert any(
        row[0] == "性能"
        and "整体吞吐 350.0 个 token/秒" in row[1]
        and "每秒完成请求数 3.50 个请求/秒" in row[1]
        and "达标率 87.5%" in row[1]
        for row in narrow_details.rows
    )
    assert any(
        row[0] == "服务端"
        and "KV Cache 命中率 70.0%" in row[1]
        and "GPU Cache 使用率 85.0%" in row[1]
        and "CPU Cache 使用率 5.0%" in row[1]
        for row in narrow_details.rows
    )

    reporter._console.size.width = 120
    medium_dashboard = reporter._render_final_results()
    medium_table = medium_dashboard["overview"].content.content
    assert [column[0] for column in medium_table.columns] == [
        "用例 / 场景", "状态", "负载", "核心指标", "结论",
    ]
    assert len(medium_table.rows) == 3
    assert "并发 4 个请求 · 总请求数 8 个请求" in medium_table.rows[0][2]
    assert "TTFT 平均 200.0 毫秒" in medium_table.rows[0][3]
    assert "500.0 毫秒" not in medium_table.rows[0][3]
    assert "最佳并发 16 个请求" in medium_table.rows[2][4]

    reporter._console.size.width = 180
    wide_dashboard = reporter._render_final_results()
    wide_table = wide_dashboard["overview"].content.content
    assert [column[0] for column in wide_table.columns] == [
        "用例 / 场景", "状态", "负载", "核心指标", "场景结果", "说明",
    ]
    assert len(wide_table.rows) == 4
    assert "最大通过并发 8 个请求" in wide_table.rows[3][4]
    assert "负载（" not in "".join(column[0] for column in wide_table.columns)
    assert "核心指标（" not in "".join(column[0] for column in wide_table.columns)

    reporter._move_final_selection(4)
    selected_dashboard = reporter._render_final_results()
    selected_table = selected_dashboard["overview"].content.content
    selected_details = selected_dashboard["details"].content.content
    assert "▶ pd" in selected_table.rows[0][0]
    assert any(
        row[0] == "性能" and "预填充吞吐 600.0 个 token/秒" in row[1]
        for row in selected_details.rows
    )


def test_rich_final_scenario_labels_are_user_facing() -> None:
    """Use clear Chinese display names while preserving unknown scenario keys."""
    from benchmark import progress as progress_module

    labels = {
        "single": "单一负载接口",
        "offline": "离线引擎",
        "mixed-workload": "混合负载接口",
        "sweep": "吞吐扫描",
        "slo-capacity-search": "服务等级目标容量搜索",
        "pd-ratio": "预填充/解码容量评估",
    }

    assert {
        scenario: progress_module.RichProgressReporter._final_scenario_label(scenario)
        for scenario in labels
    } == labels
    assert progress_module.RichProgressReporter._final_scenario_label("future-scenario") == (
        "future-scenario"
    )


def _matrix_suite_config(
    *,
    continue_on_error: bool = True,
    first_concurrencies: list[int] | None = None,
    second_concurrencies: list[int] | None = None,
) -> dict[str, object]:
    """Build a minimal no-network suite with two independent matrix groups."""
    return {
        "version": 1,
        "name": "matrix-failure-policy",
        "failure_policy": "stop-current-matrix",
        "continue_on_error": continue_on_error,
        "defaults": {
            "mode": "offline",
            "model": "placeholder",
            "max_failure_rate": 0.0,
        },
        "cases": [
            {
                "name": "first-group",
                "matrix": {"concurrency": first_concurrencies or [1, 2, 3]},
            },
            {
                "name": "second-group",
                "matrix": {"concurrency": second_concurrencies or [10, 20]},
            },
        ],
    }


def _run_matrix_suite(
    monkeypatch,
    tmp_path,
    config: dict[str, object],
    execute,
    *,
    fail_fast: bool = False,
) -> tuple[int, dict[str, object]]:
    """Run a mocked matrix suite and return its status code and report."""
    config_path = tmp_path / "suite.json"
    report_path = tmp_path / "report.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(benchmark_module, "execute_benchmark", execute)
    arguments = [
        "--config", str(config_path), "--report", str(report_path), "--no-resume",
    ]
    if fail_fast:
        arguments.append("--fail-fast")
    cli_args = build_parser().parse_args(arguments)
    exit_code = benchmark_module.run_configured_suite(str(config_path), cli_args)
    return exit_code, json.loads(report_path.read_text(encoding="utf-8"))


def _single_result(failure_rate: float) -> dict[str, object]:
    """Return the smallest result shape accepted by quality-gate validation."""
    return {
        "scenario": "single",
        "metrics": {
            "total_requests": 1,
            "successful": int(failure_rate == 0.0),
            "failed": int(failure_rate > 0.0),
            "failure_rate": failure_rate,
        },
    }


def test_stop_current_matrix_skips_remaining_variants_after_quality_failure(
    monkeypatch, tmp_path
) -> None:
    """Skip only later variants in a matrix group after a quality-gate failure."""
    executed: list[int] = []

    def execute(args, _scenario: str) -> dict[str, object]:
        executed.append(args.concurrency)
        return _single_result(1.0 if args.concurrency == 1 else 0.0)

    exit_code, report = _run_matrix_suite(
        monkeypatch, tmp_path, _matrix_suite_config(), execute
    )

    assert exit_code == 1
    assert executed == [1, 10, 20]
    assert [case["status"] for case in report["cases"]] == [
        "failed", "skipped", "skipped", "passed", "passed",
    ]
    assert report["suite"]["failure_policy"] == "stop-current-matrix"
    assert report["cases"][1]["skip_reason"] == (
        "previous matrix variant failed: first-group[concurrency=1]"
    )
    assert report["summary"]["skipped"] == 2


def test_stop_current_matrix_allows_next_group_after_final_variant_failure(
    monkeypatch, tmp_path
) -> None:
    """Do not block the next group when the failed variant is already last."""
    executed: list[int] = []

    def execute(args, _scenario: str) -> dict[str, object]:
        executed.append(args.concurrency)
        return _single_result(1.0 if args.concurrency == 2 else 0.0)

    config = _matrix_suite_config(
        first_concurrencies=[1, 2], second_concurrencies=[10]
    )
    exit_code, report = _run_matrix_suite(monkeypatch, tmp_path, config, execute)

    assert exit_code == 1
    assert executed == [1, 2, 10]
    assert [case["status"] for case in report["cases"]] == [
        "passed", "failed", "passed",
    ]
    assert report["summary"]["skipped"] == 0


def test_stop_current_matrix_skips_variants_after_execution_exception(
    monkeypatch, tmp_path
) -> None:
    """Treat executor exceptions as group-stopping failures without stopping later groups."""
    executed: list[int] = []

    def execute(args, _scenario: str) -> dict[str, object]:
        executed.append(args.concurrency)
        if args.concurrency == 1:
            raise RuntimeError("simulated executor failure")
        return _single_result(0.0)

    config = _matrix_suite_config(
        first_concurrencies=[1, 2], second_concurrencies=[10]
    )
    exit_code, report = _run_matrix_suite(monkeypatch, tmp_path, config, execute)

    assert exit_code == 1
    assert executed == [1, 10]
    assert [case["status"] for case in report["cases"]] == [
        "failed", "skipped", "passed",
    ]
    assert report["cases"][0]["error"]["type"] == "RuntimeError"


@pytest.mark.parametrize(
    ("continue_on_error", "fail_fast"),
    [(False, False), (True, True)],
)
def test_global_failure_controls_override_stop_current_matrix(
    monkeypatch, tmp_path, continue_on_error: bool, fail_fast: bool
) -> None:
    """Preserve global stopping for continue_on_error=false and --fail-fast."""
    executed: list[int] = []

    def execute(args, _scenario: str) -> dict[str, object]:
        executed.append(args.concurrency)
        return _single_result(1.0)

    config = _matrix_suite_config(
        continue_on_error=continue_on_error,
        first_concurrencies=[1, 2],
        second_concurrencies=[10],
    )
    exit_code, report = _run_matrix_suite(
        monkeypatch, tmp_path, config, execute, fail_fast=fail_fast
    )

    assert exit_code == 1
    assert executed == [1]
    assert [case["status"] for case in report["cases"]] == [
        "failed", "pending", "pending",
    ]
    assert report["suite"]["run_state"] == "stopped"
