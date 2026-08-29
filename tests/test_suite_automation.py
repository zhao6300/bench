"""Unit tests for single-host suite automation controls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmark import benchmark as benchmark_module
from benchmark.benchmark import BenchmarkConfigError, build_parser, load_suite_config
from benchmark.suite_automation import AutomationConfigError, parse_automation


class _PreflightResponse:
    """Minimal successful ``requests`` response for preflight tests."""

    status_code = 200

    def json(self) -> dict[str, object]:
        return {"data": [{"id": "placeholder"}]}


def _automation_config(
    *,
    mode: str = "offline",
    cases: list[dict[str, object]] | None = None,
    budget: dict[str, int] | None = None,
    max_duration: float | None = None,
) -> dict[str, object]:
    """Return a small automation-enabled suite with a unique report template."""
    automation: dict[str, object] = {
        "enabled": True,
        "budget": budget or {
            "max_total_requests": 100,
            "max_estimated_output_tokens": 10_000,
        },
        "api_preflight": {"enabled": True, "timeout_seconds": 1},
    }
    if max_duration is not None:
        automation["max_total_duration_seconds"] = max_duration
    defaults: dict[str, object] = {"mode": mode, "model": "placeholder"}
    if mode == "api":
        defaults["api_base"] = "http://localhost:8000/v1"
    return {
        "version": 1,
        "name": "automation-suite",
        "defaults": defaults,
        "automation": automation,
        "report": {"path": "reports/automation-{timestamp}.json"},
        "cases": cases or [{"name": "smoke", "params": {"num_prompts": 2}}],
    }


def _run_suite(monkeypatch, tmp_path, config: dict[str, object]) -> tuple[int, dict[str, object]]:
    """Run a temporary suite and return its exit status and generated report."""
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args(["--config", str(config_path)])

    exit_code = benchmark_module.run_configured_suite(str(config_path), arguments)
    reports = list((tmp_path / "reports").glob("automation-*.json"))
    assert len(reports) == 1
    return exit_code, json.loads(reports[0].read_text(encoding="utf-8"))


def _successful_result() -> dict[str, object]:
    """Return the minimum successful single-scenario result."""
    return {
        "scenario": "single",
        "metrics": {
            "total_requests": 1,
            "successful": 1,
            "failed": 0,
            "failure_rate": 0.0,
        },
    }


def test_automation_schema_requires_complete_budget_and_known_fields() -> None:
    """Reject unbounded and misspelled automation policy settings locally."""
    with pytest.raises(AutomationConfigError, match="budget is required"):
        parse_automation({"enabled": True})
    with pytest.raises(AutomationConfigError, match="unknown fields"):
        parse_automation({
            "enabled": False,
            "unexpected": True,
        })


def test_automation_requires_timestamped_config_and_cli_report_targets(tmp_path) -> None:
    """Avoid overwrite-prone automation reports before any execution starts."""
    config = _automation_config()
    config["report"] = {"path": "reports/fixed.json"}
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args(["--config", str(config_path)])

    with pytest.raises(BenchmarkConfigError, match=r"contain \{timestamp\}"):
        benchmark_module.run_configured_suite(str(config_path), arguments)

    config["report"] = {"path": "reports/valid-{timestamp}.json"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args([
        "--config", str(config_path), "--report", str(tmp_path / "fixed.json"),
    ])
    with pytest.raises(BenchmarkConfigError, match=r"contain \{timestamp\}"):
        benchmark_module.run_configured_suite(str(config_path), arguments)


def test_automation_rejects_budget_without_benchmark_traffic(monkeypatch, tmp_path) -> None:
    """Persist a rejected governance record without calling the executor."""
    executed: list[object] = []
    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda *_args: executed.append(object()),
    )
    config = _automation_config(
        mode="api",
        budget={
            "max_total_requests": 3,
            "max_estimated_output_tokens": 11,
        },
    )

    exit_code, report = _run_suite(monkeypatch, tmp_path, config)

    assert exit_code == 2
    assert executed == []
    assert report["run"]["state"] == "rejected"
    assert report["run"]["automation"]["budget"]["decision"] == "rejected"
    assert report["summary"]["skipped"] == 1
    assert report["provenance"]["execution_plan_sha256"]
    assert "api_key" not in json.dumps(report["run"])


def test_automation_preflight_success_precedes_api_execution(monkeypatch, tmp_path) -> None:
    """Run the authenticated models check before the first API benchmark call."""
    events: list[str] = []

    def get(url: str, **kwargs: object) -> _PreflightResponse:
        events.append("preflight")
        assert url == "http://localhost:8000/v1/models"
        assert kwargs["headers"] == {
            "Content-Type": "application/json",
            "Authorization": "Bearer top-secret",
        }
        assert kwargs["timeout"] == 1.0
        return _PreflightResponse()

    def execute(_args, _scenario: str) -> dict[str, object]:
        events.append("execute")
        return _successful_result()

    monkeypatch.setattr(benchmark_module, "requests", SimpleNamespace(get=get))
    monkeypatch.setattr(benchmark_module, "execute_benchmark", execute)

    config = _automation_config(mode="api")
    defaults = config["defaults"]
    assert isinstance(defaults, dict)
    defaults["api_key"] = "top-secret"
    exit_code, report = _run_suite(monkeypatch, tmp_path, config)

    assert exit_code == 0
    assert events == ["preflight", "execute"]
    assert report["run"]["state"] == "completed"
    assert "top-secret" not in json.dumps(report)
    assert report["run"]["automation"]["preflight"] == [{
        "api_base": "http://localhost:8000/v1",
        "outcome": "passed",
        "status_code": 200,
        "model_listed": True,
    }]


def test_automation_preflight_failure_blocks_execution(monkeypatch, tmp_path) -> None:
    """Stop before benchmark traffic when the models preflight cannot succeed."""
    executed: list[object] = []

    def get(_url: str, **_kwargs: object) -> object:
        raise OSError("service unavailable")

    monkeypatch.setattr(benchmark_module, "requests", SimpleNamespace(get=get))
    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda *_args: executed.append(object()),
    )

    exit_code, report = _run_suite(monkeypatch, tmp_path, _automation_config(mode="api"))

    assert exit_code == 2
    assert executed == []
    assert report["run"]["state"] == "preflight_failed"
    assert report["run"]["automation"]["preflight"][0]["error_type"] == "OSError"
    assert report["cases"][0]["skip_reason"] == "automation API preflight failed"


def test_automation_validation_and_listing_never_call_preflight(monkeypatch, tmp_path) -> None:
    """Keep local inspection modes free from network operations."""
    config = _automation_config(mode="api")
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def get(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preflight must not run")

    monkeypatch.setattr(benchmark_module, "requests", SimpleNamespace(get=get))
    for option in ("--validate-config", "--list-cases"):
        arguments = build_parser().parse_args(["--config", str(config_path), option])
        assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0


def test_automation_offline_suite_never_preflights(monkeypatch, tmp_path) -> None:
    """Do not import or call HTTP checks when all runnable cases are offline."""
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline suite must not preflight")
        )),
    )
    monkeypatch.setattr(
        benchmark_module, "execute_benchmark", lambda *_args: _successful_result()
    )

    exit_code, report = _run_suite(monkeypatch, tmp_path, _automation_config())

    assert exit_code == 0
    assert report["run"]["automation"]["preflight"] == []


def test_automation_deadline_skips_later_cases_at_case_boundary(monkeypatch, tmp_path) -> None:
    """Allow the running case to finish, then cooperatively skip remaining work."""
    executed: list[str] = []
    ticks = iter([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(benchmark_module.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda _args, _scenario: (executed.append("run") or _successful_result()),
    )
    config = _automation_config(
        cases=[{"name": "first"}, {"name": "second"}],
        max_duration=1.0,
    )

    exit_code, report = _run_suite(monkeypatch, tmp_path, config)

    assert exit_code == 124
    assert executed == ["run"]
    assert [case["status"] for case in report["cases"]] == ["passed", "skipped"]
    assert report["cases"][1]["skip_reason"] == "automation deadline exceeded"
    assert report["run"]["state"] == "timed_out"
    assert report["run"]["automation"]["deadline"]["exceeded_at"] is not None


def test_automation_can_disable_api_preflight(monkeypatch, tmp_path) -> None:
    """Respect an explicit preflight opt-out without suppressing API execution."""
    config = _automation_config(mode="api")
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["api_preflight"] = {"enabled": False, "timeout_seconds": 1}
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled preflight must not run")
        )),
    )
    monkeypatch.setattr(
        benchmark_module, "execute_benchmark", lambda *_args: _successful_result()
    )

    exit_code, report = _run_suite(monkeypatch, tmp_path, config)

    assert exit_code == 0
    assert report["run"]["automation"]["preflight"] == []
