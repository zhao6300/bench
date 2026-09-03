"""Unit tests for single-host suite automation controls."""

from __future__ import annotations

import json
from pathlib import Path
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


def test_automation_schema_allows_unbounded_budget_and_rejects_unknown_fields() -> None:
    """Treat an omitted budget as unlimited while retaining strict schema validation."""
    assert parse_automation({"enabled": True}).budget is None
    assert parse_automation({"enabled": True, "budget": None}).budget is None
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


def _add_compose_service_bundle(tmp_path, config: dict[str, object]) -> None:
    """Attach a test-local enabled Compose service bundle to automation config."""
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    (deploy / "base.yml").write_text("services: {}\n", encoding="utf-8")
    (deploy / "tp4.env").write_text("GPU_DEVICES=0,1,2,3\n", encoding="utf-8")
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["docker_service"] = {
        "enabled": True,
        "profile_name": "tp4-dp1",
        "engine": "vllm",
        "compose_files": ["deploy/base.yml"],
        "env_file": "deploy/tp4.env",
        "project_name": "benchmark-tp4-dp1",
        "start_timeout_seconds": 10,
        "ready_timeout_seconds": 10,
        "poll_interval_seconds": 1,
        "probe_timeout_seconds": 1,
    }


def test_compose_service_validation_is_opt_in_and_requires_complete_profile() -> None:
    """Keep legacy automation unchanged while rejecting incomplete managed services."""
    assert parse_automation({"enabled": False}).docker_service.enabled is False
    with pytest.raises(AutomationConfigError, match="profile_name"):
        parse_automation({
            "enabled": True,
            "budget": {
                "max_total_requests": 1,
                "max_estimated_output_tokens": 1,
            },
            "docker_service": {"enabled": True},
        })


def test_managed_compose_service_runs_before_execution_and_stops_afterwards(
    monkeypatch, tmp_path
) -> None:
    """Run one local service lifecycle around readiness and benchmark execution."""
    events: list[str] = []

    class _ManagedService:
        def __init__(self, _service) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    def get(_url: str, **kwargs: object) -> _PreflightResponse:
        events.append("ready")
        assert kwargs["timeout"] == 1.0
        return _PreflightResponse()

    def execute(_args, _scenario: str) -> dict[str, object]:
        events.append("execute")
        return _successful_result()

    config = _automation_config(mode="api")
    _add_compose_service_bundle(tmp_path, config)
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(benchmark_module, "requests", SimpleNamespace(get=get))
    monkeypatch.setattr(benchmark_module, "execute_benchmark", execute)

    exit_code, report = _run_suite(monkeypatch, tmp_path, config)

    assert exit_code == 0
    assert events == ["start", "ready", "execute", "stop"]
    service_report = report["run"]["automation"]["docker_service"]
    assert service_report["enabled"] is True
    assert service_report["state"] == "stopped"
    assert service_report["profile_name"] == "tp4-dp1"
    assert service_report["engine"] == "vllm"
    assert service_report["project_name"] == "benchmark-tp4-dp1"
    assert service_report["readiness_attempts"] == 1
    assert service_report["error"] is None
    assert service_report["started_at"] is not None
    assert service_report["ready_at"] is not None
    assert service_report["stopped_at"] is not None


def test_compose_service_is_not_started_for_suite_inspection(monkeypatch, tmp_path) -> None:
    """Keep validation and list operations free from Docker and HTTP side effects."""
    config = _automation_config(mode="api")
    _add_compose_service_bundle(tmp_path, config)
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(
        benchmark_module,
        "ManagedComposeService",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Docker must not start")),
    )
    for option in ("--validate-config", "--list-cases"):
        arguments = build_parser().parse_args(["--config", str(config_path), option])
        assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0


def test_managed_compose_service_stops_when_benchmark_is_interrupted(
    monkeypatch, tmp_path
) -> None:
    """Release a locally started inference service when the user interrupts a case."""
    events: list[str] = []

    class _ManagedService:
        def __init__(self, _service) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    config = _automation_config(mode="api")
    _add_compose_service_bundle(tmp_path, config)
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args(["--config", str(config_path)])

    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: _PreflightResponse()),
    )
    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        benchmark_module.run_configured_suite(str(config_path), arguments)

    assert events == ["start", "stop"]
    report = json.loads(next((tmp_path / "reports").glob("automation-*.json")).read_text())
    assert report["run"]["automation"]["docker_service"]["state"] == "stopped"
    assert report["suite"]["run_state"] == "interrupted"


def test_managed_compose_teardown_failure_is_a_terminal_run_failure(
    monkeypatch, tmp_path
) -> None:
    """Expose failed container cleanup instead of reporting a completed suite."""
    events: list[str] = []

    class _ManagedService:
        def __init__(self, _service) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")
            raise benchmark_module.ComposeServiceError("down failed")

    config = _automation_config(mode="api")
    _add_compose_service_bundle(tmp_path, config)
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: _PreflightResponse()),
    )
    monkeypatch.setattr(
        benchmark_module, "execute_benchmark", lambda *_args: _successful_result()
    )

    exit_code, report = _run_suite(monkeypatch, tmp_path, config)

    assert exit_code == 1
    assert events == ["start", "stop"]
    assert report["run"]["state"] == "teardown_failed"
    assert report["run"]["workload_state"] == "completed"
    assert report["suite"]["run_state"] == "teardown_failed"
    assert report["run"]["automation"]["docker_service"]["state"] == "teardown_failed"


def test_managed_compose_readiness_failure_preserves_probe_records(
    monkeypatch, tmp_path
) -> None:
    """Store the final sanitized models probe record when readiness times out."""
    events: list[str] = []

    class _ManagedService:
        def __init__(self, _service) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    config = _automation_config(mode="api")
    _add_compose_service_bundle(tmp_path, config)
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "wait_for_api_targets",
        lambda *_args, **_kwargs: ([{
            "api_base": "http://localhost:8000/v1",
            "outcome": "failed",
            "error_type": "OSError",
            "message": "service readiness timed out",
        }], 3),
    )

    exit_code, report = _run_suite(monkeypatch, tmp_path, config)

    assert exit_code == 2
    assert events == ["start", "stop"]
    assert report["run"]["state"] == "service_readiness_failed"
    assert report["run"]["automation"]["docker_service"]["readiness_attempts"] == 3
    assert report["run"]["automation"]["docker_service"]["readiness"] == [{
        "api_base": "http://localhost:8000/v1",
        "outcome": "failed",
        "error_type": "OSError",
        "message": "service readiness timed out",
    }]


def _add_compose_service_profiles(
    tmp_path, config: dict[str, object], profiles: list[dict[str, object]]
) -> None:
    """Attach a shared bundle and serial Compose profiles to a test suite."""
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    (deploy / "base.yml").write_text("services: {}\n", encoding="utf-8")
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["docker_service"] = {
        "enabled": True,
        "engine": "vllm",
        "compose_files": ["deploy/base.yml"],
        "base_environment": {
            "ENGINE_IMAGE": "registry.example/engine:stable",
            "TP_SIZE": "1",
        },
        "profiles": profiles,
        "start_timeout_seconds": 10,
        "ready_timeout_seconds": 10,
        "poll_interval_seconds": 1,
        "probe_timeout_seconds": 1,
    }


def test_compose_profiles_merge_environment_with_profile_precedence() -> None:
    """Keep shared substitutions while allowing a topology to override one value."""
    config = _automation_config(mode="api")
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["docker_service"] = {
        "enabled": True,
        "engine": "vllm",
        "compose_files": ["deploy/base.yml"],
        "base_environment": {"ENGINE_IMAGE": "base", "TP_SIZE": "1"},
        "profiles": [{
            "name": "tp4-dp2",
            "project_name": "benchmark-tp4-dp2",
            "environment": {"TP_SIZE": "4", "DP_SIZE": "2"},
        }],
    }

    policy = parse_automation(automation).docker_service
    selected = policy.select_profile(policy.profiles[0])

    assert dict(selected.environment) == {
        "ENGINE_IMAGE": "base",
        "TP_SIZE": "4",
        "DP_SIZE": "2",
    }


@pytest.mark.parametrize(
    ("profiles", "base_environment", "message"),
    [
        ([{
            "name": "tp1",
            "project_name": "benchmark-tp1",
            "environment": {"INVALID-KEY": "x"},
        }], {"ENGINE_IMAGE": "base"}, "environment variable names"),
        ([{
            "name": "tp1",
            "project_name": "benchmark-tp1",
            "environment": {"TP_SIZE": 1},
        }], {"ENGINE_IMAGE": "base"}, "must be a string"),
        ([
            {"name": "tp1", "project_name": "benchmark-tp1"},
            {"name": "tp1", "project_name": "benchmark-tp2"},
        ], {"ENGINE_IMAGE": "base"}, "duplicate docker service profile name"),
        ([
            {"name": "tp1", "project_name": "benchmark-shared"},
            {"name": "tp2", "project_name": "benchmark-shared"},
        ], {"ENGINE_IMAGE": "base"}, "duplicate docker service project_name"),
        ([{
            "name": "tp1",
            "project_name": "benchmark-tp1",
            "enabled": "yes",
        }], {"ENGINE_IMAGE": "base"}, "enabled must be true or false"),
    ],
)
def test_compose_profiles_reject_invalid_topology_values(
    profiles: list[dict[str, object]],
    base_environment: dict[str, object],
    message: str,
) -> None:
    """Reject unsafe environment maps and ambiguous profile identifiers locally."""
    config = _automation_config(mode="api")
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["docker_service"] = {
        "enabled": True,
        "engine": "vllm",
        "compose_files": ["deploy/base.yml"],
        "base_environment": base_environment,
        "profiles": profiles,
    }

    with pytest.raises(AutomationConfigError, match=message):
        parse_automation(automation)


def test_compose_profiles_reject_single_profile_fields() -> None:
    """Prevent ambiguous lifecycle ownership when profiles are configured."""
    config = _automation_config(mode="api")
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["docker_service"] = {
        "enabled": True,
        "profile_name": "legacy",
        "engine": "vllm",
        "compose_files": ["deploy/base.yml"],
        "profiles": [{"name": "tp1", "project_name": "benchmark-tp1"}],
    }

    with pytest.raises(AutomationConfigError, match="cannot be combined"):
        parse_automation(automation)


def test_compose_profiles_run_serially_and_report_deployment_parameters(monkeypatch, tmp_path) -> None:
    """Record effective profile parameters while redacting secret-shaped values."""
    events: list[str] = []

    class _ManagedService:
        def __init__(self, service) -> None:
            self._project_name = service.project_name

        def start(self) -> None:
            events.append(f"start:{self._project_name}")

        def stop(self) -> None:
            events.append(f"stop:{self._project_name}")

    config = _automation_config(mode="api")
    _add_compose_service_profiles(tmp_path, config, [
        {
            "name": "tp1-dp1",
            "project_name": "benchmark-tp1-dp1",
            "environment": {
                "TP_SIZE": "1",
                "ATTENTION_BACKEND": "flashinfer",
                "PROFILE_SECRET": "not-reported",
            },
        },
        {
            "name": "tp4-dp2",
            "project_name": "benchmark-tp4-dp2",
            "environment": {"TP_SIZE": "4", "DP_SIZE": "2"},
        },
    ])
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: _PreflightResponse()),
    )
    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda *_args: (events.append("execute") or _successful_result()),
    )

    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args(["--config", str(config_path)])

    assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0
    assert events == [
        "start:benchmark-tp1-dp1", "execute", "stop:benchmark-tp1-dp1",
        "start:benchmark-tp4-dp2", "execute", "stop:benchmark-tp4-dp2",
    ]
    reports = list((tmp_path / "reports").glob("automation-*.json"))
    assert len(reports) == 3
    summary_path = next(path for path in reports if "profiles-summary" in path.name)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["state"] == "completed"
    assert summary["summary"] == {
        "total": 2, "completed": 2, "failed": 0, "timed_out": 0,
        "skipped": 0, "not_run": 0,
    }
    profiles = {item["name"]: item for item in summary["profiles"]}
    assert profiles["tp1-dp1"]["environment"] == {
        "ENGINE_IMAGE": "registry.example/engine:stable",
        "TP_SIZE": "1",
        "ATTENTION_BACKEND": "flashinfer",
        "PROFILE_SECRET": "<redacted>",
    }
    assert profiles["tp4-dp2"]["environment"] == {
        "ENGINE_IMAGE": "registry.example/engine:stable",
        "TP_SIZE": "4",
        "DP_SIZE": "2",
    }
    for profile_name, profile in profiles.items():
        report = json.loads(Path(profile["report"]).read_text(encoding="utf-8"))
        assert report["run"]["automation"]["docker_service"]["environment"] == (
            profile["environment"]
        )


def test_disabled_compose_profiles_skip_service_and_write_aggregate(monkeypatch, tmp_path) -> None:
    """Produce an aggregate for intentionally disabled topologies without traffic."""
    config = _automation_config(mode="api")
    _add_compose_service_profiles(tmp_path, config, [
        {"name": "tp1", "enabled": False, "project_name": "benchmark-tp1"},
        {"name": "tp4", "enabled": False, "project_name": "benchmark-tp4"},
    ])
    monkeypatch.setattr(
        benchmark_module,
        "ManagedComposeService",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Docker must not start")),
    )
    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda *_args: (_ for _ in ()).throw(AssertionError("benchmark must not run")),
    )
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args(["--config", str(config_path)])

    assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0
    summary_path = next((tmp_path / "reports").glob("*profiles-summary*.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["summary"]["skipped"] == 2
    assert all(item["state"] == "skipped" for item in summary["profiles"])


@pytest.mark.parametrize("fail_fast, expected_starts, expected_not_run", [
    (False, ["benchmark-tp1", "benchmark-tp2"], 0),
    (True, ["benchmark-tp1"], 1),
])
def test_compose_profile_fail_fast_controls_later_profiles(
    monkeypatch, tmp_path, fail_fast: bool, expected_starts: list[str], expected_not_run: int
) -> None:
    """Continue failed topologies by default and mark later profiles not run with fail-fast."""
    started: list[str] = []
    current_project = [""]

    class _ManagedService:
        def __init__(self, service) -> None:
            self._project_name = service.project_name

        def start(self) -> None:
            current_project[0] = self._project_name
            started.append(self._project_name)

        def stop(self) -> None:
            pass

    config = _automation_config(mode="api")
    _add_compose_service_profiles(tmp_path, config, [
        {"name": "tp1", "project_name": "benchmark-tp1"},
        {"name": "tp2", "project_name": "benchmark-tp2"},
    ])
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: _PreflightResponse()),
    )

    def execute(*_args) -> dict[str, object]:
        if current_project[0] == "benchmark-tp1":
            raise RuntimeError("first profile failed")
        return _successful_result()

    monkeypatch.setattr(benchmark_module, "execute_benchmark", execute)
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    argv = ["--config", str(config_path)]
    if fail_fast:
        argv.append("--fail-fast")

    assert benchmark_module.run_configured_suite(
        str(config_path), build_parser().parse_args(argv)
    ) == 1
    assert started == expected_starts
    summary_path = next((tmp_path / "reports").glob("*profiles-summary*.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["summary"]["not_run"] == expected_not_run


def test_compose_profiles_validate_and_list_without_starting_docker(monkeypatch, tmp_path) -> None:
    """Retain side-effect-free inspection for profile-based Compose suites."""
    config = _automation_config(mode="api")
    _add_compose_service_profiles(tmp_path, config, [
        {"name": "tp1", "enabled": False, "project_name": "benchmark-tp1"},
        {"name": "tp2", "enabled": False, "project_name": "benchmark-tp2"},
    ])
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        benchmark_module,
        "ManagedComposeService",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Docker must not start")),
    )
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("HTTP must not run")
        )),
    )

    for option in ("--validate-config", "--list-cases"):
        arguments = build_parser().parse_args(["--config", str(config_path), option])
        assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0


def test_compose_profiles_require_a_local_aggregate_report(tmp_path) -> None:
    """Reject S3 profile summaries before any service lifecycle can begin."""
    config = _automation_config(mode="api")
    config["report"] = {"path": "s3://benchmark-reports/automation-{timestamp}.json"}
    _add_compose_service_profiles(tmp_path, config, [
        {"name": "tp1", "project_name": "benchmark-tp1"},
    ])
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="local aggregate report"):
        benchmark_module.run_configured_suite(
            str(config_path), build_parser().parse_args(["--config", str(config_path)])
        )


def test_single_compose_environment_alias_remains_compatible() -> None:
    """Accept the former top-level environment spelling for one service profile."""
    config = _automation_config(mode="api")
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["docker_service"] = {
        "enabled": True,
        "profile_name": "tp4",
        "engine": "vllm",
        "compose_files": ["deploy/base.yml"],
        "project_name": "benchmark-tp4",
        "environment": {"TP_SIZE": "4"},
    }

    assert dict(parse_automation(automation).docker_service.environment) == {"TP_SIZE": "4"}


def test_compose_service_rejects_both_environment_spellings() -> None:
    """Avoid silently ignoring the compatibility alias when both maps are set."""
    config = _automation_config(mode="api")
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["docker_service"] = {
        "enabled": True,
        "profile_name": "tp4",
        "engine": "vllm",
        "compose_files": ["deploy/base.yml"],
        "project_name": "benchmark-tp4",
        "base_environment": {"TP_SIZE": "4"},
        "environment": {"TP_SIZE": "1"},
    }

    with pytest.raises(AutomationConfigError, match="cannot combine"):
        parse_automation(automation)


def test_benchmark_container_schema_is_opt_in_and_validates_service_target() -> None:
    """Keep host execution as default while requiring a safe container API target."""
    config = _automation_config(mode="api")
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["docker_service"] = {
        "enabled": True,
        "profile_name": "tp1",
        "engine": "vllm",
        "compose_files": ["deploy/base.yml"],
        "project_name": "benchmark-tp1",
        "benchmark_container": {
            "enabled": True,
            "service": "benchmark",
            "api_base": "http://vllm:8000/v1",
            "timeout_seconds": 900,
        },
    }

    policy = parse_automation(automation).docker_service.benchmark_container
    assert policy.enabled is True
    assert policy.service == "benchmark"
    assert policy.api_base == "http://vllm:8000/v1"
    assert policy.timeout_seconds == 900.0

    automation["docker_service"]["benchmark_container"]["service"] = "Benchmark Client"
    with pytest.raises(AutomationConfigError, match="benchmark_container.service"):
        parse_automation(automation)


def test_benchmark_container_runs_after_readiness_without_host_execution(monkeypatch, tmp_path) -> None:
    """Delegate the suite to the Compose client container after host readiness succeeds."""
    events: list[str] = []

    class _ManagedService:
        def __init__(self, _service) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def run_benchmark_container(self, **kwargs) -> int:
            events.append("container")
            path = kwargs["report_directory"] / kwargs["report_template_name"].replace(
                "{timestamp}", kwargs["report_timestamp"]
            )
            path.write_text(json.dumps({
                "suite": {"run_state": "completed"},
                "run": {"state": "completed", "automation": {"docker_service": {}}},
                "cases": [],
                "summary": {"failed": 0},
            }), encoding="utf-8")
            return 0

        def stop(self) -> None:
            events.append("stop")

    config = _automation_config(mode="api")
    _add_compose_service_bundle(tmp_path, config)
    automation = config["automation"]
    assert isinstance(automation, dict)
    service = automation["docker_service"]
    assert isinstance(service, dict)
    service["benchmark_container"] = {
        "enabled": True,
        "service": "benchmark",
        "api_base": "http://vllm:8000/v1",
    }
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: (events.append("ready") or _PreflightResponse())),
    )
    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda *_args: (_ for _ in ()).throw(AssertionError("host execution must not run")),
    )

    exit_code, report = _run_suite(monkeypatch, tmp_path, config)

    assert exit_code == 0
    assert events == ["start", "ready", "container", "stop"]
    container_report = report["run"]["automation"]["docker_service"]["benchmark_container"]
    assert container_report["state"] == "completed"
    assert container_report["service"] == "benchmark"
    assert container_report["exit_code"] == 0


def test_benchmark_container_failure_without_report_stops_service(monkeypatch, tmp_path) -> None:
    """Mark the run failed and tear down the engine when the client exits early."""
    events: list[str] = []

    class _ManagedService:
        def __init__(self, _service) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def run_benchmark_container(self, **_kwargs) -> int:
            events.append("container")
            return 9

        def stop(self) -> None:
            events.append("stop")

    config = _automation_config(mode="api")
    _add_compose_service_bundle(tmp_path, config)
    automation = config["automation"]
    assert isinstance(automation, dict)
    service = automation["docker_service"]
    assert isinstance(service, dict)
    service["benchmark_container"] = {
        "enabled": True,
        "service": "benchmark",
        "api_base": "http://vllm:8000/v1",
    }
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: (events.append("ready") or _PreflightResponse())),
    )
    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda *_args: (_ for _ in ()).throw(AssertionError("host execution must not run")),
    )

    exit_code, report = _run_suite(monkeypatch, tmp_path, config)

    assert exit_code == 9
    assert events == ["start", "ready", "container", "stop"]
    assert report["run"]["state"] == "container_failed"
    assert report["suite"]["run_state"] == "container_failed"
    assert report["summary"]["skipped"] == 1
    container_report = report["run"]["automation"]["docker_service"]["benchmark_container"]
    assert container_report["state"] == "failed"
    assert container_report["service"] == "benchmark"
    assert container_report["exit_code"] == 9
    assert container_report["started_at"] is not None
    assert container_report["finished_at"] is not None
    assert container_report["error"] is None


def test_report_timestamp_rejects_unsafe_override_before_inspection(tmp_path) -> None:
    """Reject path-like report timestamp values without running a suite."""
    config = _automation_config()
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args([
        "--config", str(config_path), "--validate-config", "--report-timestamp", "../unsafe",
    ])

    with pytest.raises(BenchmarkConfigError, match="report-timestamp"):
        benchmark_module.run_configured_suite(str(config_path), arguments)


def test_host_inventory_is_written_to_individual_and_profile_summary_reports(
    monkeypatch, tmp_path
) -> None:
    """Persist one sanitized host inventory in every report form without probing hardware."""
    inventory = {"schema_version": 1, "cpu": {"model": "test-cpu"}}
    monkeypatch.setattr(benchmark_module, "collect_host_inventory", lambda: inventory)
    monkeypatch.setattr(
        benchmark_module, "execute_benchmark", lambda *_args: _successful_result()
    )
    config = _automation_config(mode="api")
    _add_compose_service_profiles(tmp_path, config, [
        {"name": "tp1", "project_name": "benchmark-tp1"},
    ])
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        benchmark_module,
        "ManagedComposeService",
        type("Managed", (), {
            "__init__": lambda self, _service: None,
            "start": lambda self: None,
            "stop": lambda self: None,
        }),
    )
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: _PreflightResponse()),
    )

    assert benchmark_module.run_configured_suite(
        str(config_path), build_parser().parse_args(["--config", str(config_path)])
    ) == 0

    reports = list((tmp_path / "reports").glob("automation-*.json"))
    individual = next(path for path in reports if "profiles-summary" not in path.name)
    summary = next(path for path in reports if "profiles-summary" in path.name)
    assert json.loads(individual.read_text(encoding="utf-8"))["environment"]["host_inventory"] == inventory
    assert json.loads(summary.read_text(encoding="utf-8"))["environment"]["host_inventory"] == inventory


def test_benchmark_container_report_keeps_parent_host_inventory(monkeypatch, tmp_path) -> None:
    """Replace container-view hardware data with the parent host inventory in its report."""
    inventory = {"schema_version": 1, "cpu": {"model": "host-cpu"}}
    monkeypatch.setattr(benchmark_module, "collect_host_inventory", lambda: inventory)
    events: list[str] = []

    class _ManagedService:
        def __init__(self, _service) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def run_benchmark_container(self, **kwargs) -> int:
            report_path = kwargs["report_directory"] / kwargs["report_template_name"].replace(
                "{timestamp}", kwargs["report_timestamp"]
            )
            report_path.write_text(json.dumps({
                "suite": {"run_state": "completed"},
                "run": {"id": "container", "state": "completed", "automation": {"docker_service": {}}},
                "environment": {"host_inventory": {"cpu": {"model": "container-cpu"}}},
                "cases": [],
                "summary": {"failed": 0},
            }), encoding="utf-8")
            return 0

        def stop(self) -> None:
            events.append("stop")

    config = _automation_config(mode="api")
    _add_compose_service_bundle(tmp_path, config)
    automation = config["automation"]
    assert isinstance(automation, dict)
    service = automation["docker_service"]
    assert isinstance(service, dict)
    service["benchmark_container"] = {
        "enabled": True,
        "service": "benchmark",
        "api_base": "http://vllm:8000/v1",
    }
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: _PreflightResponse()),
    )

    exit_code, report = _run_suite(monkeypatch, tmp_path, config)

    assert exit_code == 0
    assert report["environment"]["host_inventory"] == inventory
    assert events == ["start", "stop"]


def test_resumable_compose_suite_retries_only_unfinished_cases(monkeypatch, tmp_path) -> None:
    """Keep passed cases while restarting the service for an interrupted case."""
    events: list[str] = []
    calls: list[str] = []

    class _ManagedService:
        def __init__(self, _service) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    config = _automation_config(mode="api", cases=[
        {"name": "first", "params": {"num_prompts": 1}},
        {"name": "second", "params": {"num_prompts": 1}},
    ])
    config["report"] = {"path": "reports/compose-resume.json"}
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["resume"] = True
    _add_compose_service_bundle(tmp_path, config)
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args(["--config", str(config_path)])
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: _PreflightResponse()),
    )

    def interrupt_after_first(_args, _scenario: str) -> dict[str, object]:
        calls.append(_args.model)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return _successful_result()

    monkeypatch.setattr(benchmark_module, "execute_benchmark", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        benchmark_module.run_configured_suite(str(config_path), arguments)

    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda _args, _scenario: (calls.append(_args.model) or _successful_result()),
    )
    assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0

    report_path = tmp_path / "reports" / "compose-resume.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [record["status"] for record in report["cases"]] == ["passed", "passed"]
    assert len(calls) == 3
    assert events == ["start", "stop", "start", "stop"]


def test_resumable_benchmark_container_keeps_child_case_checkpoint(monkeypatch, tmp_path) -> None:
    """Preserve child cases written before an interrupt and forward resume on retry."""
    events: list[str] = []
    container_resume_flags: list[bool] = []
    invocation = [0]

    class _ManagedService:
        def __init__(self, _service) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def run_benchmark_container(self, **kwargs) -> int:
            events.append("container")
            container_resume_flags.append(kwargs["resume"])
            report_path = kwargs["report_directory"] / kwargs["report_template_name"]
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["run"]["id"] = f"container-{invocation[0]}"
            payload["cases"][0]["status"] = "passed"
            payload["cases"][0]["result"] = _successful_result()
            if invocation[0] == 0:
                payload["cases"][1]["status"] = "interrupted"
                payload["cases"][1]["error"] = {
                    "type": "KeyboardInterrupt",
                    "message": "interrupted by user",
                }
                payload["suite"]["run_state"] = "interrupted"
                report_path.write_text(json.dumps(payload), encoding="utf-8")
                invocation[0] += 1
                raise KeyboardInterrupt()
            assert payload["cases"][0]["status"] == "passed"
            assert payload["cases"][1]["status"] == "pending"
            payload["cases"][1]["status"] = "passed"
            payload["cases"][1]["result"] = _successful_result()
            payload["suite"]["run_state"] = "completed"
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            return 0

        def stop(self) -> None:
            events.append("stop")

    config = _automation_config(mode="api", cases=[
        {"name": "first", "params": {"num_prompts": 1}},
        {"name": "second", "params": {"num_prompts": 1}},
    ])
    config["report"] = {"path": "reports/container-resume.json"}
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["resume"] = True
    _add_compose_service_bundle(tmp_path, config)
    service = automation["docker_service"]
    assert isinstance(service, dict)
    service["benchmark_container"] = {
        "enabled": True,
        "service": "benchmark",
        "api_base": "http://vllm:8000/v1",
    }
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args(["--config", str(config_path)])
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: _PreflightResponse()),
    )

    with pytest.raises(KeyboardInterrupt):
        benchmark_module.run_configured_suite(str(config_path), arguments)
    assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0

    report = json.loads(
        (tmp_path / "reports" / "container-resume.json").read_text(encoding="utf-8")
    )
    assert [record["status"] for record in report["cases"]] == ["passed", "passed"]
    assert container_resume_flags == [True, True]
    assert events == ["start", "container", "stop", "start", "container", "stop"]


def test_automation_resume_requires_a_stable_explicit_local_report(tmp_path) -> None:
    """Reject ambiguous or per-run report targets before a resumable run starts."""
    config = _automation_config()
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["resume"] = True
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match=r"without \{timestamp\}"):
        benchmark_module.run_configured_suite(
            str(config_path), build_parser().parse_args(["--config", str(config_path)])
        )

    config["report"] = {"path": "reports/resume.json"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args([
        "--config", str(config_path), "--validate-config",
    ])
    assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0


def test_resumable_profiles_skip_completed_profile_after_interrupt(monkeypatch, tmp_path) -> None:
    """Resume only the interrupted profile while retaining its aggregate checkpoint."""
    events: list[str] = []
    current_project = [""]
    interrupted = [False]

    class _ManagedService:
        def __init__(self, service) -> None:
            self._project_name = service.project_name

        def start(self) -> None:
            current_project[0] = self._project_name
            events.append(f"start:{self._project_name}")

        def stop(self) -> None:
            events.append(f"stop:{self._project_name}")

    config = _automation_config(mode="api")
    config["report"] = {"path": "reports/profile-resume.json"}
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["resume"] = True
    _add_compose_service_profiles(tmp_path, config, [
        {"name": "tp1", "project_name": "benchmark-tp1"},
        {"name": "tp2", "project_name": "benchmark-tp2"},
    ])
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args(["--config", str(config_path)])
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: _PreflightResponse()),
    )

    def interrupt_second_profile(_args, _scenario: str) -> dict[str, object]:
        events.append(f"execute:{current_project[0]}")
        if current_project[0] == "benchmark-tp2" and not interrupted[0]:
            interrupted[0] = True
            raise KeyboardInterrupt()
        return _successful_result()

    monkeypatch.setattr(benchmark_module, "execute_benchmark", interrupt_second_profile)
    with pytest.raises(KeyboardInterrupt):
        benchmark_module.run_configured_suite(str(config_path), arguments)

    summary_path = tmp_path / "reports" / "profile-resume-profiles-summary.json"
    interrupted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert interrupted_summary["state"] == "interrupted"
    assert [item["state"] for item in interrupted_summary["profiles"]] == [
        "completed", "interrupted"
    ]
    assert interrupted_summary["resume"] == {"enabled": True}

    assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["state"] == "completed"
    assert [item["state"] for item in summary["profiles"]] == ["completed", "completed"]
    assert events == [
        "start:benchmark-tp1", "execute:benchmark-tp1", "stop:benchmark-tp1",
        "start:benchmark-tp2", "execute:benchmark-tp2", "stop:benchmark-tp2",
        "start:benchmark-tp2", "execute:benchmark-tp2", "stop:benchmark-tp2",
    ]


def test_no_resume_overrides_resumable_benchmark_container(monkeypatch, tmp_path) -> None:
    """Forward CLI --no-resume to the container despite automation.resume=true."""
    child_resume_flags: list[bool] = []

    class _ManagedService:
        def __init__(self, _service) -> None:
            pass

        def start(self) -> None:
            pass

        def run_benchmark_container(self, **kwargs) -> int:
            child_resume_flags.append(kwargs["resume"])
            report_path = kwargs["report_directory"] / kwargs["report_template_name"]
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["run"]["id"] = "container"
            payload["cases"][0]["status"] = "passed"
            payload["cases"][0]["result"] = _successful_result()
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            return 0

        def stop(self) -> None:
            pass

    config = _automation_config(mode="api")
    config["report"] = {"path": "reports/container-no-resume.json"}
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["resume"] = True
    _add_compose_service_bundle(tmp_path, config)
    service = automation["docker_service"]
    assert isinstance(service, dict)
    service["benchmark_container"] = {
        "enabled": True,
        "service": "benchmark",
        "api_base": "http://vllm:8000/v1",
    }
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args([
        "--config", str(config_path), "--no-resume",
    ])
    monkeypatch.setattr(benchmark_module, "ManagedComposeService", _ManagedService)
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: _PreflightResponse()),
    )

    assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0
    assert child_resume_flags == [False]


def test_automation_without_budget_runs_without_budget_rejection(monkeypatch, tmp_path) -> None:
    """Allow an enabled automation suite to execute when budget checking is omitted."""
    config = _automation_config(mode="api")
    automation = config["automation"]
    assert isinstance(automation, dict)
    del automation["budget"]
    executed: list[str] = []
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: _PreflightResponse()),
    )
    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda *_args: (executed.append("case") or _successful_result()),
    )

    exit_code, report = _run_suite(monkeypatch, tmp_path, config)

    assert exit_code == 0
    assert executed == ["case"]
    budget = report["run"]["automation"]["budget"]
    assert budget["limits"] is None
    assert budget["decision"] == "accepted"
    assert budget["estimate"] is not None


@pytest.mark.parametrize(
    ("engine", "model_profile"),
    [
        ("vllm", "qwen38-27b-fp8"),
        ("vllm", "deepseek-v4-flash-0731"),
        ("sglang", "qwen38-27b-fp8"),
        ("sglang", "deepseek-v4-flash-0731"),
    ],
)
def test_automation_example_selects_engine_and_model_compose_template(
    monkeypatch, engine: str, model_profile: str
) -> None:
    """Expand engine and model selectors into a matching local Compose template."""
    environment = {
        "BENCHMARK_ENGINE": engine,
        "BENCHMARK_MODEL_PROFILE": model_profile,
        "BENCHMARK_ENGINE_IMAGE": "registry.example/engine:stable",
        "BENCHMARK_IMAGE": "registry.example/benchmark:stable",
        "BENCHMARK_MODEL_ID": "example-model",
        "BENCHMARK_TOKENIZER": "example-tokenizer",
        "BENCHMARK_HOST_PORT": "8000",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    config_path = Path(__file__).resolve().parents[1] / "examples" / (
        "benchmark-config-automation.example.json"
    )
    config = load_suite_config(str(config_path))
    automation = config["automation"]
    assert isinstance(automation, dict)
    docker_service = automation["docker_service"]
    assert isinstance(docker_service, dict)
    assert docker_service["engine"] == engine
    assert docker_service["compose_files"] == [
        f"deploy/{engine}/{model_profile}/compose-profiles.example.yml"
    ]
    assert docker_service["base_environment"] == {
        "ENGINE_IMAGE": "registry.example/engine:stable",
        "BENCHMARK_IMAGE": "registry.example/benchmark:stable",
        "MODEL_ID": "example-model",
        "TOKENIZER_ID": "example-tokenizer",
        "SERVED_MODEL_NAME": "example-model",
        "HOST_PORT": "8000",
    }
    assert docker_service["profiles"] == [
        {
            "name": "tp1-dp1",
            "enabled": False,
            "project_name": f"benchmark-{engine}-{model_profile}-tp1-dp1",
            "environment": {
                "GPU_DEVICES": "0", "TP_SIZE": "1", "DP_SIZE": "1", "EP_SIZE": "1",
            },
        },
        {
            "name": "tp4-dp1",
            "enabled": False,
            "project_name": f"benchmark-{engine}-{model_profile}-tp4-dp1",
            "environment": {
                "GPU_DEVICES": "0,1,2,3", "TP_SIZE": "4", "DP_SIZE": "1", "EP_SIZE": "4",
            },
        },
        {
            "name": "tp2-dp2",
            "enabled": False,
            "project_name": f"benchmark-{engine}-{model_profile}-tp2-dp2",
            "environment": {
                "GPU_DEVICES": "0,1,2,3", "TP_SIZE": "2", "DP_SIZE": "2", "EP_SIZE": "2",
            },
        },
    ]

    arguments = build_parser().parse_args(["--config", str(config_path), "--validate-config"])
    assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0


def test_config_changed_resume_requires_explicit_allow_flag(monkeypatch, tmp_path) -> None:
    """Reject a changed execution plan unless the operator explicitly permits it."""
    config = _automation_config(cases=[{"name": "first"}])
    config["report"] = {"path": "reports/config-change-resume.json"}
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["resume"] = True
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = build_parser().parse_args(["--config", str(config_path)])
    monkeypatch.setattr(
        benchmark_module, "execute_benchmark", lambda *_args: _successful_result()
    )

    assert benchmark_module.run_configured_suite(str(config_path), arguments) == 0

    cases = config["cases"]
    assert isinstance(cases, list)
    cases.append({"name": "new", "params": {"num_prompts": 2}})
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="does not match the selected execution plan"):
        benchmark_module.run_configured_suite(str(config_path), arguments)


def test_config_changed_resume_reuses_only_unchanged_successful_cases(
    monkeypatch, tmp_path
) -> None:
    """Reuse unchanged passes but rerun failed and new cases after a plan change."""
    executed: list[int] = []
    config = _automation_config(cases=[
        {"name": "unchanged", "params": {"num_prompts": 1}},
        {"name": "retry", "params": {"num_prompts": 1}},
    ])
    config["report"] = {"path": "reports/config-change-resume.json"}
    automation = config["automation"]
    assert isinstance(automation, dict)
    automation["resume"] = True
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    initial_arguments = build_parser().parse_args(["--config", str(config_path)])
    monkeypatch.setattr(
        benchmark_module,
        "execute_benchmark",
        lambda args, _scenario: (executed.append(args.num_prompts) or _successful_result()),
    )

    assert benchmark_module.run_configured_suite(str(config_path), initial_arguments) == 0
    report_path = tmp_path / "reports" / "config-change-resume.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["cases"][1]["status"] = "failed"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    cases = config["cases"]
    assert isinstance(cases, list)
    cases.append({"name": "new", "params": {"num_prompts": 2}})
    config_path.write_text(json.dumps(config), encoding="utf-8")
    resumed_arguments = build_parser().parse_args([
        "--config", str(config_path), "--resume-allow-config-changes",
    ])

    assert benchmark_module.run_configured_suite(str(config_path), resumed_arguments) == 0

    resumed_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert executed == [1, 1, 1, 2]
    assert [record["status"] for record in resumed_report["cases"]] == [
        "passed", "passed", "passed",
    ]
    assert resumed_report["suite"]["resume_config_change"]["enabled"] is True
