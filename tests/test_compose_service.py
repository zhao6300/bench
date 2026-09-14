"""Unit tests for the local Docker Compose service lifecycle."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from benchmark.compose_service import (
    ComposeServiceError,
    ComposeServicePolicy,
    ManagedComposeService,
    resolve_compose_service,
)


def _policy() -> ComposeServicePolicy:
    """Return a complete enabled policy for a test-local Compose bundle."""
    return ComposeServicePolicy(
        enabled=True,
        profile_name="tp4-dp2-ep",
        engine="vllm",
        compose_files=("deploy/base.yml", "deploy/profile.yml"),
        env_file="deploy/profile.env",
        project_name="benchmark-tp4-dp2-ep",
        start_timeout_seconds=120.0,
        ready_timeout_seconds=600.0,
        poll_interval_seconds=2.0,
        probe_timeout_seconds=5.0,
    )


def _write_bundle(tmp_path: Path) -> None:
    """Create a minimal local Compose bundle accepted by path validation."""
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    for name in ("base.yml", "profile.yml", "profile.env"):
        (deploy / name).write_text("services: {}\n", encoding="utf-8")
    (tmp_path / "suite.json").write_text("{}\n", encoding="utf-8")


def test_resolve_compose_service_builds_fixed_relative_command(tmp_path: Path) -> None:
    """Resolve bundle paths and preserve their order in fixed Docker argv."""
    _write_bundle(tmp_path)

    service = resolve_compose_service(_policy(), str(tmp_path / "suite.json"))

    assert service is not None
    assert service.command_prefix() == [
        "docker", "compose",
        "--env-file", str(tmp_path / "deploy" / "profile.env"),
        "-f", str(tmp_path / "deploy" / "base.yml"),
        "-f", str(tmp_path / "deploy" / "profile.yml"),
        "-p", "benchmark-tp4-dp2-ep",
    ]


def test_resolve_compose_service_rejects_bundle_path_outside_suite(tmp_path: Path) -> None:
    """Do not let a suite launch a Compose file outside its own directory tree."""
    _write_bundle(tmp_path)
    policy = ComposeServicePolicy(
        **{**_policy().__dict__, "compose_files": ("../outside.yml",)}
    )

    with pytest.raises(ComposeServiceError, match="must remain under"):
        resolve_compose_service(policy, str(tmp_path / "suite.json"))


def test_managed_compose_service_uses_fixed_argv_and_cleans_up(tmp_path: Path) -> None:
    """Run config, ownership check, up, and down without shell interpolation."""
    _write_bundle(tmp_path)
    resolved = resolve_compose_service(_policy(), str(tmp_path / "suite.json"))
    assert resolved is not None
    commands: list[tuple[list[str], float]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append((command, float(kwargs["timeout"])))
        stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    service = ManagedComposeService(resolved, command_runner=runner)
    service.start()
    service.stop()

    prefix = resolved.command_prefix()
    assert commands == [
        (prefix + ["config", "-q"], 120.0),
        (prefix + ["ps", "-aq"], 120.0),
        (prefix + ["up", "--detach", "--no-build", "--pull", "never"], 120.0),
        (prefix + ["down"], 120.0),
    ]


def test_managed_compose_service_rejects_existing_project(tmp_path: Path) -> None:
    """Never attach to or tear down a Compose project not created by this run."""
    _write_bundle(tmp_path)
    resolved = resolve_compose_service(_policy(), str(tmp_path / "suite.json"))
    assert resolved is not None

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = "existing-container\n" if command[-2:] == ["ps", "-aq"] else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    with pytest.raises(ComposeServiceError, match="existing Docker Compose project"):
        ManagedComposeService(resolved, command_runner=runner).start()


def test_failed_compose_up_attempts_controlled_cleanup(tmp_path: Path) -> None:
    """Issue down after an unsuccessful up because Docker may create partial resources."""
    _write_bundle(tmp_path)
    resolved = resolve_compose_service(_policy(), str(tmp_path / "suite.json"))
    assert resolved is not None
    verbs: list[str] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "up" in command:
            verbs.append("up")
        elif "down" in command:
            verbs.append("down")
        returncode = 3 if "up" in command else 0
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="secret")

    with pytest.raises(ComposeServiceError, match="compose up"):
        ManagedComposeService(resolved, command_runner=runner).start()

    assert verbs[-2:] == ["up", "down"]


def test_profile_environment_overrides_base_for_every_compose_command(tmp_path: Path) -> None:
    """Pass only merged environment variables to each fixed Compose invocation."""
    from benchmark.compose_service import ComposeServiceProfile

    _write_bundle(tmp_path)
    policy = ComposeServicePolicy(
        **{
            **_policy().__dict__,
            "environment": (("ENGINE_IMAGE", "base-image"), ("TP_SIZE", "1")),
        }
    )
    profile = ComposeServiceProfile(
        name="tp4-dp2",
        enabled=True,
        project_name="benchmark-tp4-dp2",
        environment=(("TP_SIZE", "4"), ("DP_SIZE", "2")),
    )
    resolved = resolve_compose_service(
        policy.select_profile(profile), str(tmp_path / "suite.json")
    )
    assert resolved is not None
    environments: list[dict[str, str]] = []
    commands: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        environments.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    service = ManagedComposeService(resolved, command_runner=runner)
    service.start()
    service.stop()

    assert all(environment["ENGINE_IMAGE"] == "base-image" for environment in environments)
    assert all(environment["TP_SIZE"] == "4" for environment in environments)
    assert all(environment["DP_SIZE"] == "2" for environment in environments)
    assert all("TP_SIZE=4" not in command for command in commands)
    assert all("DP_SIZE=2" not in command for command in commands)


def test_managed_compose_service_runs_fixed_benchmark_container_command(tmp_path: Path) -> None:
    """Run only the configured client service with fixed mounts and benchmark flags."""
    from benchmark.compose_service import BenchmarkContainerPolicy

    _write_bundle(tmp_path)
    config_file = tmp_path / "generated-suite.json"
    config_file.write_text("{}\n", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    policy = ComposeServicePolicy(
        **{
            **_policy().__dict__,
            "benchmark_container": BenchmarkContainerPolicy(
                enabled=True,
                service="benchmark",
                api_base="http://vllm:8000/v1",
                timeout_seconds=900.0,
            ),
        }
    )
    resolved = resolve_compose_service(policy, str(tmp_path / "suite.json"))
    assert resolved is not None
    commands: list[tuple[list[str], float]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append((command, float(kwargs["timeout"])))
        returncode = 7 if "run" in command else 0
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")

    service = ManagedComposeService(resolved, command_runner=runner)
    service.start()
    exit_code = service.run_benchmark_container(
        config_file=config_file,
        report_directory=reports,
        report_template_name="report-{timestamp}.json",
        report_timestamp="20260830-p123",
        case_filters=["smoke"],
        tag_filters=["api"],
        fail_fast=True,
    )
    service.stop()

    prefix = resolved.command_prefix()
    assert exit_code == 7
    assert commands[3] == (
        prefix + [
            "run", "--rm", "--no-deps", "--no-tty", "--pull", "never",
            "--volume", f"{config_file}:/benchmark-input/suite.json:ro",
            "--volume", f"{reports}:/benchmark-output", "benchmark",
            "--config", "/benchmark-input/suite.json",
            "--report", "/benchmark-output/report-{timestamp}.json",
            "--report-timestamp", "20260830-p123", "--no-resume", "--progress", "off",
            "--case", "smoke", "--tag", "api", "--fail-fast",
        ],
        900.0,
    )


def test_benchmark_container_resume_forwards_config_change_flag(tmp_path: Path) -> None:
    """Forward checkpoint resume and plan-change permission into the client container."""
    from benchmark.compose_service import BenchmarkContainerPolicy

    _write_bundle(tmp_path)
    config_file = tmp_path / "generated-suite.json"
    config_file.write_text("{}\n", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    policy = ComposeServicePolicy(
        **{
            **_policy().__dict__,
            "benchmark_container": BenchmarkContainerPolicy(
                enabled=True,
                service="benchmark",
                api_base="http://vllm:8000/v1",
                timeout_seconds=900.0,
            ),
        }
    )
    resolved = resolve_compose_service(policy, str(tmp_path / "suite.json"))
    assert resolved is not None
    commands: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    service = ManagedComposeService(resolved, command_runner=runner)
    service.start()
    assert service.run_benchmark_container(
        config_file=config_file,
        report_directory=reports,
        report_template_name="report.json",
        report_timestamp="20260830-p123",
        case_filters=[],
        tag_filters=[],
        fail_fast=False,
        resume=True,
        resume_allow_config_changes=True,
    ) == 0
    service.stop()

    assert "--no-resume" not in commands[3]
    assert "--resume-allow-config-changes" in commands[3]
    assert commands[3][commands[3].index("--report") + 1] == "/benchmark-output/report.json"


def test_compose_debug_log_does_not_emit_captured_child_output(tmp_path: Path, capsys) -> None:
    """确认 debug 模式不输出可能敏感的 Compose stdout 和 stderr。"""
    from benchmark.logging_utils import configure_logging

    _write_bundle(tmp_path)
    resolved = resolve_compose_service(_policy(), str(tmp_path / "suite.json"))
    assert resolved is not None

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = "" if command[-2:] == ["ps", "-aq"] else "MODEL_SECRET_OUTPUT"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=stdout,
            stderr="BEARER secret-token",
        )

    configure_logging(True)
    try:
        service = ManagedComposeService(resolved, command_runner=runner)
        service.start()
        service.stop()
    finally:
        configure_logging(False)

    debug_output = capsys.readouterr().err
    assert "compose command started" in debug_output
    assert "MODEL_SECRET_OUTPUT" not in debug_output
    assert "secret-token" not in debug_output
