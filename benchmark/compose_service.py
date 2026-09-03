"""Managed local Docker Compose services for benchmark automation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Callable


class ComposeServiceError(RuntimeError):
    """Raised when a managed Docker Compose service cannot be controlled."""


@dataclass(frozen=True)
class ComposeServiceProfile:
    """One serially executed engine-parameter profile."""

    name: str
    enabled: bool
    project_name: str
    environment: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BenchmarkContainerPolicy:
    """Validated settings for the optional Compose benchmark client container."""

    enabled: bool
    service: str | None
    api_base: str | None
    timeout_seconds: float


@dataclass(frozen=True)
class ComposeServicePolicy:
    """Validated configuration for local Compose inference services."""

    enabled: bool
    profile_name: str | None
    engine: str | None
    compose_files: tuple[str, ...]
    env_file: str | None
    project_name: str | None
    start_timeout_seconds: float
    ready_timeout_seconds: float
    poll_interval_seconds: float
    probe_timeout_seconds: float
    benchmark_container: BenchmarkContainerPolicy = BenchmarkContainerPolicy(
        enabled=False,
        service=None,
        api_base=None,
        timeout_seconds=3600.0,
    )
    environment: tuple[tuple[str, str], ...] = ()
    profiles: tuple[ComposeServiceProfile, ...] = ()

    def select_profile(self, profile: ComposeServiceProfile) -> ComposeServicePolicy:
        """Create a single-service policy with base and profile environments.

        Args:
            profile: One validated profile selected for execution.

        Returns:
            A policy suitable for one service lifecycle.
        """
        return replace(
            self,
            profile_name=profile.name,
            project_name=profile.project_name,
            environment=(*self.environment, *profile.environment),
            profiles=(),
        )


@dataclass(frozen=True)
class ResolvedComposeService:
    """A Compose service policy resolved relative to its suite configuration."""

    policy: ComposeServicePolicy
    compose_files: tuple[Path, ...]
    env_file: Path | None
    project_name: str
    environment: tuple[tuple[str, str], ...]
    lock_path: Path

    def command_prefix(self) -> list[str]:
        """Build the fixed Compose command prefix without lifecycle verbs."""
        command = ["docker", "compose"]
        if self.env_file is not None:
            command.extend(["--env-file", str(self.env_file)])
        for compose_file in self.compose_files:
            command.extend(["-f", str(compose_file)])
        command.extend(["-p", self.project_name])
        return command


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def resolve_compose_service(
    policy: ComposeServicePolicy,
    config_path: str,
) -> ResolvedComposeService | None:
    """Resolve a service bundle and reject paths outside the suite directory.

    Args:
        policy: Parsed single-service policy.
        config_path: Path to the suite JSON file.

    Returns:
        Resolved local service configuration, or None when the service is disabled.

    Raises:
        ComposeServiceError: If a configured bundle file is missing or unsafe.
    """
    if not policy.enabled:
        return None
    if policy.profiles:
        raise ComposeServiceError("select one docker_service profile before resolving it")
    config_directory = Path(config_path).resolve().parent
    compose_files = tuple(
        _resolve_bundle_file(config_directory, value, "compose_files")
        for value in policy.compose_files
    )
    env_file = (
        _resolve_bundle_file(config_directory, policy.env_file, "env_file")
        if policy.env_file is not None else None
    )
    assert policy.project_name is not None
    digest = hashlib.sha256(
        f"{config_directory}\0{policy.project_name}".encode("utf-8")
    ).hexdigest()[:16]
    return ResolvedComposeService(
        policy=policy,
        compose_files=compose_files,
        env_file=env_file,
        project_name=policy.project_name,
        environment=policy.environment,
        lock_path=config_directory / f".benchmark-compose-{digest}.lock",
    )


class ManagedComposeService:
    """Start and stop a locally owned Docker Compose project safely."""

    def __init__(
        self,
        service: ResolvedComposeService,
        *,
        command_runner: CommandRunner = subprocess.run,
    ) -> None:
        self._service = service
        self._command_runner = command_runner
        self._lock_file: Any | None = None
        self._started = False
        self._stopped = False

    @property
    def service(self) -> ResolvedComposeService:
        """Return immutable resolved service configuration."""
        return self._service

    def start(self) -> None:
        """Validate, lock, and start the configured Compose project.

        Raises:
            ComposeServiceError: If Docker Compose validation, ownership checks, or
                startup cannot complete.
        """
        self._acquire_lock()
        try:
            self._run("config", "-q", timeout=self._service.policy.start_timeout_seconds)
            existing = self._run("ps", "-aq", timeout=self._service.policy.start_timeout_seconds)
            if existing.stdout.strip():
                raise ComposeServiceError(
                    "refusing to manage an existing Docker Compose project: "
                    f"{self._service.project_name}"
                )
            self._started = True
            self._run(
                "up",
                "--detach",
                "--no-build",
                "--pull",
                "never",
                timeout=self._service.policy.start_timeout_seconds,
            )
        except BaseException as exc:
            try:
                self.stop()
            except BaseException as cleanup_exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise ComposeServiceError(
                    f"{exc}; Docker Compose cleanup also failed: {cleanup_exc}"
                ) from exc
            raise

    def stop(self) -> None:
        """Stop only this manager's started project and release its lock.

        Raises:
            ComposeServiceError: If the owned project cannot be stopped.
        """
        if self._stopped:
            return
        try:
            if self._started:
                self._run(
                    "down",
                    timeout=self._service.policy.start_timeout_seconds,
                )
            self._stopped = True
        finally:
            self._release_lock()

    def run_benchmark_container(
        self,
        config_file: Path,
        report_directory: Path,
        report_template_name: str,
        report_timestamp: str,
        case_filters: list[str],
        tag_filters: list[str],
        fail_fast: bool,
        resume: bool = False,
        resume_allow_config_changes: bool = False,
    ) -> int:
        """Run the configured benchmark client service with fixed safe arguments.

        Args:
            config_file: Generated profile-specific suite configuration to mount read-only.
            report_directory: Local directory mounted for the container JSON report.
            report_template_name: Timestamped report file name, without a directory.
            report_timestamp: Trusted report timestamp chosen by the parent suite.
            case_filters: Optional case filters forwarded as individual CLI values.
            tag_filters: Optional tag filters forwarded as individual CLI values.
            fail_fast: Whether to stop the suite after a failed case.
            resume: Whether the child suite may load its existing checkpoint.
            resume_allow_config_changes: Whether the child may reuse unchanged
                passed cases after its execution plan changes.

        Returns:
            The benchmark container process exit code.

        Raises:
            ComposeServiceError: If the fixed Compose client invocation cannot start.
        """
        policy = self._service.policy.benchmark_container
        if not policy.enabled or policy.service is None:
            raise ComposeServiceError("benchmark container mode is not enabled")
        if not self._started or self._stopped:
            raise ComposeServiceError("benchmark container requires a running managed Compose project")
        if not config_file.is_file():
            raise ComposeServiceError("generated benchmark container config file is missing")
        if not report_directory.is_dir():
            raise ComposeServiceError("benchmark report directory is missing")
        if Path(report_template_name).name != report_template_name:
            raise ComposeServiceError("benchmark container report target must be a file name")
        arguments = [
            "run",
            "--rm",
            "--no-deps",
            "--no-tty",
            "--pull",
            "never",
            "--volume",
            f"{config_file}:/benchmark-input/suite.json:ro",
            "--volume",
            f"{report_directory}:/benchmark-output",
            policy.service,
            "--config",
            "/benchmark-input/suite.json",
            "--report",
            f"/benchmark-output/{report_template_name}",
            "--report-timestamp",
            report_timestamp,
        ]
        if not resume:
            arguments.append("--no-resume")
        if resume_allow_config_changes:
            arguments.append("--resume-allow-config-changes")
        arguments.extend([
            "--progress",
            "off",
        ])
        for case_filter in case_filters:
            arguments.extend(["--case", case_filter])
        for tag_filter in tag_filters:
            arguments.extend(["--tag", tag_filter])
        if fail_fast:
            arguments.append("--fail-fast")
        return self._run(
            *arguments,
            timeout=policy.timeout_seconds,
            allow_failure=True,
        ).returncode

    def _run(
        self,
        *arguments: str,
        timeout: float,
        allow_failure: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run a fixed Compose argv and hide potentially sensitive command output."""
        command = [*self._service.command_prefix(), *arguments]
        environment = {**os.environ, **dict(self._service.environment)}
        try:
            result = self._command_runner(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise ComposeServiceError(
                "docker CLI was not found; install Docker and Docker Compose"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ComposeServiceError(
                f"docker compose {' '.join(arguments)} timed out after {timeout:g} seconds"
            ) from exc
        except OSError as exc:
            raise ComposeServiceError(
                f"docker compose {' '.join(arguments)} could not start: {type(exc).__name__}"
            ) from exc
        if result.returncode != 0 and not allow_failure:
            raise ComposeServiceError(
                f"docker compose {' '.join(arguments)} failed with exit code "
                f"{result.returncode}"
            )
        return result

    def _acquire_lock(self) -> None:
        """Acquire a nonblocking local lifecycle lock for one Compose project."""
        import fcntl

        self._service.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self._service.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise ComposeServiceError(
                "Docker Compose project is already managed by another benchmark process: "
                f"{self._service.project_name}"
            ) from exc
        self._lock_file = lock_file

    def _release_lock(self) -> None:
        """Release the lifecycle lock when it was acquired."""
        if self._lock_file is None:
            return
        import fcntl

        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_file.close()
            self._lock_file = None


def _resolve_bundle_file(directory: Path, value: str, field: str) -> Path:
    """Resolve one regular bundle file under the suite configuration directory."""
    path = Path(value)
    if path.is_absolute():
        raise ComposeServiceError(f"docker_service.{field} entries must be relative paths")
    resolved = (directory / path).resolve()
    try:
        resolved.relative_to(directory)
    except ValueError as exc:
        raise ComposeServiceError(
            f"docker_service.{field} must remain under the suite configuration directory"
        ) from exc
    if not resolved.is_file():
        raise ComposeServiceError(f"docker_service.{field} file not found: {value}")
    return resolved
