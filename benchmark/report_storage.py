"""Storage backends for JSON benchmark reports.

Local reports retain atomic replacement and an advisory POSIX checkpoint lock.
S3 reports use the standard AWS credential provider chain and are intended for a
single writer per object key: S3 has no portable equivalent of ``flock`` here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit
import uuid


class ReportStorageError(RuntimeError):
    """Raised when a report destination cannot be parsed, read, or written."""


@dataclass(frozen=True)
class ReportLocation:
    """A validated local report path or S3 object location."""

    scheme: str
    display_name: str
    path: str | None = None
    bucket: str | None = None
    key: str | None = None

    @property
    def is_local(self) -> bool:
        return self.scheme == "local"

    @property
    def supports_checkpoint_lock(self) -> bool:
        return self.is_local


def parse_report_location(value: str, *, base_dir: str | None = None) -> ReportLocation:
    """Validate and resolve a local path or ``s3://bucket/key`` report URI."""
    if not isinstance(value, str) or not value.strip():
        raise ReportStorageError("report path must be a non-empty string")

    value = value.strip()
    parsed = urlsplit(value)
    if parsed.scheme == "s3":
        if not parsed.netloc or not parsed.path.lstrip("/"):
            raise ReportStorageError("S3 report URI must use s3://bucket/key")
        if parsed.query or parsed.fragment or "@" in parsed.netloc or ":" in parsed.netloc:
            raise ReportStorageError("S3 report URI cannot contain credentials, a port, query, or fragment")
        return ReportLocation(
            scheme="s3",
            display_name=f"s3://{parsed.netloc}/{parsed.path.lstrip('/')}",
            bucket=parsed.netloc,
            key=parsed.path.lstrip("/"),
        )

    if parsed.scheme:
        raise ReportStorageError(
            f"unsupported report URI scheme {parsed.scheme!r}; use a local path or s3://bucket/key"
        )

    path = Path(value)
    if not path.is_absolute() and base_dir:
        path = Path(base_dir) / path
    resolved = str(path.expanduser().resolve())
    return ReportLocation(scheme="local", display_name=resolved, path=resolved)


def serialize_json_report(report: dict[str, Any], indent: int = 2) -> str:
    """Encode a report consistently for both local files and S3 objects."""
    return json.dumps(
        report,
        ensure_ascii=False,
        indent=indent,
        sort_keys=False,
        allow_nan=False,
    ) + "\n"


def s3_client_kwargs_from_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build boto3 client options from non-secret configuration environment variables.

    ``BENCHMARK_S3_*`` values are intended for S3-compatible services. When no
    custom credentials are supplied, boto3 continues to resolve ``AWS_*``
    variables, profiles, and IAM roles through its standard provider chain.
    """
    env = os.environ if environ is None else environ

    def get(name: str) -> str | None:
        value = env.get(name)
        return value.strip() if isinstance(value, str) and value.strip() else None

    access_key = get("BENCHMARK_S3_ACCESS_KEY_ID")
    secret_key = get("BENCHMARK_S3_SECRET_ACCESS_KEY")
    session_token = get("BENCHMARK_S3_SESSION_TOKEN")
    if bool(access_key) != bool(secret_key):
        raise ReportStorageError(
            "BENCHMARK_S3_ACCESS_KEY_ID and BENCHMARK_S3_SECRET_ACCESS_KEY must be set together"
        )
    if session_token and not access_key:
        raise ReportStorageError(
            "BENCHMARK_S3_SESSION_TOKEN requires BENCHMARK_S3_ACCESS_KEY_ID and "
            "BENCHMARK_S3_SECRET_ACCESS_KEY"
        )

    client_kwargs: dict[str, str] = {}
    endpoint_url = get("BENCHMARK_S3_ENDPOINT_URL")
    if endpoint_url:
        parsed = urlsplit(endpoint_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
            or "@" in parsed.netloc
        ):
            raise ReportStorageError(
                "BENCHMARK_S3_ENDPOINT_URL must be an HTTP(S) URL without credentials, query, or fragment"
            )
        client_kwargs["endpoint_url"] = endpoint_url.rstrip("/")

    region = get("BENCHMARK_S3_REGION")
    if region:
        client_kwargs["region_name"] = region
    if access_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key
    if session_token:
        client_kwargs["aws_session_token"] = session_token
    return client_kwargs


class LocalReportStorage:
    """Atomic local-file report storage with a POSIX advisory checkpoint lock."""

    def __init__(self, location: ReportLocation) -> None:
        if not location.is_local or not location.path:
            raise ValueError("LocalReportStorage requires a local report location")
        self.location = location

    def read_text(self) -> str | None:
        try:
            with open(self.location.path, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ReportStorageError(f"cannot read local report {self.location.display_name!r}: {exc}") from exc

    def write_text(self, text: str) -> None:
        report_path = self.location.path
        assert report_path is not None
        parent = os.path.dirname(report_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temporary = f"{report_path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            with open(temporary, "w", encoding="utf-8") as file:
                file.write(text)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, report_path)
        except OSError as exc:
            raise ReportStorageError(f"cannot write local report {self.location.display_name!r}: {exc}") from exc
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def acquire_checkpoint_lock(self):
        """Acquire a process-lifetime lock for a local suite checkpoint."""
        import fcntl

        report_path = self.location.path
        assert report_path is not None
        lock_path = f"{report_path}.lock"
        parent = os.path.dirname(lock_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        lock_file = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise ReportStorageError(
                f"checkpoint {self.location.display_name!r} is already in use by another benchmark process"
            ) from exc
        return lock_file

    @staticmethod
    def release_checkpoint_lock(lock_file) -> None:
        if lock_file is None:
            return
        import fcntl

        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


class S3ReportStorage:
    """S3 JSON object storage configured by boto3 and ``BENCHMARK_S3_*`` env vars."""

    def __init__(self, location: ReportLocation, *, client: Any | None = None) -> None:
        if location.scheme != "s3" or not location.bucket or not location.key:
            raise ValueError("S3ReportStorage requires an s3://bucket/key location")
        self.location = location
        self._client = client

    @property
    def supports_checkpoint_lock(self) -> bool:
        return False

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError as exc:
            raise ReportStorageError(
                "S3 report output requires boto3; install it with: python -m pip install -e '.[s3]'"
            ) from exc
        try:
            client_kwargs = s3_client_kwargs_from_environment()
            self._client = boto3.client("s3", **client_kwargs)
        except ReportStorageError:
            raise
        except Exception as exc:
            raise ReportStorageError(
                f"cannot create S3 client for {self.location.display_name!r}: {exc}"
            ) from exc
        return self._client

    def read_text(self) -> str | None:
        try:
            response = self._get_client().get_object(
                Bucket=self.location.bucket,
                Key=self.location.key,
            )
            content = response["Body"].read()
            return content.decode("utf-8") if isinstance(content, bytes) else str(content)
        except Exception as exc:
            if _s3_error_code(exc) in {"NoSuchKey", "404", "NotFound"}:
                return None
            if isinstance(exc, ReportStorageError):
                raise
            raise ReportStorageError(f"cannot read S3 report {self.location.display_name!r}: {exc}") from exc

    def write_text(self, text: str) -> None:
        try:
            self._get_client().put_object(
                Bucket=self.location.bucket,
                Key=self.location.key,
                Body=text.encode("utf-8"),
                ContentType="application/json; charset=utf-8",
            )
        except Exception as exc:
            if isinstance(exc, ReportStorageError):
                raise
            raise ReportStorageError(f"cannot write S3 report {self.location.display_name!r}: {exc}") from exc

    @staticmethod
    def acquire_checkpoint_lock():
        """S3 has no portable flock equivalent; callers must enforce one writer."""
        return None

    @staticmethod
    def release_checkpoint_lock(lock_file) -> None:
        del lock_file


def _s3_error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    return error.get("Code") if isinstance(error, dict) else None


def create_report_storage(location: ReportLocation) -> LocalReportStorage | S3ReportStorage:
    """Create the concrete storage backend for a validated report location."""
    if location.is_local:
        return LocalReportStorage(location)
    if location.scheme == "s3":
        return S3ReportStorage(location)
    raise ReportStorageError(f"unsupported report location scheme {location.scheme!r}")
