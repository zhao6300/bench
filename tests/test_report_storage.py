from __future__ import annotations

from io import BytesIO
import json
import math
import sys
from types import SimpleNamespace

import pytest

from benchmark.benchmark import BenchmarkConfigError, _load_resume_report, _write_json_report, load_suite_config
from benchmark.report_storage import (
    ReportStorageError,
    S3ReportStorage,
    create_report_storage,
    parse_report_location,
    s3_client_kwargs_from_environment,
    serialize_json_report,
)


class FakeS3Error(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_requests: list[dict[str, object]] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, BytesIO]:
        try:
            return {"Body": BytesIO(self.objects[(Bucket, Key)])}
        except KeyError as exc:
            raise FakeS3Error("NoSuchKey") from exc

    def put_object(self, **kwargs: object) -> None:
        self.put_requests.append(kwargs)
        self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))] = bytes(kwargs["Body"])


def test_parse_report_location_resolves_local_paths_and_s3_uris(tmp_path) -> None:
    local = parse_report_location("reports/result.json", base_dir=str(tmp_path))
    s3 = parse_report_location("s3://benchmark-reports/results/report.json")

    assert local.is_local
    assert local.path == str(tmp_path / "reports" / "result.json")
    assert s3.scheme == "s3"
    assert s3.bucket == "benchmark-reports"
    assert s3.key == "results/report.json"
    assert s3.display_name == "s3://benchmark-reports/results/report.json"


@pytest.mark.parametrize(
    "value",
    ["s3://", "s3://bucket", "https://bucket/key", "s3://bucket/key?versionId=1"],
)
def test_parse_report_location_rejects_invalid_destinations(value: str) -> None:
    with pytest.raises(ReportStorageError):
        parse_report_location(value)


def test_local_storage_round_trip_and_resume_loading(tmp_path) -> None:
    location = parse_report_location("nested/checkpoint.json", base_dir=str(tmp_path))
    storage = create_report_storage(location)
    report = {
        "suite": {"name": "suite", "execution_plan_sha256": "plan"},
        "cases": [],
    }

    _write_json_report(storage, report)

    assert json.loads(storage.read_text()) == report
    assert _load_resume_report(storage, "suite", "plan") == report
    assert (tmp_path / "nested" / "checkpoint.json").exists()


def test_s3_storage_writes_and_reads_through_injected_client() -> None:
    client = FakeS3Client()
    location = parse_report_location("s3://benchmark-reports/reports/result.json")
    storage = S3ReportStorage(location, client=client)

    storage.write_text('{"status":"ok"}\n')

    assert client.put_requests == [{
        "Bucket": "benchmark-reports",
        "Key": "reports/result.json",
        "Body": b'{"status":"ok"}\n',
        "ContentType": "application/json; charset=utf-8",
    }]
    assert storage.read_text() == '{"status":"ok"}\n'
    assert S3ReportStorage(
        parse_report_location("s3://benchmark-reports/reports/missing.json"),
        client=client,
    ).read_text() is None


def test_s3_storage_reports_missing_optional_dependency(monkeypatch) -> None:
    storage = S3ReportStorage(parse_report_location("s3://bucket/report.json"))
    original_import = __import__

    def fail_boto3_import(name: str, *args: object, **kwargs: object):
        if name == "boto3":
            raise ImportError("simulated missing boto3")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_boto3_import)
    with pytest.raises(ReportStorageError, match="boto3"):
        storage.write_text("{}\n")


def test_report_serializer_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        serialize_json_report({"metric": math.nan})


def test_suite_config_accepts_s3_path_and_rejects_invalid_s3_path(tmp_path) -> None:
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps({
        "version": 1,
        "cases": [{"name": "smoke"}],
        "report": {"path": "s3://benchmark-reports/reports/result.json"},
    }), encoding="utf-8")

    assert load_suite_config(str(config_path))["report"]["path"].startswith("s3://")

    config_path.write_text(json.dumps({
        "version": 1,
        "cases": [{"name": "smoke"}],
        "report": {"path": "s3://benchmark-reports"},
    }), encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="invalid report.path"):
        load_suite_config(str(config_path))


def test_s3_client_kwargs_read_custom_credentials_and_endpoint() -> None:
    kwargs = s3_client_kwargs_from_environment({
        "BENCHMARK_S3_ACCESS_KEY_ID": "access-key",
        "BENCHMARK_S3_SECRET_ACCESS_KEY": "secret-key",
        "BENCHMARK_S3_SESSION_TOKEN": "session-token",
        "BENCHMARK_S3_ENDPOINT_URL": "https://oss-cn-hangzhou.aliyuncs.com/",
        "BENCHMARK_S3_REGION": "cn-hangzhou",
    })

    assert kwargs == {
        "endpoint_url": "https://oss-cn-hangzhou.aliyuncs.com",
        "region_name": "cn-hangzhou",
        "aws_access_key_id": "access-key",
        "aws_secret_access_key": "secret-key",
        "aws_session_token": "session-token",
    }


@pytest.mark.parametrize(
    "environment",
    [
        {"BENCHMARK_S3_ACCESS_KEY_ID": "access-key"},
        {"BENCHMARK_S3_SECRET_ACCESS_KEY": "secret-key"},
        {"BENCHMARK_S3_SESSION_TOKEN": "session-token"},
        {"BENCHMARK_S3_ENDPOINT_URL": "ftp://storage.example.com"},
        {"BENCHMARK_S3_ENDPOINT_URL": "https://key:secret@storage.example.com"},
    ],
)
def test_s3_client_kwargs_reject_invalid_environment(environment: dict[str, str]) -> None:
    with pytest.raises(ReportStorageError):
        s3_client_kwargs_from_environment(environment)


def test_s3_storage_creates_boto_client_from_environment(monkeypatch) -> None:
    observed: dict[str, object] = {}
    fake_client = object()

    def create_client(service_name: str, **kwargs: object) -> object:
        observed["service_name"] = service_name
        observed["kwargs"] = kwargs
        return fake_client

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=create_client))
    monkeypatch.setenv("BENCHMARK_S3_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("BENCHMARK_S3_SECRET_ACCESS_KEY", "secret-key")
    monkeypatch.setenv("BENCHMARK_S3_ENDPOINT_URL", "https://oss-cn-hangzhou.aliyuncs.com")
    monkeypatch.setenv("BENCHMARK_S3_REGION", "cn-hangzhou")

    storage = S3ReportStorage(parse_report_location("s3://reports/result.json"))

    assert storage._get_client() is fake_client
    assert observed == {
        "service_name": "s3",
        "kwargs": {
            "endpoint_url": "https://oss-cn-hangzhou.aliyuncs.com",
            "region_name": "cn-hangzhou",
            "aws_access_key_id": "access-key",
            "aws_secret_access_key": "secret-key",
        },
    }


def test_suite_config_rejects_unknown_failure_policy(tmp_path) -> None:
    """Reject unsupported suite failure policies before benchmark execution."""
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps({
        "version": 1,
        "failure_policy": "stop-all-matrices",
        "cases": [{"name": "smoke"}],
    }), encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="failure_policy must be one of"):
        load_suite_config(str(config_path))
