from __future__ import annotations

import io

from benchmark.logging_utils import configure_logging, get_logger, safe_endpoint


def test_debug_logging_is_disabled_by_default_configuration(monkeypatch) -> None:
    """确认未启用 CLI 开关时诊断通道保持静默。"""
    stream = io.StringIO()
    monkeypatch.setattr("sys.stderr", stream)

    configure_logging(False)
    get_logger("test").debug("hidden diagnostic")

    assert stream.getvalue() == ""


def test_debug_logging_follows_runtime_stderr_replacement(monkeypatch) -> None:
    """确认日志跟随 Rich 运行时重定向后的 stderr。"""
    first_stream = io.StringIO()
    second_stream = io.StringIO()
    monkeypatch.setattr("sys.stderr", first_stream)

    configure_logging(True)
    monkeypatch.setattr("sys.stderr", second_stream)
    get_logger("test").debug("visible diagnostic")

    assert first_stream.getvalue() == ""
    assert "visible diagnostic" in second_stream.getvalue()


def test_safe_endpoint_removes_credentials_and_query_parameters() -> None:
    """确认 endpoint 诊断信息不会暴露 URL 凭据或 token。"""
    endpoint = safe_endpoint(
        "https://user:password@example.com:8443/v1/models?api_key=secret#fragment"
    )

    assert endpoint == "example.com:8443/v1/models"
    assert "password" not in endpoint
    assert "secret" not in endpoint
