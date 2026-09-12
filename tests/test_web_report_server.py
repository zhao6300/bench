"""Tests for local web report server helpers."""

from __future__ import annotations

import pathlib
import sqlite3
import time

import pytest

from web import report_server
from web.auth_store import AuthStore, initialize_database


def test_run_metadata_lists_reports_without_private_paths(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use report filenames only and hide filesystem paths from payloads."""
    older = tmp_path / "older-report.json"
    newer = tmp_path / "newer-report.json"
    hidden = tmp_path / ".hidden-report.json"
    non_json = tmp_path / "not-a-report.txt"
    older.write_text('{"cases": []}', encoding="utf-8")
    hidden.write_text("{}", encoding="utf-8")
    non_json.write_text("x", encoding="utf-8")
    newer.write_text('{"cases": []}', encoding="utf-8")
    time.sleep(0.01)
    newer.touch()
    monkeypatch.setattr(report_server, "RUNS_ROOT", tmp_path)

    metadata = report_server.run_metadata()

    assert [item["filename"] for item in metadata] == [newer.name, older.name]
    assert set(metadata[0]) == {"filename", "size_bytes", "modified_at"}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/", "text/html; charset=utf-8"),
        ("/index.html", "text/html; charset=utf-8"),
        ("/assets/style.css", "text/css; charset=utf-8"),
        ("/runs/report.json", "application/json; charset=utf-8"),
    ],
)
def test_static_content_type_accepts_supported_files(path: str, expected: str) -> None:
    """Map supported page assets while retaining JSON reports as JSON."""
    assert report_server.ReportRequestHandler.static_content_type(path) == expected


@pytest.mark.parametrize(
    ("host", "allow_non_loopback", "raises"),
    [
        ("127.0.0.1", False, False),
        ("0.0.0.0", False, True),
        ("192.168.1.10", False, True),
        ("0.0.0.0", True, False),
        ("192.168.1.10", True, False),
    ],
)
def test_parse_args_requires_explicit_non_loopback_permission(
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    allow_non_loopback: bool,
    raises: bool,
) -> None:
    """Require an explicit launch flag before binding outside loopback."""
    argv = [
        "report-server",
        "--host",
        host,
        "--allow-non-loopback",
    ] if allow_non_loopback else [
        "report-server",
        "--host",
        host,
    ]
    monkeypatch.setattr("sys.argv", argv)
    if raises:
        with pytest.raises(ValueError):
            report_server.parse_args()
    else:
        parsed = report_server.parse_args()
        assert parsed.host == host
        assert parsed.allow_non_loopback is allow_non_loopback


@pytest.mark.parametrize(
    "path",
    [None, "", "index.html", "http://127.0.0.2/index.html"],
)
def test_resolve_under_root_rejects_invalid_paths(path: str | None) -> None:
    """Raise an error instead of treating malformed paths as files."""
    with pytest.raises(ValueError):
        report_server.resolve_under_root(path, pathlib.Path("/tmp/web"))


@pytest.mark.parametrize("path", ["/..", "/../secret.json", "/assets/../../benchmark.py"])
def test_resolve_under_root_rejects_traversal(path: str) -> None:
    """Keep URL resolution inside the configured application root."""
    assert report_server.resolve_under_root(path, pathlib.Path("/tmp/web")) is None


def test_parse_args_limits_host_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the report service bound to loopback until explicitly allowed."""
    monkeypatch.setattr("sys.argv", ["report-server", "--host", "127.0.0.1"])
    assert report_server.parse_args().host == "127.0.0.1"

    monkeypatch.setattr("sys.argv", ["report-server", "--host", "0.0.0.0"])
    with pytest.raises(ValueError):
        report_server.parse_args()


def test_initialize_database_creates_first_administrator(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bootstrap the first scrypt password hash as a persistent SQLite row."""
    database = tmp_path / "auth.sqlite3"
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_USERNAME", "editor")
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_PASSWORD", "benchmark-password")

    initialize_database(database)

    assert database.exists()
    assert (database.stat().st_mode & 0o777) == 0o600
    assert AuthStore(database).authenticate("editor", "benchmark-password") is True
    monkeypatch.delenv("BENCHMARK_WEB_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("BENCHMARK_WEB_ADMIN_PASSWORD", raising=False)
    initialize_database(database)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM admin_users").fetchone()[0] == 1
    assert AuthStore(database).authenticate("editor", "changed-password") is False


def test_auth_store_accepts_only_valid_credentials(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject missing, stale, and invalid logins without exposing database values."""
    database = tmp_path / "auth.sqlite3"
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_USERNAME", "operator")
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_PASSWORD", "secure-admin-password")
    store = AuthStore(database)
    assert store.authenticate("operator", "short") is False

    token = store.create_session("operator")
    assert store.username_for_session(token) == "operator"
    assert store.username_for_session("expired-" + token) is None
    store.delete_session(token)
    assert store.username_for_session(token) is None
