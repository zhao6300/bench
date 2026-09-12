"""Tests for local web report server helpers."""

from __future__ import annotations

import pathlib
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.client import HTTPResponse
from http.server import ThreadingHTTPServer

import pytest
from web import report_server
from web.auth_store import AuthStore, initialize_database
from web.report_server import create_local_server


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


def test_migrates_legacy_auth_schema_with_admin_profile_fields(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Add profile columns without losing legacy password hashes or sessions."""
    database = tmp_path / "legacy-auth.sqlite3"
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_PASSWORD", "legacy-password-value")
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_USERNAME", "legacy-admin")

    initialize_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE web_sessions")
        connection.execute(
            """
            CREATE TABLE web_sessions (
                token_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL REFERENCES admin_users(username) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            f"""
            ALTER TABLE admin_users
            RENAME COLUMN display_name TO __old_display_name
            """
        )
    store = AuthStore(database)
    profile = store.get_profile("legacy-admin")

    assert profile == {
        "username": "legacy-admin",
        "display_name": None,
        "email": None,
        "avatar_url": None,
    }
    assert store.authenticate("legacy-admin", "legacy-password-value") is True


def test_updates_auth_profile_fields(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persist optional display name, email, and avatar image data."""
    database = tmp_path / "auth.sqlite3"
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_USERNAME", "operator")
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_PASSWORD", "secure-admin-password")
    store = AuthStore(database)
    avatar_url = "data:image/png;base64,iVBORw0KGgo="

    updated = store.update_profile(
        "operator",
        display_name="运维负责人",
        email="operator@example.com",
        avatar_url=avatar_url,
    )

    assert updated == {
        "username": "operator",
        "display_name": "运维负责人",
        "email": "operator@example.com",
        "avatar_url": avatar_url,
    }
    assert store.get_profile("operator") == updated


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("display_name", "a" * 65),
        ("email", "not-an-email"),
        ("avatar_url", "https://example.invalid/i.png"),
        ("avatar_url", "data:image/png;base64,%%%"),
    ],
)
def test_rejects_invalid_auth_profile_fields(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    """Prevent unsupported profile values from reaching public pages."""
    database = tmp_path / "auth.sqlite3"
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_USERNAME", "operator")
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_PASSWORD", "secure-admin-password")
    store = AuthStore(database)

    with pytest.raises(ValueError):
        store.update_profile("operator", **{field: value})


def test_change_password_keeps_only_current_browser_session(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace the password while retaining one explicit authenticated session."""
    database = tmp_path / "auth.sqlite3"
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_USERNAME", "operator")
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_PASSWORD", "secure-admin-password")
    store = AuthStore(database)
    current_token = store.create_session("operator")
    other_token = store.create_session("operator")

    changed = store.change_password(
        "operator",
        "secure-admin-password",
        "new-secure-password",
        keep_session_token=current_token,
    )

    assert changed is True
    assert store.authenticate("operator", "secure-admin-password") is False
    assert store.authenticate("operator", "new-secure-password") is True
    assert store.username_for_session(current_token) == "operator"
    assert store.username_for_session(other_token) is None


def test_change_password_rejects_wrong_current_password(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require the existing password before accepting a replacement."""
    database = tmp_path / "auth.sqlite3"
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_USERNAME", "operator")
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_PASSWORD", "secure-admin-password")
    store = AuthStore(database)

    changed = store.change_password("operator", "wrong-current-password", "new-secure-password")

    assert changed is False
    assert store.authenticate("operator", "secure-admin-password") is True
    assert store.authenticate("operator", "new-secure-password") is False


def _start_auth_server(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    username: str,
    password: str,
) -> tuple[ThreadingHTTPServer, str, AuthStore]:
    """Start one isolated report server server on loopback."""
    database = tmp_path / "auth-api.sqlite3"
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_USERNAME", username)
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_PASSWORD", password)
    store = AuthStore(database)
    server = create_local_server(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}", store


def _start_empty_server(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ThreadingHTTPServer, str]:
    """Start an isolated report server without a pre-authorized session."""
    database = tmp_path / "auth-empty.sqlite3"
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_PASSWORD", "first-password-value")
    monkeypatch.setenv("BENCHMARK_WEB_ADMIN_USERNAME", "empty")
    store = AuthStore(database)
    server = create_local_server(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _get_profile(
    server_url: str,
    token: str | None,
    body: dict[str, str | None] | None = None,
    method: str = "GET",
) -> HTTPResponse:
    """Send one profile API request to the local test server."""
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if token is not None:
        headers["Cookie"] = f"benchmark_session={token}"
    request = urllib.request.Request(
        f"{server_url}/api/auth/profile",
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers=headers,
        method=method,
    )
    response = urllib.request.urlopen(request)
    assert isinstance(response, HTTPResponse)
    return response


def _login(
    database: pathlib.Path,
    username: str,
    password: str,
) -> str:
    """Create one server-side token for an API-request test."""
    store = AuthStore(database)
    return store.create_session(username)


def test_profile_api_requires_a_valid_session(tmp_path, monkeypatch) -> None:
    """Deny profile access when the browser has no standing authentication."""
    server, server_url = _start_empty_server(tmp_path, monkeypatch)
    try:
        with pytest.raises(urllib.error.HTTPError) as raised:
            _request(f"{server_url}/api/auth/profile", token=None)
        assert raised.value.status == 401
        assert json.loads(raised.value.read().decode("utf-8"))["message"] == "请先登录"
    finally:
        server.shutdown()
        server.server_close()


def test_profile_api_returns_current_administrator_profile(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the authenticated profile through the public local endpoint."""
    server, server_url, store = _start_auth_server(tmp_path, monkeypatch, "operator", "secure-admin-password")
    token = store.create_session("operator")
    try:
        response = _request(f"{server_url}/api/auth/profile", token=token)
        assert response.status == 200
        profile = json.loads(response.read().decode("utf-8"))
        assert profile == {
            "username": "operator",
            "display_name": None,
            "email": None,
            "avatar_url": None,
        }
    finally:
        server.shutdown()
        server.server_close()


def test_profile_api_updates_profile_fields(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist profile changes submitted through the authenticated endpoint."""
    server, server_url, store = _start_auth_server(tmp_path, monkeypatch, "operator", "secure-admin-password")
    token = store.create_session("operator")
    try:
        response = _request(
            f"{server_url}/api/auth/profile",
            body={"display_name": "运维负责人", "email": "operator@example.com", "avatar_url": None},
            token=token,
            method="PUT",
        )
        assert response.status == 200
        profile = json.loads(response.read().decode("utf-8"))
        assert profile == {
            "username": "operator",
            "display_name": "运维负责人",
            "email": "operator@example.com",
            "avatar_url": None,
        }
    finally:
        server.shutdown()
        server.server_close()


def _request(url: str, body: dict[str, str | None] | None = None, token: str | None = None, method: str = "GET"):
    """Send one loopback API request without persistent cookies."""
    headers = {}
    if token is not None:
        headers["Cookie"] = f"benchmark_session={token}"
    if body is not None:
        headers["Content-Type"] = "application/json"
    response = urllib.request.urlopen(
        urllib.request.Request(
            url,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers=headers,
            method=method,
        )
    )
    assert isinstance(response, HTTPResponse)
    return response


def _login(server_url: str, username: str, password: str) -> str:
    """Authenticate against the loopback API and extract the cookie token."""
    response = _request(
        f"{server_url}/api/auth/session",
        body={"username": username, "password": password},
        method="POST",
    )
    assert response.status == 200
    cookie_header = response.headers.get("Set-Cookie", "")
    return cookie_header.split(";")[0].removeprefix("benchmark_session=")
