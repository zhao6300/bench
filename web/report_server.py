"""Serve the static report UI and files under ``web/runs``."""

from __future__ import annotations

import argparse
import gzip
import datetime as dt
import http.server
import http.cookies
import ipaddress
import json
import pathlib
import typing
from urllib.parse import unquote, urlsplit

from web.auth_store import AuthStore, SESSION_COOKIE_NAME, SESSION_TTL_SECONDS


WEB_ROOT = pathlib.Path(__file__).resolve().parent
STATIC_ROOT = WEB_ROOT / "ui" / "dist"
RUNS_ROOT = WEB_ROOT / "runs"


def run_files() -> list[pathlib.Path]:
    """Return report files sorted by modified time, newest first.

    Returns:
        A list of report paths whose names do not start with ``.``, sorted by
        modification time from newest to oldest. Name order breaks modification
        time ties so the result remains stable.
    """
    if not RUNS_ROOT.exists():
        return []
    return sorted(
        (path for path in RUNS_ROOT.iterdir() if path.is_file() and path.suffix.lower() == ".json" and not path.name.startswith(".")),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )


def run_metadata() -> list[dict[str, typing.Any]]:
    """Build safe metadata about available report files.

    Returns:
        A JSON-compatible list containing the file name, byte size, and
        modification timestamp. Absolute file paths are intentionally omitted,
        so a public response does not expose filesystem layout.
    """
    output: list[dict[str, typing.Any]] = []
    for path in run_files():
        stat = path.stat()
        modified_at = stat.st_mtime
        output.append(
            {
                "filename": path.name,
                "size_bytes": stat.st_size,
                "modified_at": dt.datetime.fromtimestamp(modified_at, dt.timezone.utc).isoformat(),
            }
        )
    return output


def resolve_under_root(relative: str, root: pathlib.Path) -> typing.Optional[pathlib.Path]:
    """Resolve a web-relative path without allowing traversal outside ``root``.

    Args:
        relative: Raw URL path relative to the server root.
        root: Directory that forms the trust boundary.

    Returns:
        The file path, if it is immediately inside ``root`` or inside one of its
        subdirectories; otherwise ``None``.

    Raises:
        ValueError: If the URL path is missing, absolute, or malformed.
    """
    if not isinstance(relative, str) or not relative.startswith("/"):
        raise ValueError("URL path must begin with /")
    parsed = urlsplit(relative)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("URL path must not contain a scheme, network location, query, or fragment")
    candidate = pathlib.Path(unquote(parsed.path.lstrip("/")))
    if candidate.is_absolute() or candidate == pathlib.Path():
        raise ValueError("URL path must identify a file")
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        return None
    return resolved


class ReportRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for local report discovery and static pages."""

    server_version: str = "BenchmarkReportServer/1.0"
    auth_store: AuthStore

    def do_GET(self) -> None:
        """Return the runs metadata, report JSON, or a static file."""
        try:
            if self.path in {"/api/runs", "/api/reports"}:
                if self._username() is None:
                    self._send_error(401, "请先登录")
                    return
                self._send_bytes(200, "application/json; charset=utf-8", json.dumps(run_metadata()).encode("utf-8"))
                return
            if self.path == "/api/auth/session":
                username = self._username()
                if username is None:
                    self._send_bytes(
                        401,
                        "application/json; charset=utf-8",
                        json.dumps({"authenticated": False}).encode("utf-8"),
                    )
                else:
                    self._send_bytes(
                        200,
                        "application/json; charset=utf-8",
                        json.dumps({"authenticated": True, "username": username}).encode("utf-8"),
                )
                return
            if self.path == "/api/auth/profile":
                username = self._username()
                if username is None:
                    self._send_error(401, "请先登录")
                    return
                self._send_bytes(
                    200,
                    "application/json; charset=utf-8",
                    json.dumps(self.auth_store.get_profile(username)).encode("utf-8"),
                )
                return
            if self.path in {"/reports", "/reports/"}:
                if self._username() is None:
                    self._send_redirect("/login.html")
                    return
                request_path = "/index.html"
            elif self.path in {"/admin/details", "/admin/password", "/admin/avatar"}:
                if self._username() is None:
                    self._send_redirect("/login.html")
                    return
                request_path = "/index.html"
            elif self.path in {"/", "/index.html"}:
                if self._username() is not None:
                    self._send_redirect("/reports/")
                    return
                request_path = "/login.html"
            else:
                request_path = self.path
            content_type = self.static_content_type(request_path)
            if content_type is None:
                self._send_error(404, "not found")
                return
            if request_path.startswith("/runs/"):
                file_path = resolve_under_root(request_path[len("/runs"):], RUNS_ROOT)
            else:
                file_path = resolve_under_root(request_path, STATIC_ROOT)
            if file_path is None or not file_path.is_file():
                self._send_error(404, "not found")
                return
            self._send_bytes(200, content_type, file_path.read_bytes())
        except (OSError, ValueError):
            self._send_error(400, "invalid URL path")

    def do_POST(self) -> None:
        """Authenticate a browser and create its session cookie."""
        try:
            if self.path != "/api/auth/session":
                self._send_error(404, "not found")
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length < 2 or length > 4096:
                raise ValueError("invalid request length")
            credentials = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(credentials, dict):
                raise ValueError("invalid request body")
            username = str(credentials.get("username") or "")
            password = str(credentials.get("password") or "")
            if not self.auth_store.authenticate(username, password):
                self._send_error(401, "账号或密码错误")
                return
            token = self.auth_store.create_session(username)
            self._send_bytes(
                200,
                "application/json; charset=utf-8",
                json.dumps({"authenticated": True, "username": username}).encode("utf-8"),
                extra_headers=self._session_cookie(token),
            )
        except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            self._send_error(400, "登录请求无效")

    def do_PUT(self) -> None:
        """Update the current administrator profile or password."""
        username = self._username()
        if username is None:
            self._send_error(401, "请先登录")
            return
        try:
            if self.path == "/api/auth/profile":
                profile = self._read_json_body(1024 * 1024)
                updated = self.auth_store.update_profile(
                    username,
                    display_name=profile.get("display_name"),
                    email=profile.get("email"),
                    avatar_url=profile.get("avatar_url"),
                )
                self._send_bytes(
                    200,
                    "application/json; charset=utf-8",
                    json.dumps(updated).encode("utf-8"),
                )
                return
            if self.path == "/api/auth/password":
                passwords = self._read_json_body(4096)
                current_password = passwords.get("current_password")
                new_password = passwords.get("new_password")
                if not self.auth_store.change_password(
                    username,
                    str(current_password or ""),
                    str(new_password or ""),
                    keep_session_token=self.session_token(),
                ):
                    self._send_error(401, "当前密码不正确")
                    return
                self._send_bytes(
                    200,
                    "application/json; charset=utf-8",
                    json.dumps({"password_changed": True}).encode("utf-8"),
                )
                return
            self._send_error(404, "not found")
        except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError) as failure:
            if isinstance(failure, (OSError, UnicodeDecodeError, json.JSONDecodeError)):
                self._send_error(400, "请求无效")
                return
            self._send_error(400, str(failure) or "请求无效")

    @staticmethod
    def static_content_type(path: str) -> typing.Optional[str]:
        """Map a server path to its supported media type.

        Args:
            path: The request URL path.

        Returns:
            A media type, or ``None`` if the requested file extension is not
            supported.
        """
        if path in {"", "/", "/index.html"}:
            return "text/html; charset=utf-8"
        suffixes = {
            ".css": "text/css; charset=utf-8",
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
        }
        return suffixes.get(pathlib.PurePath(path).suffix.lower())

    def log_message(self, format_string: str, *args: typing.Any) -> None:
        """Write safe request metadata instead of large report contents."""
        del format_string, args

    def _read_json_body(self, maximum_length: int) -> dict[str, typing.Any]:
        """Decode one JSON request body.

        Args:
            maximum_length: The maximum accepted body bytes.

        Returns:
            The JSON object sent by the browser.

        Raises:
            ValueError: If the request exceeds its size limit or is not a JSON object.
        """
        length = int(self.headers.get("Content-Length", "0"))
        if length < 2 or length > maximum_length:
            raise ValueError("请求内容无效")
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("请求内容无效")
        return payload

    def do_DELETE(self) -> None:
        """Remove the browser's server-side session."""
        if self.path != "/api/auth/session":
            self._send_error(404, "not found")
            return
        token = self.session_token()
        self.auth_store.delete_session(token)
        self._send_bytes(
            200,
            "application/json; charset=utf-8",
            b'{"authenticated": false}',
            extra_headers=self._session_cookie("", max_age=0),
        )

    def session_token(self) -> str:
        """Read the raw session cookie token.

        Returns:
            The cookie token, or an empty string when missing or malformed.
        """
        cookie = http.cookies.SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except http.cookies.CookieError:
            return ""
        field = cookie.get(SESSION_COOKIE_NAME)
        return str(field.value) if field is not None else ""

    def _username(self) -> typing.Optional[str]:
        """Resolve the authenticated username for the request.

        Returns:
            The username, or ``None`` when the session is missing or invalid.
        """
        token = self.session_token()
        username = self.auth_store.username_for_session(token)
        if username is None and token:
            self.auth_store.delete_session(token)
        return username

    def _session_cookie(
        self,
        token: str,
        max_age: int = SESSION_TTL_SECONDS,
    ) -> list[tuple[str, str]]:
        """Build the HttpOnly authentication cookie.

        Args:
            token: The new raw token, or an empty string when clearing.
            max_age: Cookie lifetime in seconds.

        Returns:
            Headers to append to the response.
        """
        expires = "Thu, 01 Jan 1970 00:00:00 GMT" if not token else "Sat, 01 Jan 2100 00:00:00 GMT"
        return [
            (
                "Set-Cookie",
                f"{SESSION_COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}; Expires={expires}",
            )
        ]

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        body: bytes,
        extra_headers: typing.Optional[list[tuple[str, str]]] = None,
    ) -> None:
        compressed = False
        accepted = self.headers.get("Accept-Encoding", "")
        if "gzip" in accepted and "gzip" not in self.headers.get("Content-Encoding", ""):
            body = gzip.compress(body, compresslevel=6)
            compressed = True
        self.send_response(status)
        for name, value in extra_headers or []:
            self.send_header(name, value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Vary", "Accept-Encoding")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        body = json.dumps({"message": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_redirect(self, target: str) -> None:
        """Redirect a browser path to another app path.

        Args:
            target: The target URL path.
        """
        self.send_response(302)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()


def create_local_server(address: tuple[str, int], auth_store: AuthStore) -> http.server.ThreadingHTTPServer:
    """Create an HTTP server bound to the given loopback-safe address.

    Args:
        address: Host and port pair passed to the server.
        auth_store: The SQLite-backed authentication helper.

    Returns:
        A thread-per-request server configured to reuse the address and shut
        down cleanly after Ctrl-C.
    """
    ReportRequestHandler.auth_store = auth_store
    return http.server.ThreadingHTTPServer(address, ReportRequestHandler)


def parse_args() -> argparse.Namespace:
    """Parse report server command-line arguments.

    Returns:
        The validated host, port, and optional listener/print flags.
    """
    parser = argparse.ArgumentParser(description="启动基准报告查看器（默认只绑定 loopback）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅允许 loopback，例如 127.0.0.1 或 ::1")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    parser.add_argument(
        "--database",
        default=str(WEB_ROOT / "data" / "auth.sqlite3"),
        help="登录鉴权 SQLite 数据库路径",
    )
    parser.add_argument("--print-address", action="store_true", help="启动后打印可打开的 URL")
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
            help="允许绑定非 loopback 地址；服务仍要求登录，仅在受信任网络中启用",
    )
    parsed = parser.parse_args()
    address = ipaddress.ip_address(parsed.host)
    if not parsed.allow_non_loopback and not address.is_loopback:
        raise ValueError("only loopback hosts are allowed")
    return parsed


def main() -> int:
    """Attach the report handler to a local socket and wait until interrupted.

    Returns:
        The process exit code. ``130`` is returned after Ctrl-C.
    """
    args = parse_args()
    ReportRequestHandler.auth_store = AuthStore(pathlib.Path(args.database).expanduser())
    server = create_local_server((args.host, args.port), ReportRequestHandler.auth_store)
    if args.print_address:
        print(f"http://{args.host}:{args.port}", flush=True)
    exit_code = 0
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        server.server_close()
    return exit_code
