"""Serve the static report UI and files under ``web/runs``."""

from __future__ import annotations

import argparse
import datetime as dt
import http.server
import ipaddress
import json
import pathlib
import typing
from urllib.parse import unquote, urlsplit


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

    def do_GET(self) -> None:
        """Return the runs metadata, report JSON, or a static file."""
        try:
            if self.path in {"/api/runs", "/api/reports"}:
                self._send_bytes(200, "application/json; charset=utf-8", json.dumps(run_metadata()).encode("utf-8"))
                return
            request_path = "/index.html" if self.path == "/" else self.path
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

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
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


def create_local_server(address: tuple[str, int]) -> http.server.ThreadingHTTPServer:
    """Create an HTTP server bound to the given loopback-safe address.

    Args:
        address: Host and port pair passed to the server.

    Returns:
        A thread-per-request server configured to reuse the address and shut
        down cleanly with Ctrl-C.
    """

    return http.server.ThreadingHTTPServer(address, ReportRequestHandler)


def parse_args() -> argparse.Namespace:
    """Parse report server command-line arguments.

    Returns:
        The validated host, port, and optional listener/print flags.
    """
    parser = argparse.ArgumentParser(description="启动基准报告查看器（默认只绑定 loopback）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅允许 loopback，例如 127.0.0.1 或 ::1")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    parser.add_argument("--print-address", action="store_true", help="启动后打印可打开的 URL")
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="允许绑定非 loopback 地址；该服务未鉴权，仅在受信任网络中启用",
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
    server = create_local_server((args.host, args.port))
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
