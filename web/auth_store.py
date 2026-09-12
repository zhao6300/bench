"""SQLite-backed authentication and sessions for the report UI."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = WEB_ROOT / "data" / "auth.sqlite3"
ADMINS_TABLE = "admin_users"
SESSIONS_TABLE = "web_sessions"
SESSION_TTL_SECONDS = 60 * 60 * 12
SESSION_COOKIE_NAME = "benchmark_session"

_SCRYPT_BYTES = 64
_SCRYPT_N = 16384
_SCRYPT_R = 8
_SCRYPT_P = 1
_MIN_PASSWORD_LENGTH = 8
_VALID_USERNAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _normalize_username(value: object) -> str:
    """Normalize a username and reject unsafe values.

    Args:
        value: Untrusted username value.

    Returns:
        The normalized username.

    Raises:
        ValueError: If ``value`` is outside the supported username syntax.
    """
    username = str(value or "").strip()
    if not _VALID_USERNAME.fullmatch(username):
        raise ValueError("用户名只能包含 1-64 个字母、数字、点、下划线或连字符")
    return username


def _normalize_password(value: object) -> str:
    """Normalize a password and enforce the minimum length.

    Args:
        value: Untrusted password value.

    Returns:
        The normalized password.

    Raises:
        ValueError: If ``value`` is too short.
    """
    password = str(value or "")
    if len(password.encode("utf-8")) < _MIN_PASSWORD_LENGTH:
        raise ValueError("密码长度至少为 8 个字符")
    return password


def _hash_password(password: str) -> str:
    """Hash a password using the standard-library scrypt implementation.

    Args:
        password: Plain password supplied by the user.

    Returns:
        A serialized password hash including its unique random salt.
    """
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_BYTES,
    )
    return "$".join(
        (
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            salt.hex(),
            derived.hex(),
        )
    )


def _hash_session_token(token: str) -> str:
    """Hash a random session token before persisting it.

    Args:
        token: Cookie token, invalid if leaked from the database.

    Returns:
        The SHA-256 representation used in the sessions table.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def connect(database: str | Path) -> sqlite3.Connection:
    """Open an initialized SQLite authentication connection.

    Args:
        database: Database path. ``:memory:`` is supported for tests.

    Returns:
        The prepared connection.
    """
    connection = sqlite3.connect(database, timeout=5)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(
    database: str | Path = DEFAULT_DATABASE,
    admin_username: str | None = None,
    admin_password: str | None = None,
) -> None:
    """Create the auth schema and first administrator when needed.

    Args:
        database: SQLite path used by the report server.
        admin_username: First administrator name; defaults to ``admin``.
        admin_password: First administrator password. Required when no user exists.

    Raises:
        ValueError: If the first administrator credentials are invalid.
        OSError: If the SQLite file cannot be created.
    """
    maybe_username = admin_username
    if not maybe_username:
        maybe_username = os.getenv("BENCHMARK_WEB_ADMIN_USERNAME")
    path = Path(database)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.parent.resolve().chmod(0o700)
        except (OSError, NotImplementedError):
            pass

    with connect(database) as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {ADMINS_TABLE} (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SESSIONS_TABLE} (
                token_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL REFERENCES {ADMINS_TABLE}(username) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS sessions_expiry ON web_sessions(expires_at)")
        existing = connection.execute(
            f"SELECT username FROM {ADMINS_TABLE} LIMIT 1"
        ).fetchone()
        if existing is None:
            username = _normalize_username(maybe_username or "admin")
            password = admin_password or os.getenv("BENCHMARK_WEB_ADMIN_PASSWORD", "")
            password = _normalize_password(password)
            connection.execute(
                f"INSERT INTO {ADMINS_TABLE} (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, _hash_password(password), int(time.time())),
            )

    if str(path) != ":memory:":
        try:
            path.resolve().chmod(0o600)
        except (OSError, NotImplementedError):
            pass


def authenticate(database: str | Path, username: str, password: str) -> bool:
    """Verify a login against the persistent admin account.

    Args:
        database: SQLite path used by the report server.
        username: Plain username supplied by the browser.
        password: Plain password supplied by the browser.

    Returns:
        ``True`` only when both the username and password are correct.
    """
    unsafe_username = str(username or "")
    unsafe_password = str(password or "")
    if not _VALID_USERNAME.fullmatch(unsafe_username):
        return False
    with connect(database) as connection:
        row = connection.execute(
            f"SELECT username, password_hash FROM {ADMINS_TABLE} WHERE username = ? LIMIT 1",
            (unsafe_username,),
        ).fetchone()
    if row is None:
        return False
    parts = str(row["password_hash"]).split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        return False
    try:
        n, r, p = (int(value) for value in parts[1:4])
        salt = bytes.fromhex(parts[4])
    except (ValueError, OverflowError):
        return False
    derived = hashlib.scrypt(unsafe_password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=_SCRYPT_BYTES)
    return hmac.compare_digest(derived, bytes.fromhex(parts[5]))


def create_session(database: str | Path, username: str) -> str:
    """Create a random server-side session for an administrator.

    Args:
        database: SQLite path used by the report server.
        username: The authenticated administrator username.

    Returns:
        The random token to send in an HttpOnly cookie.

    Raises:
        ValueError: If the username is invalid.
    """
    normalized = _normalize_username(username)
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    with connect(database) as connection:
        connection.execute("DELETE FROM web_sessions WHERE expires_at <= ?", (now,))
        connection.execute(
            f"""
            INSERT INTO {SESSIONS_TABLE} (token_hash, username, created_at, expires_at)
            SELECT ?, username, ?, ? FROM {ADMINS_TABLE} WHERE username = ?
            """,
            (_hash_session_token(token), now, now + SESSION_TTL_SECONDS, normalized),
        )
    return token


def username_for_session(database: str | Path, token: str) -> str | None:
    """Resolve and refresh an HttpOnly authentication cookie.

    Args:
        database: SQLite path used by the report server.
        token: Raw session cookie token.

    Returns:
        The username, or ``None`` when the token is absent or expired.
    """
    raw_token = str(token or "")
    if not raw_token:
        return None
    token_hash = _hash_session_token(raw_token)
    now = int(time.time())
    with connect(database) as connection:
        row = connection.execute(
            f"""
            SELECT username, expires_at
            FROM {SESSIONS_TABLE}
            WHERE token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        if row is None or int(row["expires_at"]) <= now:
            return None
        connection.execute(
            f"UPDATE {SESSIONS_TABLE} SET expires_at = ? WHERE token_hash = ?",
            (now + SESSION_TTL_SECONDS, token_hash),
        )
        return str(row["username"])


def delete_session(database: str | Path, token: str) -> None:
    """Remove one session token from the database.

    Args:
        database: SQLite path used by the report server.
        token: Raw session cookie token.
    """
    with connect(database) as connection:
        connection.execute(
            f"DELETE FROM {SESSIONS_TABLE} WHERE token_hash = ?",
            (_hash_session_token(str(token or "")),),
        )


class AuthStore:
    """Database-backed helper used by the request handler."""

    def __init__(self, database: str | Path):
        """Initialize an auth helper.

        Args:
            database: SQLite path used by the report server.
        """
        self.database = database
        initialize_database(
            self.database,
            os.getenv("BENCHMARK_WEB_ADMIN_USERNAME"),
            os.getenv("BENCHMARK_WEB_ADMIN_PASSWORD"),
        )

    def authenticate(self, username: str, password: str) -> bool:
        """Verify login credentials.

        Args:
            username: Plain username supplied by the browser.
            password: Plain password supplied by the browser.

        Returns:
            ``True`` if the credentials are valid.
        """
        return authenticate(self.database, username, password)

    def create_session(self, username: str) -> str:
        """Create a persistent session token.

        Args:
            username: The authenticated administrator username.

        Returns:
            The random token to send in an HttpOnly cookie.
        """
        return create_session(self.database, username)

    def username_for_session(self, token: str) -> str | None:
        """Resolve and refresh an authentication cookie.

        Args:
            token: Raw session cookie token.

        Returns:
            The username, or ``None`` when the token is absent or expired.
        """
        return username_for_session(self.database, token)

    def delete_session(self, token: str) -> None:
        """Remove one authentication cookie token.

        Args:
            token: Raw session cookie token.
        """
        delete_session(self.database, token)


def main() -> int:
    """Initialize the SQLite database for the report UI.

    Returns:
        Zero after successful initialization.

    Raises:
        SystemExit: With 2 for invalid arguments.
    """
    parser = argparse.ArgumentParser(description="初始化 Web 报告鉴权 SQLite 数据库")
    parser.add_argument("--database", default=str(DEFAULT_DATABASE), help="SQLite 数据库路径")
    arguments = parser.parse_args()
    AuthStore(arguments.database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
