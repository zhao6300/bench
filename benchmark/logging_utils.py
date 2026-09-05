"""统一的 benchmark 调试日志配置。"""

from __future__ import annotations

import logging
import sys
from urllib.parse import urlsplit

_LOGGER_NAME = "llm_benchmark"


class _DynamicStderr:
    """将日志写入当前的 ``sys.stderr``，兼容 Rich 运行时重定向。"""

    @property
    def encoding(self) -> str:
        """返回当前 stderr 的编码。"""
        return getattr(sys.stderr, "encoding", None) or "utf-8"

    def write(self, message: str) -> int:
        """将消息写入当前 stderr。"""
        return sys.stderr.write(message)

    def flush(self) -> None:
        """刷新当前 stderr。"""
        sys.stderr.flush()

    def isatty(self) -> bool:
        """返回当前 stderr 是否连接到终端。"""
        return sys.stderr.isatty()


def configure_logging(debug: bool = False) -> None:
    """配置 benchmark 的进程内调试日志。

    Args:
        debug: 是否启用 DEBUG 级别日志。
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(_DynamicStderr())
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.CRITICAL + 1)
    logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """返回 benchmark 命名空间下的 logger。

    Args:
        name: 子模块名称；省略时返回根 benchmark logger。

    Returns:
        配置由 ``configure_logging`` 控制的 logger。
    """
    if not name:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def safe_endpoint(url: str) -> str:
    """返回不包含查询参数、用户名或密钥的 endpoint 摘要。

    Args:
        url: 待摘要的 HTTP(S) URL。

    Returns:
        ``host[:port]/path`` 形式的安全摘要。
    """
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or "<unknown-host>"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port is not None:
            host = f"{host}:{port}"
        return f"{host}{parsed.path or '/'}"
    except (TypeError, ValueError):
        return "<invalid-endpoint>"


def exception_type(exc: BaseException) -> str:
    """返回异常类型名称，不暴露异常文本中的 URL 或凭据。

    Args:
        exc: 捕获的异常。

    Returns:
        异常类的限定名称。
    """
    return type(exc).__name__
