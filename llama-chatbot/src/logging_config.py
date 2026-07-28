"""Application-wide logging configuration.

Usage
-----
    from src.logging_config import setup_logging

    setup_logging("INFO")
    logger = logging.getLogger(__name__)
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Final

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOGGER_NAME: Final[str] = "llama_chatbot"
_ROOT_LOGGER_NAME: Final[str] = "root"
_HANDLER_NAME: Final[str] = "llama_chatbot_console"
_REDACT_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    # Catch common credential patterns: API keys, tokens, auth headers, etc.
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*['\"]?)[^\s'\"#]+"), r"\1***REDACTED***"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*['\"]?)[^\s'\"#]+"), r"\1***REDACTED***"),
    (re.compile(r"(?i)(bearer\s+)[a-z0-9._-]+"), r"\1***REDACTED***"),
]

_LOGGED_REDACTION_WARNING: bool = False


# ---------------------------------------------------------------------------
# Redaction filter (secondary safeguard — never rely on this alone)
# ---------------------------------------------------------------------------
class SecretsRedactionFilter(logging.Filter):
    """A best-effort filter that redacts common credential patterns.

    .. caution::
       This is an **extra safeguard** and must never be the primary mechanism
       for keeping secrets out of logs.  Application code must *never* pass
       secrets to logging calls in the first place.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact the message itself
        if record.msg and isinstance(record.msg, str):
            for pattern, replacement in _REDACT_PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)

        # Redact positional arguments that are strings
        if record.args:
            sanitised_args: list[object] = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in _REDACT_PATTERNS:
                        arg = pattern.sub(replacement, arg)
                sanitised_args.append(arg)
            record.args = tuple(sanitised_args)  # type: ignore[assignment]

        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    """Configure the application logger.

    This function is **idempotent**: calling it multiple times will not add
    duplicate handlers.  Third-party SDK loggers (``groq``, ``ollama``,
    ``httpx``, ``urllib3``) are set to ``WARNING`` to reduce noise.

    Parameters
    ----------
    level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
    """
    app_logger = logging.getLogger(_LOGGER_NAME)

    # ── Idempotency guard ─────────────────────────────────────────────
    if any(h.name == _HANDLER_NAME for h in app_logger.handlers if hasattr(h, "name")):
        # Already configured — just update the level in case it changed
        app_logger.setLevel(level.upper())
        return

    # ── Level ─────────────────────────────────────────────────────────
    resolved_level: int = getattr(logging, level.upper(), logging.INFO)
    app_logger.setLevel(resolved_level)
    app_logger.propagate = False

    # ── Console handler ───────────────────────────────────────────────
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(resolved_level)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    # Attach redaction filter as a secondary safeguard
    handler.addFilter(SecretsRedactionFilter())

    app_logger.addHandler(handler)

    # ── Quiet noisy third-party loggers ──────────────────────────────
    for noisy in ("groq", "ollama", "httpx", "urllib3", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Log the setup (this message itself is safe)
    app_logger.debug("Logging configured at level %s", level.upper())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of the application namespace.

    Parameters
    ----------
    name:
        Sub-namespace, e.g. ``"providers.groq"``.  If *None*, returns the
        root app logger.
    """
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)
