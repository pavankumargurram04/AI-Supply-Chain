"""Structured logging configuration.

Provides:
- `configure_logging()` — call once at process start.
- `get_logger(name)` — return a logger with a stable interface.
- `JsonFormatter` — JSON log lines suitable for aggregation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

from core.config import get_settings


_RESERVED_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """Minimal, dependency-free JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # Include any custom extras set via logger.info("...", extra={"k": v})
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except TypeError:
                    payload[key] = repr(value)

        return json.dumps(payload, default=str)


class PlainFormatter(logging.Formatter):
    DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)-40s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.DEFAULT_FMT, datefmt="%Y-%m-%d %H:%M:%S")


_configured = False


def configure_logging() -> None:
    """Configure the root logger. Idempotent."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    if settings.LOG_FORMAT.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(PlainFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Tame overly chatty libraries.
    for noisy in ("uvicorn.access", "urllib3", "asyncio", "botocore"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger; configures logging on first call."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)