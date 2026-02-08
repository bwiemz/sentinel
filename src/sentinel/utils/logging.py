"""Structured logging configuration for SENTINEL.

Uses structlog processors on top of stdlib logging so existing
``logging.getLogger(__name__)`` calls continue to work transparently.
Supports console, JSON, and file output modes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    log_json: bool = False,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file. ``None`` disables file logging.
        log_json: If True, render log lines as JSON instead of human-readable.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # --- structlog shared processors ---
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if log_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # --- stdlib logging handlers ---
    handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    handlers.append(console_handler)

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)

    # --- wire structlog into stdlib ---
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    for h in handlers:
        h.setFormatter(formatter)

    root = logging.getLogger("sentinel")
    root.handlers.clear()
    root.setLevel(numeric_level)
    for h in handlers:
        root.addHandler(h)
    root.propagate = False

    # Quiet down noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
