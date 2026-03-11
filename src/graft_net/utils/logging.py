"""Structured JSON logging for training runs."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a logger that emits JSON to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "metrics"):
            payload["metrics"] = record.metrics
        return json.dumps(payload)
