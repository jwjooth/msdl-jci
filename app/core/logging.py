"""Centralised logging configuration."""

import logging
import sys
from typing import Union

_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: Union[int, str] = logging.INFO) -> None:
    """Configure root logger with a consistent format.

    Safe to call multiple times – subsequent calls update the existing handlers.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove stale handlers to avoid duplicates on re-import
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that inherits the root configuration."""
    return logging.getLogger(name)
