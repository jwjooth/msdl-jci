"""Project-wide logging configuration."""

import logging
import sys

_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logger with formatted stream handler."""
    root = logging.getLogger()
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(level)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Obtain logger instance by name."""
    return logging.getLogger(name)
