"""Process-wide logging configuration helpers."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-scoped logger with a consistent, structured format.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.
    level:
        Logging level for this specific logger (root handler is configured once).
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _CONFIGURED = True
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
