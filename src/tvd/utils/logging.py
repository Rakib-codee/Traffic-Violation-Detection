"""Standard logging setup for all tvd entry points."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a module logger, configuring the root handler once."""
    global _configured
    if not _configured:
        logging.basicConfig(format=_FORMAT, level=level)
        _configured = True
    return logging.getLogger(name)
