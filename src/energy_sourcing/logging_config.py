"""Single place to configure logging for CLIs and services."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once; safe to call repeatedly."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Third-party libraries are chatty at INFO level.
    for noisy in ("urllib3", "httpx", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
