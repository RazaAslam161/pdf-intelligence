"""Structured application logging (A7).

A small wrapper over the stdlib so every module logs through one namespaced
logger (``ragpdf.*``) whose level honors the ``LOG_LEVEL`` env var. We do not
hijack the root logger: under uvicorn/Streamlit the host already configures
handlers and our records propagate to them; for a bare ``python`` run we install
a basic handler only if nothing else has.
"""

from __future__ import annotations

import logging
import os

_ROOT_NAME = "ragpdf"


def configure_logging() -> None:
    """Configure the application logger once, honoring LOG_LEVEL (default INFO)."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger(_ROOT_NAME).setLevel(level)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced application logger (configures logging on first use)."""
    configure_logging()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
