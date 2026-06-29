"""Structured application logging.

Thin wrapper over the stdlib so everything logs through one namespaced logger
(``ragpdf.*``) at ``LOG_LEVEL``. We avoid touching the root logger when a host
like uvicorn/Streamlit already set up handlers, and only install a basic one for
bare ``python`` runs.
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
