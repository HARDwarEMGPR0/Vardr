"""Centralized logging setup for CLI and modules."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure app-wide logging format and level."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
