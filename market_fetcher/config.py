"""Runtime configuration helpers for the market fetcher project."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    polymarket_limit: int = int(os.getenv("POLYMARKET_LIMIT", "50"))
    kalshi_limit: int = int(os.getenv("KALSHI_LIMIT", "200"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()


def get_settings() -> Settings:
    """Return parsed application settings."""

    return Settings()
