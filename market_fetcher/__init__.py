"""Market fetcher package for Polymarket and Kalshi data collection."""

from .fetch_data import fetch_all
from .fetch_kalshi import fetch_kalshi_snapshot
from .fetch_polymarket import fetch_polymarket_snapshots

__all__ = [
    "fetch_all",
    "fetch_kalshi_snapshot",
    "fetch_polymarket_snapshots",
]
