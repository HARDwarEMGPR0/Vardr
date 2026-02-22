"""Aggregator module for collecting snapshots from all configured venues."""

from __future__ import annotations

import logging
from typing import Any

from .fetch_kalshi import fetch_kalshi_snapshot
from .fetch_polymarket import fetch_polymarket_snapshots

LOGGER = logging.getLogger(__name__)


def fetch_all(polymarket_top_n: int = 5) -> dict[str, Any]:
    """Fetch normalized data from all venues.

    Args:
        polymarket_top_n: Number of Polymarket records to include.

    Returns:
        Dictionary keyed by venue containing normalized snapshots.
    """

    LOGGER.info("Fetching data from all venues")
    return {
        "polymarket": fetch_polymarket_snapshots(top_n=polymarket_top_n),
        "kalshi": fetch_kalshi_snapshot(),
    }
