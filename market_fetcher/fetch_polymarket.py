"""Polymarket data fetcher module."""

from __future__ import annotations

import logging
from typing import Any

import requests

from .config import get_settings

LOGGER = logging.getLogger(__name__)


def as_float(value: Any) -> float:
    """Convert a value to float, returning 0.0 if conversion fails."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """Fetch JSON from a URL with timeout and error handling."""

    settings = get_settings()
    response = requests.get(url, params=params, timeout=settings.request_timeout)
    LOGGER.debug("GET %s | %s", response.url, response.status_code)
    response.raise_for_status()
    return response.json()


def fetch_polymarket_snapshots(top_n: int = 5) -> list[dict[str, Any]]:
    """Fetch top Polymarket markets and normalize them into snapshot records.

    Args:
        top_n: Number of top-ranked markets to return.

    Returns:
        List of normalized market snapshots.
    """

    settings = get_settings()
    markets = _get_json(
        "https://gamma-api.polymarket.com/markets",
        params={
            "active": "true",
            "closed": "false",
            "limit": settings.polymarket_limit,
            "offset": 0,
        },
    )

    LOGGER.info("Polymarket markets returned: %s", len(markets))

    ranked = sorted(
        markets,
        key=lambda m: (
            as_float(m.get("liquidityNum") or 0),
            as_float(m.get("volume24hr") or 0),
            m.get("updatedAt") or "",
        ),
        reverse=True,
    )

    snapshots: list[dict[str, Any]] = []
    for market in ranked[:top_n]:
        snapshots.append(
            {
                "venue": "polymarket",
                "market_id": market.get("conditionId") or market.get("id"),
                "slug": market.get("slug"),
                "question": market.get("question"),
                "endDate": market.get("endDate"),
                "closed": market.get("closed"),
                "active": market.get("active"),
                "bestBid": as_float(market.get("bestBid")),
                "bestAsk": as_float(market.get("bestAsk")),
                "spread": as_float(market.get("spread")),
                "lastTradePrice": as_float(market.get("lastTradePrice")),
                "liquidity": as_float(market.get("liquidityNum")),
                "volume24hr": as_float(market.get("volume24hr")),
                "oneHourPriceChange": as_float(market.get("oneHourPriceChange")),
                "oneDayPriceChange": as_float(market.get("oneDayPriceChange")),
            }
        )

    return snapshots
