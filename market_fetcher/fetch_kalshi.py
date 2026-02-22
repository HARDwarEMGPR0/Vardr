"""Kalshi data fetcher module."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

from .config import get_settings

LOGGER = logging.getLogger(__name__)
BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _get_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fetch JSON from Kalshi API with timeout and error handling."""

    settings = get_settings()
    url = f"{BASE}{path}"
    response = requests.get(url, params=params, timeout=settings.request_timeout)
    LOGGER.debug("GET %s | %s", response.url, response.status_code)
    response.raise_for_status()
    return response.json()


def _book_sides(orderbook_response: dict[str, Any]) -> tuple[list[list[int]], list[list[int]]]:
    """Extract yes/no sides from a Kalshi orderbook response."""

    orderbook = orderbook_response.get("orderbook") or {}
    yes = orderbook.get("yes") or []
    no = orderbook.get("no") or []
    return yes, no


def _depth_top_n(levels: list[list[int]], top_n: int = 5) -> int:
    """Return cumulative quantity from top N orderbook levels."""

    return sum(quantity for _, quantity in levels[:top_n]) if levels else 0


def fetch_kalshi_snapshot() -> dict[str, Any]:
    """Pick a Kalshi market with strong demo signal and return a normalized snapshot."""

    settings = get_settings()
    markets = _get_json("/markets", params={"limit": settings.kalshi_limit, "status": "open"}).get(
        "markets", []
    )
    if not markets:
        raise RuntimeError("Kalshi returned no open markets.")

    LOGGER.info("Kalshi open markets fetched: %s", len(markets))

    best: tuple[tuple[int, int, int, int], str, dict[str, Any], dict[str, Any]] | None = None

    for market in markets:
        ticker = market["ticker"]
        orderbook = _get_json(f"/markets/{ticker}/orderbook")
        trades = _get_json("/markets/trades", params={"ticker": ticker, "limit": 50})

        yes_levels, no_levels = _book_sides(orderbook)
        trades_list = trades.get("trades", [])
        trade_count = len(trades_list)
        two_sided = int(bool(yes_levels) and bool(no_levels))
        depth_yes = _depth_top_n(yes_levels)
        depth_no = _depth_top_n(no_levels)
        total_depth = depth_yes + depth_no

        score = (int(trade_count > 0), two_sided, total_depth, trade_count)

        if best is None or score > best[0]:
            best = (score, ticker, orderbook, trades)

        if score[0] == 1 and score[1] == 1 and total_depth > 200:
            break

    if best is None:
        raise RuntimeError("Unable to select a Kalshi market.")

    score, ticker, orderbook, trades = best
    yes_levels, no_levels = _book_sides(orderbook)

    LOGGER.info("Selected Kalshi ticker: %s with score=%s", ticker, score)

    return {
        "venue": "kalshi",
        "ticker": ticker,
        "score": {
            "has_trades": score[0],
            "two_sided": score[1],
            "total_depth_top5": score[2],
            "trade_count": score[3],
        },
        "ts": datetime.now(timezone.utc).isoformat(),
        "best_yes_bid": yes_levels[0][0] if yes_levels else None,
        "best_no_bid": no_levels[0][0] if no_levels else None,
        "depth_yes_top5": _depth_top_n(yes_levels),
        "depth_no_top5": _depth_top_n(no_levels),
        "trades_sample_n": len(trades.get("trades", [])),
    }
