from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from market_fetcher.vardr1_client import fetch_vardr1_leader_markets, map_vardr1_market_to_leader

LOGGER = logging.getLogger(__name__)


def _normalize_leader_market(market: dict, source: str) -> dict:
    if source == "vardr1":
        return map_vardr1_market_to_leader(market)

    return {
        "title": market.get("title") or market.get("question") or "",
        "market_key": (
            market.get("market_key")
            or market.get("id")
            or market.get("conditionId")
            or market.get("ticker")
            or market.get("slug")
            or ""
        ),
        "reference_event": market.get("reference_event"),
        "mid": market.get("mid"),
        "computed_mid_delta": market.get("computed_mid_delta"),
        "one_hour_price_change": market.get("one_hour_price_change"),
        "one_day_price_change": market.get("one_day_price_change"),
        "source": market.get("source") or source,
    }


def load_leader_markets(source: str | None = None) -> list[dict]:
    """Return a list of leader-market dicts.

    source=None  → empty list (no external leaders configured)
    source=path  → load and return the list from the JSON file at that path
    """
    if source is None:
        return []

    path = Path(source)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Leader markets file must contain a JSON array, got {type(data).__name__}")
    return data


def load_leader_markets_from_vardr_api(url: str, timeout_seconds: float = 10.0) -> list[dict]:
    """Load and normalize leader markets from a Vardr API endpoint.

    Network and decoding failures are treated as unavailable external input and
    return an empty list. Invalid successful response shapes raise ValueError so
    callers can surface a configuration/data-contract problem clearly.
    """
    try:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
        LOGGER.warning("failed to load Vardr leader markets from %s: %s", url, exc)
        return []

    if isinstance(data, list):
        raw_markets = data
    elif isinstance(data, dict):
        raw_markets = None
        for key in ("leader_markets", "markets", "results"):
            value = data.get(key)
            if value is not None:
                raw_markets = value
                break
        if raw_markets is None:
            raise ValueError(
                "Vardr leader API response dict must contain one of: "
                "leader_markets, markets, results"
            )
    else:
        raise ValueError(
            f"Vardr leader API response must be a JSON array or object, got {type(data).__name__}"
        )

    if not isinstance(raw_markets, list):
        raise ValueError(
            f"Vardr leader API markets field must be a JSON array, got {type(raw_markets).__name__}"
        )

    normalized = []
    for item in raw_markets:
        if not isinstance(item, dict):
            raise ValueError(
                f"Vardr leader API market entries must be objects, got {type(item).__name__}"
            )
        normalized.append(_normalize_leader_market(item, source="vardr_api"))
    return normalized


def load_leader_markets_from_vardr1(
    base_url: str | None = None,
    window: str = "24h",
    limit: int = 25,
    timeout_seconds: float = 10.0,
) -> list[dict]:
    return fetch_vardr1_leader_markets(
        base_url=base_url,
        window=window,
        limit=limit,
        timeout_seconds=timeout_seconds,
    )
