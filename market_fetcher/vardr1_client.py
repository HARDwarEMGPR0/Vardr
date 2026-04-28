from __future__ import annotations

import hashlib
import json
import logging
import os
from urllib.parse import urljoin

import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_VARDR1_API_BASE_URL = "http://localhost:9002"
VARDR1_LEADER_PATH = "/api/suspicious-markets"


def vardr1_base_url() -> str:
    return os.getenv("VARDR1_API_BASE_URL", DEFAULT_VARDR1_API_BASE_URL).strip().rstrip("/")


def _fallback_market_key(title: str) -> str:
    digest = hashlib.sha1(title.lower().strip().encode("utf-8")).hexdigest()[:12]
    return f"vardr1_{digest}"


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(market: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        value = market.get(key)
        if value is not None:
            return value
    return None


def compute_leader_score(market: dict) -> float:
    """Rank leaders by movement magnitude with light liquidity/recency boosts."""

    move_1h = abs(_coerce_float(_first_present(market, ("one_hour_price_change", "price_change_1h"))) or 0.0)
    move_24h = abs(_coerce_float(_first_present(market, ("one_day_price_change", "price_change_24h"))) or 0.0)
    liquidity = _coerce_float(_first_present(market, ("liquidity_proxy", "liquidity", "liquidityNum", "depth_top5"))) or 0.0
    recency = _coerce_float(_first_present(market, ("recency_score", "freshness_score"))) or 1.0
    liquidity_boost = min(1.0, liquidity / 100_000.0)
    return round((2.0 * move_1h) + move_24h + (0.15 * liquidity_boost) + (0.10 * recency), 6)


def _extract_markets(data: object) -> list[dict]:
    if isinstance(data, list):
        raw_markets = data
    elif isinstance(data, dict):
        raw_markets = None
        for key in ("market_leaders", "leader_markets", "suspicious_markets", "markets", "results", "data"):
            value = data.get(key)
            if value is not None:
                raw_markets = value
                break
        if raw_markets is None:
            raise ValueError(
                "Vardr-1 response dict must contain one of: "
                "market_leaders, leader_markets, suspicious_markets, markets, results, data"
            )
    else:
        raise ValueError(f"Vardr-1 response must be a JSON array or object, got {type(data).__name__}")

    if not isinstance(raw_markets, list):
        raise ValueError(f"Vardr-1 markets field must be a JSON array, got {type(raw_markets).__name__}")
    if not all(isinstance(item, dict) for item in raw_markets):
        raise ValueError("Vardr-1 market entries must be JSON objects")
    return raw_markets


def map_vardr1_market_to_leader(market: dict) -> dict:
    nested_market = market.get("market")
    if isinstance(nested_market, dict):
        market = {**nested_market, **market}

    title = _first_present(market, ("title", "question", "market_title", "name"))
    if not title:
        return {}

    market_key = _first_present(
        market,
        ("market_key", "market_id", "id", "conditionId", "condition_id", "ticker", "slug", "url"),
    )
    computed_mid_delta = _coerce_float(
        _first_present(
            market,
            (
                "computed_mid_delta",
                "mid_delta",
                "price_delta",
                "probability_delta",
                "move",
                "change",
                "suspicious_move",
            ),
        )
    )
    one_hour_price_change = _coerce_float(
        _first_present(market, ("one_hour_price_change", "one_hour_change", "change_1h", "price_change_1h"))
    )
    one_day_price_change = _coerce_float(
        _first_present(market, ("one_day_price_change", "one_day_change", "change_24h", "price_change_24h"))
    )

    if computed_mid_delta is None:
        computed_mid_delta = one_day_price_change if one_day_price_change is not None else one_hour_price_change

    current_price = _coerce_float(
        _first_present(market, ("mid", "current_price", "probability", "price", "last_price"))
    )
    leader_score = _coerce_float(_first_present(market, ("leader_score", "score")))
    if leader_score is None:
        leader_score = compute_leader_score(
            {
                **market,
                "one_hour_price_change": one_hour_price_change,
                "one_day_price_change": one_day_price_change,
            }
        )

    return {
        "market_id": str(market_key) if market_key else _fallback_market_key(str(title)),
        "market_title": str(title),
        "current_price": current_price,
        "price_change_1h": one_hour_price_change,
        "price_change_24h": one_day_price_change,
        "leader_score": leader_score,
        "title": str(title),
        "market_key": str(market_key) if market_key else _fallback_market_key(str(title)),
        "reference_event": _first_present(market, ("reference_event", "referenceEvent")),
        "mid": current_price,
        "computed_mid_delta": computed_mid_delta,
        "one_hour_price_change": one_hour_price_change,
        "one_day_price_change": one_day_price_change,
        "source": "vardr1",
        "reason": _first_present(market, ("reason", "suspicion_reason", "explanation", "leader_score")),
    }


def fetch_vardr1_leader_markets(
    base_url: str | None = None,
    *,
    window: str = "24h",
    limit: int = 25,
    timeout_seconds: float = 10.0,
) -> list[dict]:
    base = (base_url or vardr1_base_url()).strip().rstrip("/")
    url = urljoin(f"{base}/", VARDR1_LEADER_PATH.lstrip("/"))
    LOGGER.warning("Vardr-1 base URL: %s", base)
    LOGGER.warning("requesting Vardr-1 leader markets: %s params=%s", url, {"window": window, "limit": limit})
    try:
        response = requests.get(url, params={"window": window, "limit": limit}, timeout=timeout_seconds)
        LOGGER.warning("Vardr-1 response status: %s", response.status_code)
        LOGGER.warning("Vardr-1 response body preview: %s", getattr(response, "text", "")[:500])
        response.raise_for_status()
        data = response.json()
        raw_markets = _extract_markets(data)
    except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
        LOGGER.warning("failed to fetch Vardr-1 leader markets from %s: %s", url, exc)
        return []

    leaders = [leader for market in raw_markets if (leader := map_vardr1_market_to_leader(market))]
    leaders.sort(key=lambda item: float(item.get("leader_score") or 0.0), reverse=True)
    LOGGER.warning("parsed %d Vardr-1 leader markets after mapping", len(leaders))
    return leaders
