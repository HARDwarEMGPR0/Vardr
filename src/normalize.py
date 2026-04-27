"""Normalization helpers for venue snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from market_fetcher.resolution_parser import parse_resolution_metadata


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_title(market: dict[str, Any]) -> tuple[str | None, str | None]:
    title = market.get("title") or market.get("question") or market.get("event_title")
    question = market.get("question") or market.get("title") or market.get("event_title")
    return title, question


def normalize_polymarket_market(market: dict[str, Any], ts_utc: str | None = None) -> dict[str, Any]:
    """Normalize a Polymarket market object into a common snapshot schema."""

    ts = ts_utc or utc_now_iso()

    best_yes_bid = to_float_or_none(market.get("best_yes_bid"))
    best_yes_ask = to_float_or_none(market.get("best_yes_ask"))
    best_no_bid = to_float_or_none(market.get("best_no_bid"))
    best_no_ask = to_float_or_none(market.get("best_no_ask"))

    spread_raw = to_float_or_none(market.get("spread_reported") or market.get("spread"))
    last_trade = to_float_or_none(market.get("lastTradePrice"))

    if best_yes_bid is not None and best_yes_ask is not None:
        mid = (best_yes_bid + best_yes_ask) / 2.0
        spread = best_yes_ask - best_yes_bid
    else:
        mid = to_float_or_none(market.get("mid"))
        if mid is None:
            mid = last_trade
        spread = spread_raw

    title, question = _extract_title(market)

    depth_yes = to_float_or_none(market.get("depth_yes_top5")) or 0.0
    depth_no = to_float_or_none(market.get("depth_no_top5")) or 0.0
    depth_total = to_float_or_none(market.get("depth_top5"))
    if depth_total is None:
        depth_total = depth_yes + depth_no

    snapshot = {
        "ts_utc": ts,
        "venue": "polymarket",
        "market_key": market.get("conditionId") or market.get("id") or market.get("market_key"),
        "title": title,
        "question": question,
        "slug": market.get("slug") or market.get("event_slug"),
        "close_time_utc": market.get("closedTime") or market.get("endDate"),
        "mid": mid,
        "spread": spread,
        "spread_reported": spread_raw if spread_raw is not None else spread,
        "depth_top5": depth_total,
        "depth_yes_top5": depth_yes,
        "depth_no_top5": depth_no,
        "best_yes_bid": best_yes_bid,
        "best_yes_ask": best_yes_ask,
        "best_no_bid": best_no_bid,
        "best_no_ask": best_no_ask,
        "volume_proxy": to_float_or_none(market.get("volume24hr") or market.get("volume_proxy")),
        "liquidity_proxy": to_float_or_none(market.get("liquidityNum") or market.get("liquidity_proxy")) or 0.0,
        "one_hour_price_change": to_float_or_none(market.get("oneHourPriceChange")),
        "one_day_price_change": to_float_or_none(market.get("oneDayPriceChange")),
        "last_trade_price": last_trade,
        "trades_count_sample": None,
        "minute_features": None,
    }
    snapshot["resolution_meta"] = parse_resolution_metadata({**market, **snapshot})
    return snapshot


def _sum_trades_volume(trades_json: dict[str, Any]) -> float | None:
    trades = trades_json.get("trades") if isinstance(trades_json, dict) else []
    if not isinstance(trades, list) or not trades:
        return None

    total = 0.0
    for trade in trades:
        if not isinstance(trade, dict):
            continue

        value = (
            trade.get("count")
            or trade.get("quantity")
            or trade.get("size")
            or trade.get("volume")
            or 0
        )
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue

    if total <= 0:
        return float(len(trades))
    return total


def normalize_kalshi_market(
    ticker: str,
    market_json: dict[str, Any],
    orderbook_json: dict[str, Any],
    trades_json: dict[str, Any],
    ts_utc: str | None = None,
    minute_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize Kalshi market/orderbook/trades payloads into common snapshot schema."""

    ts = ts_utc or utc_now_iso()

    best_yes_bid = to_float_or_none(market_json.get("best_yes_bid"))
    best_no_bid = to_float_or_none(market_json.get("best_no_bid"))

    if best_yes_bid is None:
        orderbook = orderbook_json.get("orderbook") if isinstance(orderbook_json, dict) else {}
        if isinstance(orderbook, dict) and isinstance(orderbook.get("yes"), list) and orderbook.get("yes"):
            best_yes_bid = to_float_or_none(orderbook["yes"][0][0])
    if best_no_bid is None:
        orderbook = orderbook_json.get("orderbook") if isinstance(orderbook_json, dict) else {}
        if isinstance(orderbook, dict) and isinstance(orderbook.get("no"), list) and orderbook.get("no"):
            best_no_bid = to_float_or_none(orderbook["no"][0][0])

    best_yes_ask = (100.0 - best_no_bid) if best_no_bid is not None else to_float_or_none(market_json.get("best_yes_ask"))
    best_no_ask = (100.0 - best_yes_bid) if best_yes_bid is not None else to_float_or_none(market_json.get("best_no_ask"))

    if best_yes_bid is not None and best_yes_ask is not None:
        spread = best_yes_ask - best_yes_bid
        mid = (best_yes_bid + best_yes_ask) / 2.0
    elif best_yes_bid is not None:
        mid = best_yes_bid
        spread = None
    elif best_no_bid is not None:
        mid = 100.0 - best_no_bid
        spread = None
    else:
        mid = None
        spread = None

    depth_yes = to_float_or_none(market_json.get("depth_yes_top5")) or 0.0
    depth_no = to_float_or_none(market_json.get("depth_no_top5")) or 0.0
    depth_total = to_float_or_none(market_json.get("depth_top5"))
    if depth_total is None:
        depth_total = depth_yes + depth_no

    title, question = _extract_title(market_json)

    snapshot = {
        "ts_utc": ts,
        "venue": "kalshi",
        "market_key": ticker,
        "title": title,
        "question": question,
        "slug": market_json.get("ticker") or ticker,
        "close_time_utc": market_json.get("close_time")
        or market_json.get("expiration_time")
        or market_json.get("end_date"),
        "mid": mid,
        "spread": spread,
        "spread_reported": spread,
        "depth_top5": depth_total,
        "depth_yes_top5": depth_yes,
        "depth_no_top5": depth_no,
        "best_yes_bid": best_yes_bid,
        "best_yes_ask": best_yes_ask,
        "best_no_bid": best_no_bid,
        "best_no_ask": best_no_ask,
        "volume_proxy": _sum_trades_volume(trades_json),
        "liquidity_proxy": depth_total,
        "trades_count_sample": len(trades_json.get("trades", [])) if isinstance(trades_json, dict) else 0,
        "one_hour_price_change": None,
        "one_day_price_change": None,
        "last_trade_price": None,
    }

    snapshot["minute_features"] = minute_features
    snapshot["resolution_meta"] = parse_resolution_metadata({**market_json, **snapshot})

    return snapshot
