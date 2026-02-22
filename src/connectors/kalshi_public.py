"""Kalshi connector with robust two-sided selection and fallback support."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_TIMEOUT = (3.05, 10)
MAX_RETRIES = 2
KALSHI_MARKETS_FIXTURE_FILE = "kalshi_markets_open.json"
KALSHI_ORDERBOOK_FIXTURE_FILE = "kalshi_orderbook.json"
KALSHI_TRADES_FIXTURE_FILE = "kalshi_trades.json"
KALSHI_DETAILS_FIXTURE_FILE = "kalshi_market_details.json"

LoadFixtureFn = Callable[[Path], Any]
SaveFixtureFn = Callable[[Path, Any], None]
ClassifyNetworkErrorFn = Callable[[Exception], bool]


def _default_load_fixture(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_save_fixture_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _default_classify_network_error(exc: Exception) -> bool:
    return isinstance(exc, requests.RequestException)


def _request_json(url: str, *, params: dict[str, Any] | None = None, timeout: tuple[float, float] = DEFAULT_TIMEOUT) -> Any:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(0.2 * (attempt + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError("request failed without exception")


def _safe_levels(levels: Any) -> list[list[float]]:
    if not isinstance(levels, list):
        return []

    parsed: list[list[float]] = []
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        try:
            price = float(level[0])
            qty = float(level[1])
        except (TypeError, ValueError):
            continue
        parsed.append([price, qty])
    return parsed


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_markets_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        markets = payload.get("markets", [])
        return markets if isinstance(markets, list) else []
    if isinstance(payload, list):
        return payload
    return []


def _extract_by_ticker(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("by_ticker"), dict):
        return {str(k): v for k, v in payload["by_ticker"].items() if isinstance(v, dict)}
    return {}


def depth_top5_from_orderbook(orderbook_json: dict[str, Any]) -> tuple[float, float, float]:
    orderbook = orderbook_json.get("orderbook") if isinstance(orderbook_json, dict) else {}
    if not isinstance(orderbook, dict):
        orderbook = {}

    yes_levels = _safe_levels(orderbook.get("yes"))
    no_levels = _safe_levels(orderbook.get("no"))

    depth_yes = sum(level[1] for level in yes_levels[:5])
    depth_no = sum(level[1] for level in no_levels[:5])
    return depth_yes, depth_no, depth_yes + depth_no


def fetch_open_markets_live(
    limit: int = 200,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    payload = _request_json(
        f"{BASE_URL}/markets",
        params={"status": "open", "limit": max(1, int(limit))},
        timeout=timeout,
    )
    return _extract_markets_from_payload(payload)


def fetch_market_detail_live(
    ticker: str,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    payload = _request_json(f"{BASE_URL}/markets/{ticker}", timeout=timeout)
    return payload if isinstance(payload, dict) else {}


def fetch_orderbook_live(
    ticker: str,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    payload = _request_json(f"{BASE_URL}/markets/{ticker}/orderbook", timeout=timeout)
    return payload if isinstance(payload, dict) else {}


def fetch_trades_live(
    ticker: str,
    limit: int = 50,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    payload = _request_json(
        f"{BASE_URL}/markets/trades",
        params={"ticker": ticker, "limit": max(1, int(limit))},
        timeout=timeout,
    )
    return payload if isinstance(payload, dict) else {}


def _is_excluded_mve(ticker: str, include_mve: bool) -> bool:
    return (not include_mve) and ticker.startswith("KXMVESPORTSMULTIGAMEEXTENDED")


def _detail_quote(detail: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _to_float_or_none(detail.get(key))
        if value is not None:
            return value
    return None


def _evaluate_market(
    market: dict[str, Any],
    detail: dict[str, Any],
    orderbook: dict[str, Any],
    trades: dict[str, Any],
    exclusions: Counter[str],
) -> dict[str, Any] | None:
    ticker = str(market.get("ticker", ""))
    if not ticker:
        exclusions["missing_ticker"] += 1
        return None

    orderbook_root = orderbook.get("orderbook") if isinstance(orderbook, dict) else {}
    if not isinstance(orderbook_root, dict):
        orderbook_root = {}

    yes_levels = _safe_levels(orderbook_root.get("yes"))
    no_levels = _safe_levels(orderbook_root.get("no"))

    # Prefer explicit quote fields from market detail when present.
    best_yes_bid = _detail_quote(detail, ("yes_bid", "best_yes_bid", "yesBid"))
    best_yes_ask = _detail_quote(detail, ("yes_ask", "best_yes_ask", "yesAsk"))
    best_no_bid = _detail_quote(detail, ("no_bid", "best_no_bid", "noBid"))
    best_no_ask = _detail_quote(detail, ("no_ask", "best_no_ask", "noAsk"))

    if best_yes_bid is None and yes_levels:
        best_yes_bid = yes_levels[0][0]
    if best_no_bid is None and no_levels:
        best_no_bid = no_levels[0][0]

    if best_yes_ask is None and best_no_bid is not None:
        best_yes_ask = 100.0 - best_no_bid
    if best_no_ask is None and best_yes_bid is not None:
        best_no_ask = 100.0 - best_yes_bid

    spread = None
    if best_yes_bid is not None and best_yes_ask is not None:
        spread = best_yes_ask - best_yes_bid

    depth_yes, depth_no, depth_total = depth_top5_from_orderbook(orderbook)
    trades_list = trades.get("trades") if isinstance(trades, dict) else []
    if not isinstance(trades_list, list):
        trades_list = []

    is_two_sided = (
        best_yes_bid is not None
        and best_yes_ask is not None
        and best_no_bid is not None
        and best_no_ask is not None
        and spread is not None
    )
    if not is_two_sided:
        exclusions["not_two_sided"] += 1
        return None

    if spread < 0:
        exclusions["negative_spread"] += 1
        return None

    open_interest = detail.get("open_interest") or detail.get("openInterest") or market.get("open_interest") or 0
    try:
        open_interest_float = float(open_interest)
    except (TypeError, ValueError):
        open_interest_float = 0.0

    enriched = dict(market)
    enriched.update(detail)
    enriched.update(
        {
            "ticker": ticker,
            "best_yes_bid": best_yes_bid,
            "best_yes_ask": best_yes_ask,
            "best_no_bid": best_no_bid,
            "best_no_ask": best_no_ask,
            "spread": spread,
            "depth_yes_top5": depth_yes,
            "depth_no_top5": depth_no,
            "depth_top5": depth_total,
            "trades_count_sample": len(trades_list),
            "open_interest_value": open_interest_float,
            "orderbook": orderbook,
            "trades": trades,
        }
    )
    return enriched


def _rank(markets: list[dict[str, Any]], n_markets: int) -> list[dict[str, Any]]:
    ranked = sorted(
        markets,
        key=lambda m: (
            float(m.get("depth_top5") or 0.0),
            int(m.get("trades_count_sample") or 0),
            float(m.get("open_interest_value") or 0.0),
        ),
        reverse=True,
    )
    return ranked[: max(1, int(n_markets))]


def fetch_markets_live(
    limit: int = 200,
    n_markets: int = 5,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    include_mve: bool = False,
    max_market_scans: int = 25,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    markets = fetch_open_markets_live(limit=limit, timeout=timeout)
    exclusions: Counter[str] = Counter()

    by_ticker_orderbook: dict[str, dict[str, Any]] = {}
    by_ticker_trades: dict[str, dict[str, Any]] = {}
    by_ticker_detail: dict[str, dict[str, Any]] = {}

    eligible: list[dict[str, Any]] = []
    scanned = 0
    for market in markets:
        if scanned >= max_market_scans:
            break
        scanned += 1
        ticker = str(market.get("ticker", ""))
        if not ticker:
            exclusions["missing_ticker"] += 1
            continue
        if _is_excluded_mve(ticker, include_mve=include_mve):
            exclusions["excluded_mve_bundle"] += 1
            continue

        detail = fetch_market_detail_live(ticker, timeout=timeout)
        orderbook = fetch_orderbook_live(ticker, timeout=timeout)
        trades = fetch_trades_live(ticker, timeout=timeout)

        by_ticker_detail[ticker] = detail
        by_ticker_orderbook[ticker] = orderbook
        by_ticker_trades[ticker] = trades

        enriched = _evaluate_market(market, detail, orderbook, trades, exclusions)
        if enriched is not None:
            eligible.append(enriched)

    selected = _rank(eligible, n_markets=n_markets)
    debug = {
        "fetched_candidates": scanned,
        "eligible_two_sided": len(eligible),
        "selected": len(selected),
        "excluded": max(0, scanned - len(eligible)),
        "exclusion_reasons": dict(exclusions),
    }

    raw = {
        "markets": markets,
        "details": by_ticker_detail,
        "orderbooks": by_ticker_orderbook,
        "trades": by_ticker_trades,
    }
    return selected, debug, raw, by_ticker_detail


def fetch_markets_fixture(
    fixtures_dir: str | Path = "fixtures",
    n_markets: int = 5,
    loader: LoadFixtureFn | None = None,
    include_mve: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    load_fixture = loader or _default_load_fixture

    markets_payload = load_fixture(Path(fixtures_dir) / KALSHI_MARKETS_FIXTURE_FILE)
    orderbook_payload = load_fixture(Path(fixtures_dir) / KALSHI_ORDERBOOK_FIXTURE_FILE)
    trades_payload = load_fixture(Path(fixtures_dir) / KALSHI_TRADES_FIXTURE_FILE)

    details_payload: dict[str, Any] = {}
    details_path = Path(fixtures_dir) / KALSHI_DETAILS_FIXTURE_FILE
    if details_path.exists():
        details_payload = load_fixture(details_path)

    markets = _extract_markets_from_payload(markets_payload)
    orderbook_map = _extract_by_ticker(orderbook_payload)
    trades_map = _extract_by_ticker(trades_payload)
    details_map = _extract_by_ticker(details_payload)

    exclusions: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []

    for market in markets:
        ticker = str(market.get("ticker", ""))
        if not ticker:
            exclusions["missing_ticker"] += 1
            continue
        if _is_excluded_mve(ticker, include_mve=include_mve):
            exclusions["excluded_mve_bundle"] += 1
            continue

        detail = details_map.get(ticker, {})
        orderbook = orderbook_map.get(ticker)
        if orderbook is None and isinstance(orderbook_payload, dict) and str(orderbook_payload.get("ticker")) == ticker:
            orderbook = orderbook_payload
        orderbook = orderbook or {"ticker": ticker, "orderbook": {"yes": [], "no": []}}

        trades = trades_map.get(ticker)
        if trades is None and isinstance(trades_payload, dict) and str(trades_payload.get("ticker")) == ticker:
            trades = trades_payload
        trades = trades or {"ticker": ticker, "trades": []}

        enriched = _evaluate_market(market, detail, orderbook, trades, exclusions)
        if enriched is not None:
            eligible.append(enriched)

    selected = _rank(eligible, n_markets=n_markets)
    debug = {
        "fetched_candidates": len(markets),
        "eligible_two_sided": len(eligible),
        "selected": len(selected),
        "excluded": len(markets) - len(eligible),
        "exclusion_reasons": dict(exclusions),
    }
    raw = {
        "markets": markets_payload,
        "details": details_payload,
        "orderbooks": orderbook_payload,
        "trades": trades_payload,
    }
    return selected, debug, raw, details_map


def fetch_markets(
    mode: str = "live",
    allow_fallback: bool = True,
    fixtures_dir: str | Path = "fixtures",
    save_fixtures: bool = False,
    limit: int = 200,
    n_markets: int = 5,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    include_mve: bool = False,
    max_market_scans: int = 25,
    loader: LoadFixtureFn | None = None,
    saver: SaveFixtureFn | None = None,
    classify_network_error: ClassifyNetworkErrorFn | None = None,
) -> tuple[list[dict[str, Any]], str, str | None, dict[str, Any]]:
    load_fixture = loader or _default_load_fixture
    save_fixture_atomic = saver or _default_save_fixture_atomic
    is_network_error = classify_network_error or _default_classify_network_error

    if mode == "fixture":
        try:
            selected, debug, _raw, _details = fetch_markets_fixture(
                fixtures_dir=fixtures_dir,
                n_markets=n_markets,
                loader=load_fixture,
                include_mve=include_mve,
            )
            return selected, "fixture", None, debug
        except Exception as exc:  # pragma: no cover
            return [], "none", f"fixture_error: {exc}", {
                "fetched_candidates": 0,
                "eligible_two_sided": 0,
                "selected": 0,
                "excluded": 0,
                "exclusion_reasons": {"fixture_error": 1},
            }

    try:
        selected, debug, raw, details = fetch_markets_live(
            limit=limit,
            n_markets=n_markets,
            timeout=timeout,
            include_mve=include_mve,
            max_market_scans=max_market_scans,
        )
        if save_fixtures:
            save_fixture_atomic(Path(fixtures_dir) / KALSHI_MARKETS_FIXTURE_FILE, raw["markets"])
            save_fixture_atomic(Path(fixtures_dir) / KALSHI_ORDERBOOK_FIXTURE_FILE, {"by_ticker": raw["orderbooks"]})
            save_fixture_atomic(Path(fixtures_dir) / KALSHI_TRADES_FIXTURE_FILE, {"by_ticker": raw["trades"]})
            save_fixture_atomic(Path(fixtures_dir) / KALSHI_DETAILS_FIXTURE_FILE, {"by_ticker": details})
        return selected, "live", None, debug
    except Exception as exc:
        live_error = f"live_error: {exc}"
        if allow_fallback and is_network_error(exc):
            try:
                selected, debug, _raw, _details = fetch_markets_fixture(
                    fixtures_dir=fixtures_dir,
                    n_markets=n_markets,
                    loader=load_fixture,
                    include_mve=include_mve,
                )
                return selected, "fixture", live_error, debug
            except Exception as fixture_exc:
                return [], "none", f"{live_error} | fixture_error: {fixture_exc}", {
                    "fetched_candidates": 0,
                    "eligible_two_sided": 0,
                    "selected": 0,
                    "excluded": 0,
                    "exclusion_reasons": {"live_and_fixture_error": 1},
                }
        return [], "none", live_error, {
            "fetched_candidates": 0,
            "eligible_two_sided": 0,
            "selected": 0,
            "excluded": 0,
            "exclusion_reasons": {"live_error": 1},
        }


def fetch_kalshi_snapshot_by_ticker(
    ticker: str,
    mode: str = "live",
    fixtures_dir: str | Path = "fixtures",
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    loader: LoadFixtureFn | None = None,
) -> dict[str, Any] | None:
    """Fetch a single Kalshi enriched market snapshot by ticker."""

    load_fixture = loader or _default_load_fixture
    exclusions: Counter[str] = Counter()

    if mode == "fixture":
        markets_payload = load_fixture(Path(fixtures_dir) / KALSHI_MARKETS_FIXTURE_FILE)
        orderbook_payload = load_fixture(Path(fixtures_dir) / KALSHI_ORDERBOOK_FIXTURE_FILE)
        trades_payload = load_fixture(Path(fixtures_dir) / KALSHI_TRADES_FIXTURE_FILE)

        details_payload: dict[str, Any] = {}
        details_path = Path(fixtures_dir) / KALSHI_DETAILS_FIXTURE_FILE
        if details_path.exists():
            details_payload = load_fixture(details_path)

        markets = _extract_markets_from_payload(markets_payload)
        market = next((m for m in markets if str(m.get("ticker")) == str(ticker)), None)
        if market is None:
            return None

        orderbook_map = _extract_by_ticker(orderbook_payload)
        trades_map = _extract_by_ticker(trades_payload)
        details_map = _extract_by_ticker(details_payload)
        orderbook = orderbook_map.get(str(ticker), {"ticker": ticker, "orderbook": {"yes": [], "no": []}})
        trades = trades_map.get(str(ticker), {"ticker": ticker, "trades": []})
        detail = details_map.get(str(ticker), {})
        return _evaluate_market(market, detail, orderbook, trades, exclusions)

    detail = fetch_market_detail_live(ticker, timeout=timeout)
    orderbook = fetch_orderbook_live(ticker, timeout=timeout)
    trades = fetch_trades_live(ticker, timeout=timeout)
    market = dict(detail)
    market.setdefault("ticker", ticker)
    return _evaluate_market(market, detail, orderbook, trades, exclusions)
