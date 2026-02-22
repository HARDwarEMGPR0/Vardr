"""Polymarket connector using Gamma discovery + CLOB orderbook selection."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import requests

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
DEFAULT_TIMEOUT = (3.05, 10)
MAX_RETRIES = 2
POLY_FIXTURE_FILE = "polymarket_markets_open.json"

LoadFixtureFn = Callable[[Path], Any]
SaveFixtureFn = Callable[[Path, Any], None]
ClassifyNetworkErrorFn = Callable[[Exception], bool]
LOGGER = logging.getLogger(__name__)


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float:
    parsed = _to_float_or_none(value)
    return parsed if parsed is not None else 0.0


def _default_load_fixture(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_save_fixture_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _default_classify_network_error(exc: Exception) -> bool:
    return isinstance(exc, requests.RequestException)


def _request_json(url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, timeout: tuple[float, float] = DEFAULT_TIMEOUT) -> Any:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 404:
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise
            last_exc = exc
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
        if attempt >= MAX_RETRIES:
            break
        time.sleep(0.2 * (attempt + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError("request failed without exception")


def _extract_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        return [event for event in payload["events"] if isinstance(event, dict)]
    return []


def _extract_event_markets(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    for event in events:
        event_markets = event.get("markets")
        if not isinstance(event_markets, list):
            continue
        for market in event_markets:
            if not isinstance(market, dict):
                continue
            if market.get("active") is not True or market.get("closed") is True:
                continue
            merged = dict(market)
            merged.setdefault("event_title", event.get("title"))
            merged.setdefault("event_slug", event.get("slug"))
            markets.append(merged)
    return markets


def _parse_token_ids(market: dict[str, Any]) -> tuple[str | None, str | None]:
    raw = market.get("clobTokenIds")
    if raw is None:
        return None, None

    ids: list[Any]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
        ids = parsed if isinstance(parsed, list) else []
    elif isinstance(raw, list):
        ids = raw
    else:
        return None, None

    yes = str(ids[0]) if len(ids) > 0 else None
    no = str(ids[1]) if len(ids) > 1 else None
    return yes, no


def _book_headers() -> dict[str, str]:
    api_key = os.getenv("POLYMARKET_API_KEY")
    if not api_key:
        return {}

    # Some deployments accept bearer auth while others use API key headers.
    return {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,
    }


def _parse_levels(levels: Any) -> list[tuple[float, float]]:
    if not isinstance(levels, list):
        return []

    parsed: list[tuple[float, float]] = []
    for level in levels:
        if isinstance(level, dict):
            price = _to_float_or_none(level.get("price"))
            size = _to_float_or_none(level.get("size") or level.get("quantity") or level.get("qty"))
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price = _to_float_or_none(level[0])
            size = _to_float_or_none(level[1])
        else:
            continue
        if price is None or size is None:
            continue
        parsed.append((price, size))
    return parsed


def _extract_book_metrics(book: dict[str, Any]) -> tuple[float | None, float | None, float, bool]:
    bids = _parse_levels(book.get("bids"))
    asks = _parse_levels(book.get("asks"))
    has_depth = bool(bids or asks)

    best_bid = max((price for price, _ in bids), default=None)
    best_ask = min((price for price, _ in asks), default=None)

    top_bids = sorted(bids, key=lambda item: item[0], reverse=True)[:5]
    depth_top5_bids = sum(size for _, size in top_bids)

    return best_bid, best_ask, depth_top5_bids, has_depth


def _is_valid_spread(spread: float | None) -> bool:
    return spread is not None and 0.0 <= spread <= 1.0


def fetch_events_live(limit: int = 200, offset: int = 0, timeout: tuple[float, float] = DEFAULT_TIMEOUT) -> list[dict[str, Any]]:
    params = {
        "active": "true",
        "closed": "false",
        "limit": max(1, int(limit)),
        "offset": max(0, int(offset)),
    }
    payload = _request_json(GAMMA_EVENTS_URL, params=params, timeout=timeout)
    return _extract_events(payload)


def fetch_book_live(token_id: str, timeout: tuple[float, float] = DEFAULT_TIMEOUT) -> dict[str, Any]:
    payload = _request_json(CLOB_BOOK_URL, params={"token_id": token_id}, headers=_book_headers(), timeout=timeout)
    return payload if isinstance(payload, dict) else {}


def _process_candidate(
    market: dict[str, Any],
    timeout: tuple[float, float],
    exclusion_reasons: Counter[str],
) -> dict[str, Any] | None:
    yes_token, no_token = _parse_token_ids(market)
    if not yes_token or not no_token:
        exclusion_reasons["missing_clob_token_ids"] += 1
        return None

    try:
        yes_book = fetch_book_live(yes_token, timeout=timeout)
        no_book = fetch_book_live(no_token, timeout=timeout)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            LOGGER.info("Polymarket skip %s: CLOB book 404", market.get("slug") or market.get("id"))
            exclusion_reasons["book_404"] += 1
            return None
        raise

    yes_bid, yes_ask, depth_yes, yes_has_depth = _extract_book_metrics(yes_book)
    no_bid, no_ask, depth_no, no_has_depth = _extract_book_metrics(no_book)

    if not yes_has_depth or not no_has_depth:
        LOGGER.info("Polymarket skip %s: empty book sides", market.get("slug") or market.get("id"))
        exclusion_reasons["empty_book"] += 1
        return None

    if yes_bid is None or yes_ask is None or no_bid is None or no_ask is None:
        exclusion_reasons["missing_bid_or_ask"] += 1
        return None

    spread_yes = yes_ask - yes_bid
    spread_no = no_ask - no_bid

    if not _is_valid_spread(spread_yes) or not _is_valid_spread(spread_no):
        exclusion_reasons["invalid_spread"] += 1
        return None

    enriched = dict(market)
    enriched.update(
        {
            "best_yes_bid": yes_bid,
            "best_yes_ask": yes_ask,
            "best_no_bid": no_bid,
            "best_no_ask": no_ask,
            "spread_yes": spread_yes,
            "spread_no": spread_no,
            "spread": spread_yes,
            "spread_reported": spread_yes,
            "mid": (yes_bid + yes_ask) / 2.0,
            "depth_yes_top5": depth_yes,
            "depth_no_top5": depth_no,
            "depth_top5": depth_yes + depth_no,
        }
    )
    return enriched


def _rank_markets(markets: list[dict[str, Any]], n_markets: int) -> list[dict[str, Any]]:
    ranked = sorted(
        markets,
        key=lambda market: (
            _to_float(market.get("depth_top5")),
            _to_float(market.get("liquidityNum") or market.get("liquidity_proxy")),
            _to_float(market.get("volume24hr") or market.get("volume_proxy")),
            market.get("updatedAt") or "",
        ),
        reverse=True,
    )
    return ranked[: max(1, int(n_markets))]


def fetch_markets_live(
    limit: int = 200,
    offset: int = 0,
    n_markets: int = 5,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    max_market_scans: int = 40,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = fetch_events_live(limit=limit, offset=offset, timeout=timeout)
    candidates = _extract_event_markets(events)

    exclusion_reasons: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []

    scanned = 0
    for market in candidates:
        if scanned >= max_market_scans:
            break
        scanned += 1
        processed = _process_candidate(market, timeout=timeout, exclusion_reasons=exclusion_reasons)
        if processed is not None:
            eligible.append(processed)

    selected = _rank_markets(eligible, n_markets=n_markets)
    debug = {
        "fetched_candidates": scanned,
        "eligible_two_sided": len(eligible),
        "selected": len(selected),
        "excluded": max(0, scanned - len(eligible)),
        "exclusion_reasons": dict(exclusion_reasons),
    }
    return selected, debug


def fetch_markets_fixture(fixtures_dir: str | Path = "fixtures", loader: LoadFixtureFn | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    load_fixture = loader or _default_load_fixture
    payload = load_fixture(Path(fixtures_dir) / POLY_FIXTURE_FILE)

    if isinstance(payload, list):
        markets = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("markets"), list):
        markets = [item for item in payload["markets"] if isinstance(item, dict)]
    else:
        markets = []

    debug = {
        "fetched_candidates": len(markets),
        "eligible_two_sided": len(markets),
        "selected": len(markets),
        "excluded": 0,
        "exclusion_reasons": {},
    }
    return markets, debug


def fetch_markets(
    mode: str = "live",
    allow_fallback: bool = True,
    fixtures_dir: str | Path = "fixtures",
    save_fixtures: bool = False,
    limit: int = 200,
    offset: int = 0,
    n_markets: int = 5,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    max_market_scans: int = 40,
    loader: LoadFixtureFn | None = None,
    saver: SaveFixtureFn | None = None,
    classify_network_error: ClassifyNetworkErrorFn | None = None,
) -> tuple[list[dict[str, Any]], str, str | None, dict[str, Any]]:
    load_fixture = loader or _default_load_fixture
    save_fixture_atomic = saver or _default_save_fixture_atomic
    is_network_error = classify_network_error or _default_classify_network_error

    if mode == "fixture":
        try:
            markets, debug = fetch_markets_fixture(fixtures_dir=fixtures_dir, loader=load_fixture)
            selected = _rank_markets(markets, n_markets=n_markets)
            debug["selected"] = len(selected)
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
        selected, debug = fetch_markets_live(
            limit=limit,
            offset=offset,
            n_markets=n_markets,
            timeout=timeout,
            max_market_scans=max_market_scans,
        )
        if save_fixtures:
            save_fixture_atomic(Path(fixtures_dir) / POLY_FIXTURE_FILE, selected)
        return selected, "live", None, debug
    except Exception as exc:
        live_error = f"live_error: {exc}"
        if allow_fallback and is_network_error(exc):
            try:
                markets, debug = fetch_markets_fixture(fixtures_dir=fixtures_dir, loader=load_fixture)
                selected = _rank_markets(markets, n_markets=n_markets)
                debug["selected"] = len(selected)
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


def fetch_polymarket_snapshot_by_condition_id(
    condition_id: str,
    mode: str = "live",
    fixtures_dir: str | Path = "fixtures",
    limit: int = 400,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    loader: LoadFixtureFn | None = None,
) -> dict[str, Any] | None:
    """Fetch a single tradable Polymarket snapshot by condition id."""

    if mode == "fixture":
        markets, _debug = fetch_markets_fixture(fixtures_dir=fixtures_dir, loader=loader)
        target = next((m for m in markets if str(m.get("conditionId") or m.get("id")) == str(condition_id)), None)
        return target

    events = fetch_events_live(limit=limit, offset=0, timeout=timeout)
    candidates = _extract_event_markets(events)
    target_market = next(
        (market for market in candidates if str(market.get("conditionId") or market.get("id")) == str(condition_id)),
        None,
    )
    if target_market is None:
        return None

    exclusion_reasons: Counter[str] = Counter()
    return _process_candidate(target_market, timeout=timeout, exclusion_reasons=exclusion_reasons)
