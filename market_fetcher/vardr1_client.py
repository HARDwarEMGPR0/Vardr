from __future__ import annotations

import hashlib
import json
import logging
import os
from urllib.parse import urlencode, urljoin

import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_VARDR1_API_BASE_URL = "http://localhost:9002"
VARDR1_MARKET_LEADER_PATH = "/api/suspicious-markets"
VARDR1_ANOMALY_FALLBACK_PATH = "/api/suspicious"
VARDR1_DEFAULT_FALLBACK_LIMIT = 100
_LAST_FETCH_METADATA: dict[str, object] = {}


def vardr1_base_url() -> str:
    return os.getenv("VARDR1_API_BASE_URL", DEFAULT_VARDR1_API_BASE_URL).strip().rstrip("/")


def _fallback_market_key(title: str) -> str:
    digest = hashlib.sha1(title.lower().strip().encode("utf-8")).hexdigest()[:12]
    return f"vardr1_{digest}"


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if coerced != coerced:
        return None
    return coerced


def _coerce_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _first_present(market: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        value = market.get(key)
        if value is not None:
            return value
    return None


def _first_present_with_key(market: dict, keys: tuple[str, ...]) -> tuple[str | None, object | None]:
    for key in keys:
        value = market.get(key)
        if value is not None:
            return key, value
    return None, None


def _derive_price_change(
    market: dict,
    *,
    current_price: float | None,
    previous_price_keys: tuple[str, ...],
) -> tuple[float | None, str | None]:
    if current_price is None:
        return None, None
    previous_key, previous_raw = _first_present_with_key(market, previous_price_keys)
    previous_price = _coerce_float(previous_raw)
    if previous_price is None:
        return None, None
    return current_price - previous_price, f"derived_from_{previous_key}"


def _has_nonzero_movement(leader: dict) -> bool:
    return bool(abs(leader.get("price_change_1h") or 0.0) > 0 or abs(leader.get("price_change_24h") or 0.0) > 0)


def _is_stale_or_resolved(market: dict) -> bool:
    resolved = _coerce_bool(market.get("resolved"))
    closed = _coerce_bool(market.get("closed"))
    active = _coerce_bool(market.get("active"))
    time_to_resolution_hours = _coerce_float(market.get("time_to_resolution_hours"))
    return bool(
        market.get("stale_or_resolved") is True
        or resolved is True
        or closed is True
        or active is False
        or (time_to_resolution_hours is not None and time_to_resolution_hours < 0)
    )


def _dedupe_leaders(leaders: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for leader in leaders:
        market_id = str(leader.get("market_id") or leader.get("market_key") or leader.get("title") or "")
        current = best.get(market_id)
        if current is None:
            best[market_id] = leader
            continue
        leader_move = max(abs(leader.get("price_change_1h") or 0.0), abs(leader.get("price_change_24h") or 0.0))
        current_move = max(abs(current.get("price_change_1h") or 0.0), abs(current.get("price_change_24h") or 0.0))
        leader_key = (
            leader.get("stale_or_resolved") is not True,
            leader_move > 0,
            leader_move,
            float(leader.get("leader_score") or 0.0),
        )
        current_key = (
            current.get("stale_or_resolved") is not True,
            current_move > 0,
            current_move,
            float(current.get("leader_score") or 0.0),
        )
        if leader_key > current_key:
            best[market_id] = leader
    return list(best.values())


def compute_leader_score(market: dict) -> float:
    """Rank leaders by movement magnitude with anomaly/risk support."""

    move_1h = abs(_coerce_float(_first_present(market, ("one_hour_price_change", "price_change_1h"))) or 0.0)
    move_24h = abs(_coerce_float(_first_present(market, ("one_day_price_change", "price_change_24h"))) or 0.0)
    risk_score = _coerce_float(_first_present(market, ("risk_score", "max_risk_score", "risk")))
    anomaly_score = _coerce_float(_first_present(market, ("anomaly_score", "max_raw_risk", "raw_risk", "anomaly")))
    if risk_score is not None or anomaly_score is not None:
        return max(risk_score or 0.0, anomaly_score or 0.0, move_1h, move_24h)

    liquidity = _coerce_float(_first_present(market, ("liquidity_proxy", "liquidity", "liquidityNum", "depth_top5"))) or 0.0
    recency = _coerce_float(_first_present(market, ("recency_score", "freshness_score"))) or 1.0
    liquidity_boost = min(1.0, liquidity / 100_000.0)
    return round((2.0 * move_1h) + move_24h + (0.15 * liquidity_boost) + (0.10 * recency), 6)


def _extract_markets(data: object) -> list[dict]:
    if isinstance(data, list):
        raw_markets = data
    elif isinstance(data, dict):
        raw_markets = None
        for key in ("market_leaders", "leader_markets", "suspicious_markets", "markets", "results", "data", "value"):
            value = data.get(key)
            if value is not None:
                raw_markets = value
                break
        if raw_markets is None:
            raise ValueError(
                "Vardr-1 response dict must contain one of: "
                "market_leaders, leader_markets, suspicious_markets, markets, results, data, value"
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

    title = _first_present(market, ("market_title", "question", "title", "name"))
    if not title:
        return {}

    condition_id = _first_present(market, ("condition_id", "conditionId", "conditionid"))
    clob_token_id = _first_present(
        market,
        ("clob_token_id", "clobTokenId", "clobtokenid", "token_id", "tokenId", "tokenid", "yes_token_id", "yesTokenId"),
    )
    universe_market_id = _first_present(market, ("universe_market_id", "universeMarketId", "gamma_market_id"))
    market_key = _first_present(
        market,
        ("market_key", "market_id", "condition_id", "conditionId", "conditionid", "id", "ticker", "slug", "url"),
    )
    current_price = _coerce_float(
        _first_present(
            market,
            ("current_price", "price", "mid", "probability", "last_price", "lasttradeprice", "lastTradePrice"),
        )
    )
    computed_mid_delta_key, computed_mid_delta_raw = _first_present_with_key(
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
    computed_mid_delta = _coerce_float(computed_mid_delta_raw)
    one_hour_key, one_hour_raw = _first_present_with_key(
        market,
        (
            "price_change_1h",
            "one_hour_price_change",
            "one_hour_change",
            "change_1h",
            "hourly_price_change",
            "price_delta_1h",
            "probability_delta_1h",
        ),
    )
    one_day_key, one_day_raw = _first_present_with_key(
        market,
        (
            "price_change_24h",
            "one_day_price_change",
            "one_day_change",
            "change_24h",
            "daily_price_change",
            "price_delta_24h",
            "probability_delta_24h",
            "abs_move_24h",
            "absolute_move_24h",
            "abs_price_change_24h",
            "abs_change_24h",
        ),
    )
    one_hour_price_change = _coerce_float(one_hour_raw)
    one_day_price_change = _coerce_float(one_day_raw)
    one_hour_source = one_hour_key
    one_day_source = one_day_key

    if one_hour_price_change is None:
        one_hour_price_change, one_hour_source = _derive_price_change(
            market,
            current_price=current_price,
            previous_price_keys=(
                "price_1h_ago",
                "one_hour_ago_price",
                "previous_price_1h",
                "previous_mid_1h",
                "start_price_1h",
            ),
        )
    if one_day_price_change is None:
        one_day_price_change, one_day_source = _derive_price_change(
            market,
            current_price=current_price,
            previous_price_keys=(
                "price_24h_ago",
                "one_day_ago_price",
                "previous_price_24h",
                "previous_mid_24h",
                "start_price_24h",
                "previous_price",
            ),
        )

    if computed_mid_delta is None:
        computed_mid_delta = one_day_price_change if one_day_price_change is not None else one_hour_price_change

    leader_score_key, leader_score_raw = _first_present_with_key(
        market,
        ("leader_score", "score", "anomaly_score", "max_raw_risk", "raw_risk", "risk_score", "max_risk_score"),
    )
    leader_score = _coerce_float(leader_score_raw)
    if leader_score is None:
        leader_score = compute_leader_score(
            {
                **market,
                "one_hour_price_change": one_hour_price_change,
                "one_day_price_change": one_day_price_change,
            }
        )
    recent_volume = _coerce_float(_first_present(market, ("recent_volume", "trade_size", "volume", "volumeNum")))
    liquidity = _coerce_float(_first_present(market, ("liquidity", "liquidityNum", "liquiditynum", "liquidity_proxy")))
    risk_score = _coerce_float(_first_present(market, ("risk_score", "max_risk_score", "risk")))
    anomaly_score = _coerce_float(_first_present(market, ("anomaly_score", "max_raw_risk", "raw_risk", "anomaly")))
    raw_risk = _coerce_float(_first_present(market, ("raw_risk", "max_raw_risk")))
    real_movement_source = (
        (one_day_source if abs(one_day_price_change or 0.0) > 1e-9 else None)
        or (one_hour_source if abs(one_hour_price_change or 0.0) > 1e-9 else None)
        or (computed_mid_delta_key if abs(computed_mid_delta or 0.0) > 1e-9 else None)
    )
    has_movement_fields = bool(real_movement_source)
    movement_source = real_movement_source
    if movement_source is None and (anomaly_score is not None or risk_score is not None or leader_score is not None):
        movement_source = "anomaly_score"
    leader_score_source = leader_score_key or ("anomaly_score" if anomaly_score is not None else "computed")

    return {
        "market_id": str(market_key) if market_key else _fallback_market_key(str(title)),
        "condition_id": str(condition_id) if condition_id is not None else None,
        "clob_token_id": str(clob_token_id) if clob_token_id is not None else None,
        "universe_market_id": str(universe_market_id) if universe_market_id is not None else None,
        "market_title": str(title),
        "current_price": current_price,
        "price": current_price,
        "price_change_1h": one_hour_price_change if one_hour_price_change is not None else 0.0,
        "price_change_24h": one_day_price_change if one_day_price_change is not None else 0.0,
        "leader_score": leader_score,
        "platform": _first_present(market, ("platform", "venue")),
        "recent_volume": recent_volume,
        "volume": recent_volume,
        "liquidity": liquidity,
        "title": str(title),
        "market_key": str(market_key) if market_key else _fallback_market_key(str(title)),
        "reference_event": _first_present(market, ("reference_event", "referenceEvent")),
        "mid": current_price,
        "computed_mid_delta": computed_mid_delta,
        "one_hour_price_change": one_hour_price_change if one_hour_price_change is not None else 0.0,
        "one_day_price_change": one_day_price_change if one_day_price_change is not None else 0.0,
        "source": "vardr1_market_movement" if has_movement_fields else "vardr1_anomaly_fallback",
        "reason": _first_present(
            market,
            (
                "reason",
                "suspicion_reason",
                "explanation",
                "leader_score",
                "risk_score",
                "max_risk_score",
                "anomaly_score",
                "max_raw_risk",
                "raw_risk",
            ),
        ),
        "risk_score": risk_score,
        "anomaly_score": anomaly_score,
        "raw_risk": raw_risk,
        "movement_source": movement_source,
        "leader_score_source": leader_score_source,
        "active": _coerce_bool(market.get("active")),
        "closed": _coerce_bool(market.get("closed")),
        "resolved": _coerce_bool(market.get("resolved")),
        "end_date": _first_present(market, ("end_date", "endDate", "enddate", "close_time_utc", "resolution_date", "resolutionDate")),
        "resolution_date": _first_present(market, ("resolution_date", "resolutionDate", "end_date", "endDate", "enddate", "close_time_utc")),
        "stale_or_resolved": _is_stale_or_resolved(market),
        "_has_movement_fields": has_movement_fields,
    }


def _raw_field_names(markets: list[dict]) -> list[str]:
    return sorted({str(key) for market in markets[:5] for key in market.keys()})


def _leader_example(leader: dict) -> dict[str, object]:
    fields = (
        "market_id",
        "market_title",
        "current_price",
        "price_change_1h",
        "price_change_24h",
        "leader_score",
        "risk_score",
        "anomaly_score",
        "movement_source",
        "leader_score_source",
        "source",
        "stale_or_resolved",
    )
    return {field: leader.get(field) for field in fields}


def _mapping_skip_examples(raw_markets: list[dict], leaders: list[dict]) -> list[dict[str, object]]:
    if len(leaders) == len(raw_markets):
        return []
    examples: list[dict[str, object]] = []
    for market in raw_markets:
        if map_vardr1_market_to_leader(market):
            continue
        missing_fields = []
        if not _first_present(market, ("market_title", "question", "title", "name")):
            missing_fields.append("title/question/market_title")
        examples.append(
            {
                "reason": "missing_required_leader_fields",
                "missing_fields": missing_fields,
                "raw_fields": sorted(str(key) for key in market.keys()),
            }
        )
        if len(examples) >= 5:
            break
    return examples


def _fetch_metadata(
    url: str,
    raw_markets: list[dict],
    leaders: list[dict],
    *,
    anomaly_fallback_used: bool,
    endpoint_attempts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "leader_source_endpoint": url,
        "leader_rows_raw_loaded": len(raw_markets),
        "market_rows_loaded": len(leaders),
        "market_rows_with_nonzero_movement": sum(1 for leader in leaders if _has_nonzero_movement(leader)),
        "market_rows_filtered_stale_or_resolved": sum(1 for leader in leaders if leader.get("stale_or_resolved") is True),
        "anomaly_fallback_used": anomaly_fallback_used,
        "raw_leader_field_names": _raw_field_names(raw_markets),
        "raw_leader_examples": raw_markets[:3],
        "normalized_leader_examples": [_leader_example(leader) for leader in leaders[:3]],
        "leader_mapping_skip_examples": _mapping_skip_examples(raw_markets, leaders),
        "leader_fetch_endpoint_attempts": endpoint_attempts or [],
    }


def _endpoint_url(base: str, path: str, params: dict[str, object]) -> str:
    url = urljoin(f"{base}/", path.lstrip("/"))
    return f"{url}?{urlencode(params)}" if params else url


def _fetch_vardr1_rows(
    *,
    base: str,
    path: str,
    params: dict[str, object],
    timeout_seconds: float,
) -> tuple[list[dict], str, dict[str, object]]:
    url = urljoin(f"{base}/", path.lstrip("/"))
    endpoint = _endpoint_url(base, path, params)
    LOGGER.warning("requesting Vardr-1 leader markets: %s params=%s", url, params)
    response = requests.get(url, params=params, timeout=timeout_seconds)
    response_text = getattr(response, "text", "")
    attempt: dict[str, object] = {
        "endpoint": endpoint,
        "status_code": getattr(response, "status_code", None),
        "response_length": len(response_text),
        "response_preview": response_text[:500],
    }
    LOGGER.warning("Vardr-1 response status: %s", response.status_code)
    LOGGER.warning("Vardr-1 response body preview: %s", response_text[:500])
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and status_code >= 400:
        attempt["error"] = f"HTTP {status_code}"
        return [], endpoint, attempt
    response.raise_for_status()
    try:
        payload = response.json()
        raw_markets = _extract_markets(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        attempt["error"] = str(exc)
        return [], endpoint, attempt
    attempt["raw_count"] = len(raw_markets)
    LOGGER.warning("Vardr-1 raw leader field names from %s: %s", endpoint, _raw_field_names(raw_markets))
    LOGGER.warning("Vardr-1 raw first 5 leader objects from %s: %s", endpoint, json.dumps(raw_markets[:5], default=str)[:8000])
    return raw_markets, endpoint, attempt


def vardr1_last_fetch_metadata() -> dict[str, object]:
    return dict(_LAST_FETCH_METADATA)


def fetch_vardr1_leader_markets(
    base_url: str | None = None,
    *,
    window: str = "24h",
    limit: int = 25,
    timeout_seconds: float = 10.0,
) -> list[dict]:
    global _LAST_FETCH_METADATA
    base = (base_url or vardr1_base_url()).strip().rstrip("/")
    LOGGER.warning("Vardr-1 base URL: %s", base)
    _LAST_FETCH_METADATA = {
        "leader_source_endpoint": VARDR1_ANOMALY_FALLBACK_PATH,
        "leader_rows_raw_loaded": 0,
        "market_rows_loaded": 0,
        "market_rows_with_nonzero_movement": 0,
        "market_rows_filtered_stale_or_resolved": 0,
        "anomaly_fallback_used": False,
        "raw_leader_field_names": [],
        "raw_leader_examples": [],
        "normalized_leader_examples": [],
        "leader_mapping_skip_examples": [],
        "leader_fetch_endpoint_attempts": [],
        "leader_fetch_error": None,
        "leader_fetch_response_length": 0,
        "leader_fetch_response_preview": "",
    }
    endpoint_attempts: list[dict[str, object]] = []
    endpoint_specs = [
        (
            VARDR1_ANOMALY_FALLBACK_PATH,
            {"window": window, "limit": limit},
            True,
        ),
        (
            VARDR1_MARKET_LEADER_PATH,
            {"window": window, "limit": VARDR1_DEFAULT_FALLBACK_LIMIT},
            False,
        ),
    ]
    last_raw_markets: list[dict] = []
    last_url = _endpoint_url(base, VARDR1_ANOMALY_FALLBACK_PATH, {"window": window, "limit": limit})
    last_leaders: list[dict] = []
    last_error: str | None = None
    for path, params, anomaly_source in endpoint_specs:
        try:
            raw_markets, url, attempt = _fetch_vardr1_rows(
                base=base,
                path=path,
                params=params,
                timeout_seconds=timeout_seconds,
            )
            leaders = _dedupe_leaders([
                {**leader, "source": "vardr1_anomaly_fallback"}
                if anomaly_source and leader.get("source") == "vardr1_anomaly_fallback"
                else leader
                for market in raw_markets
                if (leader := map_vardr1_market_to_leader(market))
            ])
            attempt["normalized_count"] = len(leaders)
            endpoint_attempts.append(attempt)
            last_raw_markets = raw_markets
            last_url = url
            last_leaders = leaders
            LOGGER.warning(
                "Vardr-1 leader endpoint result endpoint=%s raw=%d normalized=%d",
                url,
                len(raw_markets),
                len(leaders),
            )
            if leaders:
                LOGGER.warning("Vardr-1 leader endpoint succeeded: %s", url)
                leaders.sort(key=lambda item: float(item.get("leader_score") or 0.0), reverse=True)
                _LAST_FETCH_METADATA = _fetch_metadata(
                    url,
                    raw_markets,
                    leaders,
                    anomaly_fallback_used=anomaly_source,
                    endpoint_attempts=endpoint_attempts,
                )
                LOGGER.warning("parsed %d Vardr-1 leader markets after mapping", len(leaders))
                return leaders
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            endpoint = _endpoint_url(base, path, params)
            endpoint_attempts.append(
                {
                    "endpoint": endpoint,
                    "status_code": None,
                    "response_length": 0,
                    "response_preview": "",
                    "raw_count": 0,
                    "normalized_count": 0,
                    "error": last_error,
                }
            )
            LOGGER.warning("failed to fetch Vardr-1 leaders from %s: %s", endpoint, exc)

    last_attempt = endpoint_attempts[-1] if endpoint_attempts else {}
    _LAST_FETCH_METADATA = _fetch_metadata(
        last_url,
        last_raw_markets,
        last_leaders,
        anomaly_fallback_used=False,
        endpoint_attempts=endpoint_attempts,
    )
    _LAST_FETCH_METADATA["leader_fetch_error"] = (
        "No Vardr-1 leaders loaded. "
        f"endpoints_tried={[attempt.get('endpoint') for attempt in endpoint_attempts]} "
        f"last_response_length={last_attempt.get('response_length', 0)} "
        f"last_response_preview={last_attempt.get('response_preview', '')}"
    )
    _LAST_FETCH_METADATA["leader_fetch_response_length"] = last_attempt.get("response_length", 0)
    _LAST_FETCH_METADATA["leader_fetch_response_preview"] = last_attempt.get("response_preview", "")
    if last_error:
        _LAST_FETCH_METADATA["leader_fetch_last_exception"] = last_error
    LOGGER.error(_LAST_FETCH_METADATA["leader_fetch_error"])
    return []
