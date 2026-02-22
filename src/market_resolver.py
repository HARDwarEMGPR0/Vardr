"""Deterministic market resolver for Kalshi and Polymarket."""

from __future__ import annotations

import re
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.connectors.kalshi_public import fetch_kalshi_snapshot_by_ticker, fetch_open_markets_live
from src.connectors.polymarket_gamma import (
    fetch_events_live,
    fetch_polymarket_snapshot_by_condition_id,
)
from src.normalize import normalize_kalshi_market, normalize_polymarket_market
from src.scoring import compute_drivers, compute_execution_risk, compute_regime

STOPWORDS = {"will", "the", "a", "an", "in", "on", "by", "of", "to", "win", "2026"}
CATEGORY_WORDS = {
    "sports": {"super", "bowl", "nfl", "nba", "mlb", "playoffs", "world", "cup", "fifa", "seahawks"},
    "macro": {"cpi", "inflation", "fed", "rates", "fomc", "gdp", "yoy"},
    "crypto": {"btc", "bitcoin", "eth", "ethereum", "crypto"},
    "politics": {"election", "mayor", "president", "senate", "vote"},
}


@dataclass
class BaseDump:
    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TradeIntent(BaseDump):
    query: str
    side: Literal["YES", "NO"]
    notional_usd: float
    trade_id: str | None = None
    category_hint: str | None = None
    event_date: str | None = None
    venue_preference: str | None = None

    def __post_init__(self) -> None:
        if self.side not in {"YES", "NO"}:
            raise ValueError("side must be YES or NO")


@dataclass
class MarketCandidate(BaseDump):
    venue: Literal["kalshi", "polymarket"]
    market_key: str
    title: str | None = None
    slug: str | None = None
    close_time_utc: str | None = None
    liquidity_proxy: float | None = None
    volume_proxy: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedMarket(BaseDump):
    trade_id: str | None
    intent_query: str
    resolved_at_utc: str
    best_kalshi: dict[str, Any] | None
    best_polymarket: dict[str, Any] | None
    match_quality: Literal["HIGH", "MED", "LOW"]
    explanation: list[str]
    candidates: dict[str, list[dict[str, Any]]]


@dataclass
class ResolveResult(BaseDump):
    resolved: ResolvedMarket
    venue_status: dict[str, dict[str, Any]]
    debug: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str | None) -> set[str]:
    if not text:
        return set()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return {t for t in cleaned.split() if t and t not in STOPWORDS}


def extract_entities(text: str | None) -> set[str]:
    tokens = normalize_text(text)
    return {t for t in tokens if len(t) > 2}


def _detect_category(tokens: set[str]) -> str | None:
    for category, words in CATEGORY_WORDS.items():
        if tokens & words:
            return category
    return None


def candidate_score(intent: TradeIntent, candidate: MarketCandidate) -> float:
    q_tokens = normalize_text(intent.query)
    c_tokens = normalize_text(candidate.title)

    if not q_tokens or not c_tokens:
        return 0.0

    jaccard = len(q_tokens & c_tokens) / len(q_tokens | c_tokens)
    entity_bonus = 0.10 if extract_entities(intent.query) & extract_entities(candidate.title) else 0.0

    q_nums = {t for t in q_tokens if t.isdigit()}
    c_nums = {t for t in c_tokens if t.isdigit()}
    number_bonus = 0.10 if q_nums and (q_nums & c_nums) else 0.0

    category_bonus = 0.0
    hint = (intent.category_hint or "").strip().lower()
    if hint and hint in CATEGORY_WORDS and (c_tokens & CATEGORY_WORDS[hint]):
        category_bonus = 0.05
    elif not hint:
        detected = _detect_category(q_tokens)
        if detected and (c_tokens & CATEGORY_WORDS[detected]):
            category_bonus = 0.05

    liq = candidate.liquidity_proxy or 0.0
    vol = candidate.volume_proxy or 0.0
    liquidity_bonus = min(0.05, (liq / 20000.0) + (vol / 20000.0))

    return round(jaccard + entity_bonus + number_bonus + category_bonus + liquidity_bonus, 8)


def _rank_candidates(intent: TradeIntent, candidates: list[MarketCandidate]) -> list[tuple[MarketCandidate, float]]:
    scored = [(cand, candidate_score(intent, cand)) for cand in candidates]
    scored.sort(
        key=lambda item: (
            item[1],
            float(item[0].liquidity_proxy or 0.0),
            float(item[0].volume_proxy or 0.0),
            item[0].market_key,
        ),
        reverse=True,
    )
    return scored


def find_candidates_polymarket(
    intent: TradeIntent,
    mode: str = "live",
    fixtures_dir: str = "fixtures",
    allow_fallback: bool = True,
    limit: int = 300,
) -> tuple[list[MarketCandidate], dict[str, Any], str, str | None]:
    markets: list[dict[str, Any]] = []
    source = "none"
    error: str | None = None
    debug: dict[str, Any] = {
        "fetched_candidates": 0,
        "eligible_two_sided": 0,
        "selected": 0,
        "excluded": 0,
        "exclusion_reasons": {},
    }

    fixture_path = Path(fixtures_dir) / "polymarket_markets_open.json"
    if mode == "fixture":
        try:
            payload = json.loads(fixture_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                markets = [m for m in payload if isinstance(m, dict)]
            elif isinstance(payload, dict) and isinstance(payload.get("markets"), list):
                markets = [m for m in payload.get("markets", []) if isinstance(m, dict)]
            source = "fixture"
        except Exception as exc:
            error = f"fixture_error: {exc}"
            source = "none"
    else:
        try:
            events = fetch_events_live(limit=limit, offset=0, timeout=(3.05, 10))
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
            source = "live"
        except Exception as exc:
            error = f"live_error: {exc}"
            source = "none"
            if allow_fallback:
                try:
                    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        markets = [m for m in payload if isinstance(m, dict)]
                    elif isinstance(payload, dict) and isinstance(payload.get("markets"), list):
                        markets = [m for m in payload.get("markets", []) if isinstance(m, dict)]
                    source = "fixture"
                except Exception as fexc:
                    error = f"{error} | fixture_error: {fexc}"

    candidates: list[MarketCandidate] = []
    for market in markets:
        candidates.append(
            MarketCandidate(
                venue="polymarket",
                market_key=str(market.get("conditionId") or market.get("id") or market.get("market_key")),
                title=market.get("question") or market.get("title") or market.get("event_title"),
                slug=market.get("slug") or market.get("event_slug"),
                close_time_utc=market.get("endDate") or market.get("closedTime"),
                liquidity_proxy=float(market.get("liquidityNum") or market.get("liquidity_proxy") or 0.0),
                volume_proxy=float(market.get("volume24hr") or market.get("volume_proxy") or 0.0),
                raw=dict(market),
            )
        )

    ranked = _rank_candidates(intent, candidates)
    top = [cand for cand, _ in ranked[:5]]
    debug.update(
        {
            "fetched_candidates": len(markets),
            "eligible_two_sided": len(markets),
            "selected": len(top),
        }
    )
    return top, debug, source, error


def find_candidates_kalshi(
    intent: TradeIntent,
    mode: str = "live",
    fixtures_dir: str = "fixtures",
    allow_fallback: bool = True,
    limit: int = 200,
    include_mve: bool = False,
) -> tuple[list[MarketCandidate], dict[str, Any], str, str | None]:
    markets: list[dict[str, Any]] = []
    source = "none"
    error: str | None = None
    debug: dict[str, Any] = {
        "fetched_candidates": 0,
        "eligible_two_sided": 0,
        "selected": 0,
        "excluded": 0,
        "exclusion_reasons": {},
    }
    fixture_path = Path(fixtures_dir) / "kalshi_markets_open.json"
    if mode == "fixture":
        try:
            payload = json.loads(fixture_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("markets"), list):
                markets = [m for m in payload.get("markets", []) if isinstance(m, dict)]
            elif isinstance(payload, list):
                markets = [m for m in payload if isinstance(m, dict)]
            source = "fixture"
        except Exception as exc:
            error = f"fixture_error: {exc}"
            source = "none"
    else:
        try:
            fetched = fetch_open_markets_live(limit=limit, timeout=(3.05, 10))
            markets = [m for m in fetched if isinstance(m, dict)]
            source = "live"
        except Exception as exc:
            error = f"live_error: {exc}"
            source = "none"
            if allow_fallback:
                try:
                    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict) and isinstance(payload.get("markets"), list):
                        markets = [m for m in payload.get("markets", []) if isinstance(m, dict)]
                    elif isinstance(payload, list):
                        markets = [m for m in payload if isinstance(m, dict)]
                    source = "fixture"
                except Exception as fexc:
                    error = f"{error} | fixture_error: {fexc}"

    candidates: list[MarketCandidate] = []
    for market in markets:
        ticker = str(market.get("ticker") or "")
        if ticker.startswith("KXMVESPORTSMULTIGAMEEXTENDED") and not include_mve:
            continue
        candidates.append(
            MarketCandidate(
                venue="kalshi",
                market_key=str(market.get("ticker") or market.get("market_key")),
                title=market.get("question") or market.get("title"),
                slug=market.get("ticker"),
                close_time_utc=market.get("close_time") or market.get("expiration_time"),
                liquidity_proxy=float(market.get("depth_top5") or market.get("liquidity_proxy") or 0.0),
                volume_proxy=float(market.get("trades_count_sample") or market.get("volume_proxy") or 0.0),
                raw=dict(market),
            )
        )

    ranked = _rank_candidates(intent, candidates)
    top = [cand for cand, _ in ranked[:5]]
    debug.update(
        {
            "fetched_candidates": len(markets),
            "eligible_two_sided": len(candidates),
            "selected": len(top),
            "excluded": max(0, len(markets) - len(candidates)),
        }
    )
    return top, debug, source, error


def _quality(score: float) -> Literal["HIGH", "MED", "LOW"]:
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MED"
    return "LOW"


def _score_snapshot(snapshot: dict[str, Any], venue: str) -> dict[str, Any]:
    risk = compute_execution_risk(snapshot, venue=venue)
    return {
        "mid": snapshot.get("mid"),
        "spread": snapshot.get("spread"),
        "depth_top5": snapshot.get("depth_top5"),
        "execution_risk_index": risk,
        "regime": compute_regime(risk),
        "drivers": compute_drivers(snapshot, venue=venue),
    }


def resolve_trade_detailed(
    intent: TradeIntent,
    mode: str = "live",
    fixtures_dir: str = "fixtures",
    allow_fallback: bool = True,
    include_mve: bool = False,
) -> ResolveResult:
    poly_candidates, poly_debug, poly_source, poly_error = find_candidates_polymarket(
        intent=intent,
        mode=mode,
        fixtures_dir=fixtures_dir,
        allow_fallback=allow_fallback,
    )
    kalshi_candidates, kalshi_debug, kalshi_source, kalshi_error = find_candidates_kalshi(
        intent=intent,
        mode=mode,
        fixtures_dir=fixtures_dir,
        allow_fallback=allow_fallback,
        include_mve=include_mve,
    )

    ranked_poly = _rank_candidates(intent, poly_candidates)
    ranked_kalshi = _rank_candidates(intent, kalshi_candidates)

    best_poly = ranked_poly[0][0] if ranked_poly else None
    best_poly_score = ranked_poly[0][1] if ranked_poly else 0.0
    best_kalshi = ranked_kalshi[0][0] if ranked_kalshi else None
    best_kalshi_score = ranked_kalshi[0][1] if ranked_kalshi else 0.0

    min_match_threshold = 0.25
    if best_poly_score < min_match_threshold:
        best_poly = None
    if best_kalshi_score < min_match_threshold:
        best_kalshi = None

    venue_status = {
        "polymarket": {
            "ok": poly_source in {"live", "fixture"},
            "source": poly_source,
            "error": poly_error,
        },
        "kalshi": {
            "ok": kalshi_source in {"live", "fixture"},
            "source": kalshi_source,
            "error": kalshi_error,
        },
    }

    resolved_poly: dict[str, Any] | None = None
    if best_poly is not None:
        try:
            raw = fetch_polymarket_snapshot_by_condition_id(
                best_poly.market_key,
                mode=mode,
                fixtures_dir=fixtures_dir,
                timeout=(3.05, 10),
            )
            if raw is not None:
                snapshot = normalize_polymarket_market(raw)
                resolved_poly = {
                    "market_key": snapshot.get("market_key"),
                    "title": snapshot.get("title") or snapshot.get("question"),
                    "slug": snapshot.get("slug"),
                    **_score_snapshot(snapshot, venue="polymarket"),
                    "match_score": best_poly_score,
                }
            else:
                venue_status["polymarket"]["ok"] = False
                venue_status["polymarket"]["error"] = "resolved_market_snapshot_unavailable"
        except Exception as exc:
            venue_status["polymarket"]["ok"] = False
            venue_status["polymarket"]["error"] = f"snapshot_error: {exc}"

    resolved_kalshi: dict[str, Any] | None = None
    if best_kalshi is not None:
        try:
            raw = fetch_kalshi_snapshot_by_ticker(
                best_kalshi.market_key,
                mode=mode,
                fixtures_dir=fixtures_dir,
                timeout=(3.05, 10),
            )
            if raw is not None:
                snapshot = normalize_kalshi_market(
                    ticker=str(raw.get("ticker") or raw.get("market_key")),
                    market_json=raw,
                    orderbook_json=raw.get("orderbook", {}),
                    trades_json=raw.get("trades", {}),
                )
                resolved_kalshi = {
                    "market_key": snapshot.get("market_key"),
                    "title": snapshot.get("title") or snapshot.get("question"),
                    "slug": snapshot.get("slug"),
                    **_score_snapshot(snapshot, venue="kalshi"),
                    "match_score": best_kalshi_score,
                }
            else:
                venue_status["kalshi"]["ok"] = False
                venue_status["kalshi"]["error"] = "resolved_market_snapshot_unavailable"
        except Exception as exc:
            venue_status["kalshi"]["ok"] = False
            venue_status["kalshi"]["error"] = f"snapshot_error: {exc}"

    overall = max(best_poly_score, best_kalshi_score)
    explanation = [
        f"Polymarket candidates considered: {len(poly_candidates)}",
        f"Kalshi candidates considered: {len(kalshi_candidates)}",
        f"Polymarket selection debug: {poly_debug}",
        f"Kalshi selection debug: {kalshi_debug}",
    ]

    resolved = ResolvedMarket(
        trade_id=intent.trade_id,
        intent_query=intent.query,
        resolved_at_utc=datetime.now(timezone.utc).isoformat(),
        best_kalshi=resolved_kalshi,
        best_polymarket=resolved_poly,
        match_quality=_quality(overall),
        explanation=explanation,
        candidates={
            "kalshi": [candidate.model_dump() for candidate in kalshi_candidates[:5]],
            "polymarket": [candidate.model_dump() for candidate in poly_candidates[:5]],
        },
    )
    debug = {"polymarket": poly_debug, "kalshi": kalshi_debug}
    return ResolveResult(resolved=resolved, venue_status=venue_status, debug=debug)


def resolve_trade(
    intent: TradeIntent,
    mode: str = "live",
    fixtures_dir: str = "fixtures",
    allow_fallback: bool = True,
    include_mve: bool = False,
) -> ResolvedMarket:
    return resolve_trade_detailed(
        intent=intent,
        mode=mode,
        fixtures_dir=fixtures_dir,
        allow_fallback=allow_fallback,
        include_mve=include_mve,
    ).resolved
