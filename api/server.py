"""FastAPI server for market resolver."""

from __future__ import annotations

import logging
import os
import hashlib
import re
import traceback
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from market_fetcher.leader_markets import load_leader_markets_from_vardr1
from market_fetcher.lag_candidates import (
    DEFAULT_LEADER_FETCH_LIMIT,
    DEFAULT_MAX_ABS_LEADER_MOVE,
    DEFAULT_MAX_CANDIDATES_PER_EVENT_OUTPUT,
    DEFAULT_MAX_CANDIDATES_PER_LEADER_OUTPUT,
    DEFAULT_MAX_LEADERS_EVALUATED,
    DEFAULT_MIN_DIVERGENCE,
    DEFAULT_MIN_RELATED_ABS_MOVE_24H,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_MIN_SIMILARITY_FOR_TRADE,
    DEFAULT_MIN_VALID_CANDIDATES,
    DEFAULT_TIMEOUT_SECONDS,
    build_lag_candidates_debug,
    build_lag_candidates_with_metadata,
    load_local_polymarket_universe_with_metadata,
)
from market_fetcher.vardr1_client import map_vardr1_market_to_leader
from src.market_resolver import TradeIntent, resolve_trade_detailed

LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]

app = FastAPI(title="True Markets Resolver API", version="1.0.0")

LEADER_MARKET_FIELDS = (
    "title",
    "market_key",
    "reference_event",
    "mid",
    "computed_mid_delta",
    "one_hour_price_change",
    "one_day_price_change",
    "source",
    "reason",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResolveMarketRequest(BaseModel):
    query: str = Field(min_length=3)
    side: Literal["yes", "no"]
    notional_usd: float = Field(gt=0)
    mode: Literal["live", "fixture"] | None = None
    allow_fallback: bool | None = None
    trade_id: str | None = None
    category_hint: str | None = None
    event_date: str | None = None
    venue_preference: str | None = None

    @field_validator("side", mode="before")
    @classmethod
    def normalize_side(cls, value: str) -> str:
        if isinstance(value, str):
            return value.lower()
        return value


class ResolveErrorResponse(BaseModel):
    error_code: str
    error_type: str
    message: str
    request_id: str | None = None
    venue: str | None = None
    stack_trace: str | None = None
    query: str | None = None
    candidates: dict | None = None
    cross_venue: dict | None = None
    venue_status: dict | None = None


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _fallback_market_key(title: str) -> str:
    normalized = _normalize_text(title)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"vardr_{digest}"


def _best_leader_move(market: dict) -> float | None:
    for field in ("computed_mid_delta", "one_hour_price_change", "one_day_price_change"):
        value = market.get(field)
        if value is not None:
            return float(value)
    return None


def normalize_leader_market(raw: dict) -> dict:
    normalized = map_vardr1_market_to_leader(raw)
    if not normalized:
        return {}
    if not normalized["market_key"]:
        normalized["market_key"] = _fallback_market_key(normalized["title"])
    return normalized


def get_leader_markets(limit: int = DEFAULT_LEADER_FETCH_LIMIT) -> list[dict]:
    return load_leader_markets_from_vardr1(limit=limit)


def _demo_lag_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    leaders = [
        {
            "title": "Will the Fed cut rates by June 2026?",
            "market_key": "demo-leader-fed-cut-june-2026",
            "market_id": "demo-leader-fed-cut-june-2026",
            "market_title": "Will the Fed cut rates by June 2026?",
            "current_price": 0.58,
            "one_hour_price_change": 0.16,
            "one_day_price_change": 0.21,
            "price_change_1h": 0.16,
            "price_change_24h": 0.21,
            "leader_score": 0.63,
            "source": "demo_snapshot",
        }
    ]
    universe = [
        {
            "venue": "polymarket",
            "market_key": "demo-related-mortgage-rates-below-6",
            "market_id": "demo-related-mortgage-rates-below-6",
            "title": "Will 30-year mortgage rates fall below 6% by June 2026?",
            "question": "Will 30-year mortgage rates fall below 6% by June 2026?",
            "one_hour_price_change": 0.01,
            "one_day_price_change": 0.018,
            "mid": 0.44,
            "current_price": 0.44,
            "spread": 0.025,
            "depth_top5": 4200,
            "liquidity_proxy": 4200,
            "source": "demo_snapshot",
        }
    ]
    return leaders, universe, {
        "source": "demo_snapshot",
        "source_path": "built_in_demo_fixture",
        "row_count": len(universe),
    }


def _candidate_trade_view(candidate: dict[str, Any]) -> dict[str, Any]:
    def rounded(value: Any) -> Any:
        if isinstance(value, (int, float)):
            return round(float(value), 3)
        return value

    execution_risk_score = rounded(candidate.get("execution_risk_score"))
    similarity_score = rounded(candidate.get("similarity_score"))
    divergence = rounded(candidate.get("divergence_score"))
    confidence = rounded(candidate.get("trade_rank_score"))
    trade_rank_score = rounded(candidate.get("trade_rank_score"))
    reason = (
        "The leader market moved sharply while the related market barely moved, despite a "
        f"{candidate.get('expected_correlation_direction')} causal relationship. Because execution risk is "
        f"{candidate.get('execution_risk_label')}, Vardr flags the related market as a "
        f"{candidate.get('trade_bucket')} trade candidate."
    )
    return {
        "leader_market": {
            "market_id": candidate.get("leader_market_id"),
            "title": candidate.get("leader_market_title"),
            "price_change_1h": candidate.get("leader_price_change_1h"),
            "price_change_24h": candidate.get("leader_price_change_24h"),
        },
        "lagging_market": {
            "market_id": candidate.get("related_market_id"),
            "title": candidate.get("related_market_title"),
            "price_change_1h": candidate.get("related_price_change_1h"),
            "price_change_24h": candidate.get("related_price_change_24h"),
        },
        "suggested_trade_direction": candidate.get("suggested_trade_direction"),
        "relationship_type": candidate.get("relationship_type"),
        "expected_direction": candidate.get("expected_correlation_direction"),
        "similarity_score": similarity_score,
        "divergence": divergence,
        "confidence": confidence,
        "execution_risk_score": execution_risk_score,
        "execution_risk_label": candidate.get("execution_risk_label"),
        "execution_risk_drivers": candidate.get("execution_risk_drivers"),
        "liquidity_score": rounded(candidate.get("liquidity_score")),
        "trade_rank_score": trade_rank_score,
        "trade_bucket": candidate.get("trade_bucket"),
        "execution_risk": {
            "risk_score": execution_risk_score,
            "risk_label": candidate.get("execution_risk_label"),
            "drivers": candidate.get("execution_risk_drivers"),
        },
        "reason": reason,
        "trade_reason": reason,
        "data_source": candidate.get("data_source") or candidate.get("source") or "live",
    }


def _claude_trade_context(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [
        _candidate_trade_view(candidate)
        for candidate in candidates
        if candidate.get("trade_bucket") == "primary" and candidate.get("tradable_signal") is True
    ]
    review = [
        _candidate_trade_view(candidate)
        for candidate in candidates
        if candidate.get("trade_bucket") != "primary" or candidate.get("tradable_signal") is not True
    ]
    reasoning = [
        {
            "market_id": candidate.get("related_market_id"),
            "relationship_type": candidate.get("relationship_type"),
            "relationship_reason": candidate.get("relationship_reason") or candidate.get("causal_link_reason"),
            "do_not_hallucinate_causality": candidate.get("relationship_type") not in {
                "CAUSALLY_LINKED",
                "SAME_OPTION_SET_NEGATIVE_CORRELATION",
                "SAME_OPTION_SET_POSITIVE_CORRELATION",
            },
        }
        for candidate in candidates
    ]
    return {
        "primary_trades": primary,
        "review_candidates": review,
        "reasoning_context": reasoning,
    }


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("TRUE_MARKETS_API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/leader-markets")
def leader_markets(
    limit: int = Query(default=10, ge=1),
    min_abs_move: float = Query(default=0.02, ge=0.0),
) -> list[dict]:
    leaders = []
    for market in get_leader_markets():
        move = _best_leader_move(market)
        if move is None or abs(move) < min_abs_move:
            continue
        leaders.append({field: market.get(field) for field in LEADER_MARKET_FIELDS})
        if len(leaders) >= limit:
            break
    return leaders


@app.get("/lag-candidates")
def lag_candidates(
    limit: int = Query(default=50, ge=1),
    min_similarity: float = Query(default=DEFAULT_MIN_SIMILARITY, ge=0.0, le=1.0),
    min_divergence: float = Query(default=DEFAULT_MIN_DIVERGENCE, ge=0.0),
    leader_fetch_limit: int = Query(default=DEFAULT_LEADER_FETCH_LIMIT, ge=1),
    leader_offset: int = Query(default=0, ge=0),
    leader_page_size: int = Query(default=25, ge=1, le=50),
    max_leaders_evaluated: int = Query(default=DEFAULT_MAX_LEADERS_EVALUATED, ge=1),
    min_valid_candidates: int = Query(default=DEFAULT_MIN_VALID_CANDIDATES, ge=0),
    max_universe: int = Query(default=5000, ge=1),
    max_candidates_per_leader: int = Query(default=25, ge=1),
    max_candidates_per_leader_output: int = Query(default=DEFAULT_MAX_CANDIDATES_PER_LEADER_OUTPUT, ge=1),
    max_candidates_per_event_output: int = Query(default=DEFAULT_MAX_CANDIDATES_PER_EVENT_OUTPUT, ge=1),
    include_review_only: bool = Query(default=False),
    max_abs_leader_move: float = Query(default=DEFAULT_MAX_ABS_LEADER_MOVE, ge=0.0),
    exploratory: bool = Query(default=False),
    include_weak_similarity: bool = Query(default=False),
    timeout_seconds: float = Query(default=DEFAULT_TIMEOUT_SECONDS, ge=0.1, le=60.0),
    min_leaders_before_timeout: int = Query(default=5, ge=0),
    min_related_abs_move_24h: float = Query(default=DEFAULT_MIN_RELATED_ABS_MOVE_24H, ge=0.0),
    min_similarity_for_trade: float = Query(default=DEFAULT_MIN_SIMILARITY_FOR_TRADE, ge=0.0, le=1.0),
    demo: bool = Query(default=False),
) -> dict[str, Any]:
    start = time.monotonic()
    if demo:
        leaders, polymarket_universe, universe_metadata = _demo_lag_fixture()
        leaders_done = time.monotonic()
        universe_done = leaders_done
    else:
        leaders = get_leader_markets(limit=leader_fetch_limit)
        leaders_done = time.monotonic()
        polymarket_universe, universe_metadata = load_local_polymarket_universe_with_metadata(ROOT_DIR)
        universe_done = time.monotonic()
    total_leaders_available = len(leaders)
    leader_slice_start = min(leader_offset, total_leaders_available)
    leader_slice_end = min(leader_slice_start + leader_page_size, total_leaders_available)
    leader_slice = leaders[leader_slice_start:leader_slice_end]
    candidates, candidate_metadata = build_lag_candidates_with_metadata(
        leader_markets=leader_slice,
        polymarket_universe=polymarket_universe,
        min_similarity=min_similarity,
        min_divergence=min_divergence,
        limit=limit,
        max_leaders_evaluated=max_leaders_evaluated,
        min_valid_candidates=min_valid_candidates,
        max_universe=max_universe,
        max_candidates_per_leader=max_candidates_per_leader,
        max_candidates_per_leader_output=max_candidates_per_leader_output,
        max_candidates_per_event_output=max_candidates_per_event_output,
        include_review_only=include_review_only,
        max_abs_leader_move=max_abs_leader_move,
        exploratory=exploratory,
            include_weak_similarity=include_weak_similarity,
            timeout_seconds=timeout_seconds,
            min_leaders_before_timeout=min_leaders_before_timeout,
            min_related_abs_move_24h=min_related_abs_move_24h,
            min_similarity_for_trade=0.30 if demo else min_similarity_for_trade,
        )
    scoring_done = time.monotonic()
    candidate_metadata["universe_row_count"] = universe_metadata.get("row_count")
    candidate_metadata["leader_offset"] = leader_offset
    candidate_metadata["leader_page_size"] = leader_page_size
    candidate_metadata["leader_slice_start"] = leader_slice_start
    candidate_metadata["leader_slice_end"] = leader_slice_end
    candidate_metadata["total_leaders_available"] = total_leaders_available
    candidate_metadata["next_leader_offset"] = leader_slice_end if leader_slice_end < total_leaders_available else None
    candidate_metadata["demo_mode"] = demo
    candidate_metadata["data_source"] = "demo_snapshot" if demo else "live"
    if demo:
        for candidate in candidates:
            candidate["data_source"] = "demo_snapshot"
            candidate["demo_candidate"] = True
    LOGGER.warning(
        "lag_candidates timing fetch_leaders_s=%.4f load_universe_s=%.4f scoring_s=%.4f leaders_fetched=%d leader_slice=%d:%d leaders_evaluated=%d leaders_skipped=%d returned=%d universe_rows=%s source=%s",
        leaders_done - start,
        universe_done - leaders_done,
        scoring_done - universe_done,
        len(leaders),
        leader_slice_start,
        leader_slice_end,
        candidate_metadata["leaders_evaluated"],
        candidate_metadata["leaders_skipped"],
        len(candidates),
        universe_metadata.get("row_count"),
        universe_metadata.get("source_path"),
    )
    claude_input = _claude_trade_context(candidates)
    return {
        "candidates": candidates,
        "primary_trades": claude_input["primary_trades"],
        "review_candidates": claude_input["review_candidates"],
        "reasoning_context": claude_input["reasoning_context"],
        "claude_input": claude_input,
        "metadata": candidate_metadata,
    }


@app.get("/lag-candidates/debug")
def lag_candidates_debug(
    leader_limit: int = Query(default=5, ge=1),
    leader_fetch_limit: int = Query(default=DEFAULT_LEADER_FETCH_LIMIT, ge=1),
    max_universe: int = Query(default=5000, ge=1),
    top_k: int = Query(default=10, ge=1),
    min_valid_candidates: int = Query(default=DEFAULT_MIN_VALID_CANDIDATES, ge=0),
    exploratory: bool = Query(default=False),
    include_weak_similarity: bool = Query(default=False),
    min_similarity: float = Query(default=DEFAULT_MIN_SIMILARITY, ge=0.0, le=1.0),
    min_divergence: float = Query(default=DEFAULT_MIN_DIVERGENCE, ge=0.0),
    timeout_seconds: float = Query(default=DEFAULT_TIMEOUT_SECONDS, ge=0.1, le=60.0),
    min_leaders_before_timeout: int = Query(default=5, ge=0),
    max_abs_leader_move: float = Query(default=DEFAULT_MAX_ABS_LEADER_MOVE, ge=0.0),
    min_related_abs_move_24h: float = Query(default=DEFAULT_MIN_RELATED_ABS_MOVE_24H, ge=0.0),
    min_similarity_for_trade: float = Query(default=DEFAULT_MIN_SIMILARITY_FOR_TRADE, ge=0.0, le=1.0),
) -> list[dict[str, Any]]:
    start = time.monotonic()
    leaders = get_leader_markets(limit=leader_fetch_limit)
    leaders_done = time.monotonic()
    polymarket_universe, universe_metadata = load_local_polymarket_universe_with_metadata(ROOT_DIR)
    universe_done = time.monotonic()
    rows = build_lag_candidates_debug(
        leader_markets=leaders,
        polymarket_universe=polymarket_universe,
        leader_limit=leader_limit,
        top_candidates=top_k,
        min_similarity=min_similarity,
        min_divergence=min_divergence,
        min_valid_candidates=min_valid_candidates,
        max_universe=max_universe,
        timeout_seconds=timeout_seconds,
        min_leaders_before_timeout=min_leaders_before_timeout,
        max_abs_leader_move=max_abs_leader_move,
        min_related_abs_move_24h=min_related_abs_move_24h,
        min_similarity_for_trade=min_similarity_for_trade,
        exploratory=exploratory,
        include_weak_similarity=include_weak_similarity,
        universe_source_path=universe_metadata.get("source_path"),
        universe_row_count=universe_metadata.get("row_count"),
    )
    scoring_done = time.monotonic()
    LOGGER.warning(
        "lag_candidates_debug timing fetch_leaders_s=%.4f load_universe_s=%.4f scoring_s=%.4f leaders=%d returned=%d universe_rows=%s source=%s",
        leaders_done - start,
        universe_done - leaders_done,
        scoring_done - universe_done,
        len(leaders),
        len(rows),
        universe_metadata.get("row_count"),
        universe_metadata.get("source_path"),
    )
    return rows


@app.post("/resolve_market")
def resolve_market(
    intent: ResolveMarketRequest,
    mode: Literal["live", "fixture"] | None = Query(default=None),
    allow_fallback: bool | None = Query(default=None),
    _auth: None = Depends(require_api_key),
) -> JSONResponse:
    request_id = str(uuid.uuid4())
    start = time.monotonic()
    LOGGER.info(
        "resolve_market_request request_id=%s query=%r side=%s notional_usd=%s mode=%s allow_fallback=%s",
        request_id,
        intent.query,
        intent.side,
        intent.notional_usd,
        mode,
        allow_fallback,
    )

    resolved_mode = mode or intent.mode or os.getenv("RESOLVER_MODE", "live")
    fixtures_dir = os.getenv("RESOLVER_FIXTURES_DIR", "fixtures")
    resolved_allow_fallback = allow_fallback if allow_fallback is not None else intent.allow_fallback
    if resolved_allow_fallback is None:
        resolved_allow_fallback = os.getenv("RESOLVER_ALLOW_FALLBACK", "true").lower() in {"1", "true", "yes"}
    include_mve = os.getenv("RESOLVER_INCLUDE_MVE", "false").lower() in {"1", "true", "yes"}
    debug_mode = os.getenv("DEBUG", "0") in {"1", "true", "TRUE"}

    try:
        LOGGER.info("resolve_market_step request_id=%s step=resolver_start", request_id)
        result = resolve_trade_detailed(
            intent=TradeIntent(
                trade_id=intent.trade_id,
                query=intent.query,
                side=intent.side.upper(),
                notional_usd=float(intent.notional_usd),
                category_hint=intent.category_hint,
                event_date=intent.event_date,
                venue_preference=intent.venue_preference,
            ),
            mode=resolved_mode,
            fixtures_dir=fixtures_dir,
            allow_fallback=resolved_allow_fallback,
            include_mve=include_mve,
        )
        LOGGER.info("resolve_market_step request_id=%s step=resolver_done elapsed_s=%.3f", request_id, time.monotonic() - start)

        resolved = result.resolved.model_dump()
        venue_status = result.venue_status

        LOGGER.info(
            "resolve_market_result request_id=%s kalshi=%s polymarket=%s venue_status=%s elapsed_s=%.3f",
            request_id,
            (resolved.get("best_kalshi") or {}).get("market_key"),
            (resolved.get("best_polymarket") or {}).get("market_key"),
            venue_status,
            time.monotonic() - start,
        )

        kalshi_candidates = resolved.get("candidates", {}).get("kalshi", [])[:5]
        poly_candidates = resolved.get("candidates", {}).get("polymarket", [])[:5]
        cross_venue = {
            "matched_pairs": [],
            "kalshi_only_examples": [
                {
                    "kalshi_ticker": candidate.get("market_key"),
                    "kalshi_title": candidate.get("title"),
                }
                for candidate in kalshi_candidates[:2]
            ],
            "polymarket_only_examples": [
                {
                    "polymarket_market_key": candidate.get("market_key"),
                    "polymarket_title": candidate.get("title"),
                }
                for candidate in poly_candidates[:2]
            ],
        }
        while len(cross_venue["kalshi_only_examples"]) < 2:
            cross_venue["kalshi_only_examples"].append({"kalshi_ticker": None, "kalshi_title": None})
        while len(cross_venue["polymarket_only_examples"]) < 2:
            cross_venue["polymarket_only_examples"].append({"polymarket_market_key": None, "polymarket_title": None})

        if resolved.get("best_kalshi") and resolved.get("best_polymarket"):
            cross_venue["matched_pairs"].append(
                {
                    "kalshi_ticker": resolved["best_kalshi"].get("market_key"),
                    "kalshi_title": resolved["best_kalshi"].get("title"),
                    "polymarket_market_key": resolved["best_polymarket"].get("market_key"),
                    "polymarket_title": resolved["best_polymarket"].get("title"),
                    "match_score": max(
                        float(resolved["best_kalshi"].get("match_score") or 0.0),
                        float(resolved["best_polymarket"].get("match_score") or 0.0),
                    ),
                }
            )

        if not venue_status.get("kalshi", {}).get("ok") and not venue_status.get("polymarket", {}).get("ok"):
            return JSONResponse(
                status_code=503,
                content=ResolveErrorResponse(
                    error_code="venues_unavailable",
                    error_type="venues_unavailable",
                    message="Both venues failed during market discovery/snapshot fetch.",
                    request_id=request_id,
                    venue_status=venue_status,
                    stack_trace=None,
                ).model_dump(),
            )

        scores: dict[str, Any] = {
            "kalshi": None,
            "polymarket": [],
        }
        if resolved.get("best_kalshi"):
            scores["kalshi"] = {
                "market_key": resolved["best_kalshi"].get("market_key"),
                "execution_risk_index": resolved["best_kalshi"].get("execution_risk_index"),
                "regime": resolved["best_kalshi"].get("regime"),
                "drivers": resolved["best_kalshi"].get("drivers"),
            }
        if resolved.get("best_polymarket"):
            scores["polymarket"] = [
                {
                    "market_key": resolved["best_polymarket"].get("market_key"),
                    "execution_risk_index": resolved["best_polymarket"].get("execution_risk_index"),
                    "regime": resolved["best_polymarket"].get("regime"),
                    "drivers": resolved["best_polymarket"].get("drivers"),
                }
            ]

        resolved["venue_status"] = venue_status
        resolved["debug"] = result.debug
        resolved["request_id"] = request_id
        resolved["cross_venue"] = cross_venue
        resolved["resolution_status"] = (
            "resolved" if (resolved.get("best_kalshi") or resolved.get("best_polymarket")) else "not_found"
        )
        resolved["kalshi_snapshot"] = resolved.get("best_kalshi")
        resolved["kalshi_snapshots"] = [resolved["best_kalshi"]] if resolved.get("best_kalshi") else []
        resolved["polymarket_snapshots"] = [resolved["best_polymarket"]] if resolved.get("best_polymarket") else []
        resolved["snapshots"] = {
            "kalshi": resolved["kalshi_snapshots"],
            "polymarket": resolved["polymarket_snapshots"],
        }
        resolved["scores"] = scores
        return JSONResponse(status_code=200, content=resolved)

    except HTTPException as exc:
        raise exc
    except Exception as exc:
        LOGGER.exception("resolve_market_internal_error request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content=ResolveErrorResponse(
                error_code="internal_error",
                error_type="internal_error",
                message=str(exc),
                request_id=request_id,
                stack_trace=traceback.format_exc() if debug_mode else None,
            ).model_dump(),
        )
