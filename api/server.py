"""FastAPI server for market resolver."""

from __future__ import annotations

import logging
import os
import hashlib
import re
import traceback
import time
import uuid
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.market_resolver import TradeIntent, resolve_trade_detailed

LOGGER = logging.getLogger(__name__)

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

DEMO_LEADER_MARKETS = [
    {
        "title": "Will bitcoin hit $1m before GTA VI?",
        "market_key": "vardr_btc_leader_1",
        "reference_event": "GTA VI",
        "mid": 0.56,
        "computed_mid_delta": 0.05,
        "one_hour_price_change": 0.05,
        "one_day_price_change": 0.08,
        "source": "vardr",
        "reason": "Demo leader market showing a meaningful positive move.",
    },
    {
        "title": "Will Ethereum ETF volume rise this week?",
        "market_key": "vardr_eth_leader_low_move",
        "reference_event": None,
        "mid": 0.49,
        "computed_mid_delta": 0.01,
        "one_hour_price_change": 0.01,
        "one_day_price_change": 0.015,
        "source": "vardr",
        "reason": "Demo low-move leader used to exercise filtering.",
    },
]

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
    title = raw.get("title") or raw.get("question")
    if not title:
        return {}

    normalized = {
        "title": str(title),
        "market_key": raw.get("market_key") or raw.get("id") or raw.get("ticker") or raw.get("slug"),
        "reference_event": raw.get("reference_event"),
        "mid": raw.get("mid"),
        "computed_mid_delta": raw.get("computed_mid_delta"),
        "one_hour_price_change": raw.get("one_hour_price_change"),
        "one_day_price_change": raw.get("one_day_price_change"),
        "source": raw.get("source") or "vardr",
        "reason": raw.get("reason"),
    }
    if not normalized["market_key"]:
        normalized["market_key"] = _fallback_market_key(normalized["title"])
    return normalized


def get_leader_markets() -> list[dict]:
    normalized = [normalize_leader_market(market) for market in DEMO_LEADER_MARKETS]
    return [market for market in normalized if market]


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
