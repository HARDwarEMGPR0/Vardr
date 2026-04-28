from __future__ import annotations

from fastapi.testclient import TestClient

import api.server as api_server
import src.market_resolver as resolver_module
from api.server import app


client = TestClient(app)


def test_health_ok() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_leader_markets_returns_list_with_required_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        api_server,
        "get_leader_markets",
        lambda: [
            {
                "title": "Will Bitcoin rally?",
                "market_key": "btc-rally",
                "reference_event": None,
                "mid": 0.55,
                "computed_mid_delta": 0.05,
                "one_hour_price_change": 0.03,
                "one_day_price_change": 0.05,
                "source": "vardr1",
                "reason": "test",
            }
        ],
    )
    resp = client.get("/leader-markets")

    assert resp.status_code == 200
    payload = resp.json()
    assert isinstance(payload, list)
    assert payload
    required = {
        "title",
        "market_key",
        "reference_event",
        "mid",
        "computed_mid_delta",
        "one_hour_price_change",
        "one_day_price_change",
        "source",
        "reason",
    }
    assert required.issubset(payload[0])


def test_leader_markets_limit_parameter_works(monkeypatch) -> None:
    monkeypatch.setattr(
        api_server,
        "get_leader_markets",
        lambda: [
            api_server.normalize_leader_market({"title": "One", "market_key": "one", "computed_mid_delta": 0.03}),
            api_server.normalize_leader_market({"title": "Two", "market_key": "two", "computed_mid_delta": 0.04}),
        ],
    )
    resp = client.get("/leader-markets?limit=1&min_abs_move=0")

    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_leader_markets_min_abs_move_filters_low_move_leaders(monkeypatch) -> None:
    monkeypatch.setattr(
        api_server,
        "get_leader_markets",
        lambda: [
            api_server.normalize_leader_market(
                {"title": "Low move", "market_key": "low", "computed_mid_delta": 0.01}
            )
        ],
    )
    resp = client.get("/leader-markets?min_abs_move=0.06")

    assert resp.status_code == 200
    assert resp.json() == []


def test_leader_markets_returns_empty_when_vardr1_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(api_server, "get_leader_markets", lambda: [])
    resp = client.get("/leader-markets")

    assert resp.status_code == 200
    assert resp.json() == []


def test_leader_markets_creates_fallback_key_and_skips_missing_title(monkeypatch) -> None:
    monkeypatch.setattr(
        api_server,
        "get_leader_markets",
        lambda: [
            api_server.normalize_leader_market({
                "title": "Fallback key market",
                "computed_mid_delta": 0.03,
            }),
            api_server.normalize_leader_market({
                "computed_mid_delta": 0.03,
            }),
        ],
    )

    resp = client.get("/leader-markets")

    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["market_key"].startswith("vardr1_")


def test_lag_candidates_uses_vardr1_leaders_and_local_polymarket_universe(monkeypatch) -> None:
    monkeypatch.setattr(
        api_server,
        "get_leader_markets",
        lambda **_kw: [
            {
                "title": "Will Bitcoin ETF be approved by end of 2026?",
                "market_key": "leader-btc-etf-approved",
                "one_hour_price_change": 0.12,
                "one_day_price_change": 0.20,
            }
        ],
    )
    markets = [
            {
                "venue": "polymarket",
                "market_key": "leader-btc-etf-approved",
                "title": "Will Bitcoin ETF be approved by end of 2026?",
                "slug": "btc-self",
                "one_hour_price_change": 0.0,
                "one_day_price_change": 0.0,
            },
            {
                "venue": "polymarket",
                "market_key": "related-btc-etf-launch",
                "title": "Will Bitcoin ETF launch by end of 2026?",
                "slug": "btc-etf-launch",
                "one_hour_price_change": 0.01,
                "one_day_price_change": 0.02,
            },
            {
                "venue": "polymarket",
                "market_key": "duplicate-btc-etf",
                "title": "Will Bitcoin ETF launch by end of 2026?",
                "slug": "btc-etf-launch",
                "one_hour_price_change": 0.0,
                "one_day_price_change": 0.0,
            },
            {
                "venue": "polymarket",
                "market_key": "unrelated-weather",
                "title": "Will it rain in New York tomorrow?",
                "slug": "rain-nyc",
                "one_hour_price_change": 0.0,
                "one_day_price_change": 0.0,
            },
        ]
    monkeypatch.setattr(
        api_server,
        "load_local_polymarket_universe_with_metadata",
        lambda _root: (markets, {"row_count": len(markets), "source_path": "mock.parquet"}),
    )

    resp = client.get("/lag-candidates?min_similarity=0.1&min_divergence=0.001")

    assert resp.status_code == 200
    payload = resp.json()
    assert "candidates" in payload
    assert "metadata" in payload
    candidates = payload["candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["leader_market_title"] == "Will Bitcoin ETF be approved by end of 2026?"
    assert candidate["leader_market_id"] == "leader-btc-etf-approved"
    assert candidate["related_market_title"] == "Will Bitcoin ETF launch by end of 2026?"
    assert candidate["related_market_id"] == "related-btc-etf-launch"
    assert candidate["relationship_type"] == "CAUSALLY_LINKED"
    assert candidate["expected_correlation_direction"] == "positive"
    assert candidate["similarity_score"] > 0
    assert candidate["leader_price_change_1h"] == 0.12
    assert candidate["related_price_change_1h"] == 0.01
    assert candidate["leader_price_change_24h"] == 0.20
    assert candidate["related_price_change_24h"] == 0.02
    assert candidate["divergence_score"] > 0
    assert candidate["suggested_trade_direction"] == "buy_yes_related"
    assert candidate["execution_risk_score"] is not None
    assert candidate["execution_risk_label"] in {"LOW", "MEDIUM", "HIGH"}
    assert candidate["trade_rank_score"] is not None
    assert candidate["trade_bucket"] == "primary"
    assert candidate["exploratory_match"] is False
    assert candidate["strong_match"] is True
    assert "Leader moved" in candidate["explanation"]
    assert payload["primary_trades"]
    assert payload["claude_input"]["primary_trades"] == payload["primary_trades"]
    assert set(payload["claude_input"]) == {"primary_trades", "review_candidates", "reasoning_context"}

    meta = payload["metadata"]
    assert meta["leaders_fetched"] == 1
    assert meta["leaders_evaluated"] == 1
    assert meta["leaders_skipped"] == 0
    assert meta["valid_candidates_found"] == 1
    assert meta["universe_row_count"] == len(markets)


def test_lag_candidates_demo_mode_returns_structured_candidate() -> None:
    resp = client.get("/lag-candidates?demo=true")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["metadata"]["demo_mode"] is True
    assert payload["metadata"]["data_source"] == "demo_snapshot"
    assert len(payload["primary_trades"]) + len(payload["review_candidates"]) >= 1
    candidate = (payload["primary_trades"] or payload["review_candidates"])[0]
    for field in (
        "leader_market",
        "lagging_market",
        "relationship_type",
        "expected_direction",
        "divergence",
        "confidence",
        "execution_risk_score",
        "execution_risk_label",
        "execution_risk_drivers",
        "liquidity_score",
        "trade_rank_score",
        "trade_bucket",
        "reason",
    ):
        assert field in candidate
    assert candidate["data_source"] == "demo_snapshot"


def test_lag_candidates_pages_leaders_before_scoring(monkeypatch) -> None:
    leaders = [
        {
            "title": f"Leader {idx}",
            "market_key": f"leader-{idx}",
            "one_hour_price_change": 0.12,
            "one_day_price_change": 0.20,
        }
        for idx in range(8)
    ]
    observed = {}

    monkeypatch.setattr(api_server, "get_leader_markets", lambda **_kw: leaders)
    monkeypatch.setattr(
        api_server,
        "load_local_polymarket_universe_with_metadata",
        lambda _root: ([], {"row_count": 0, "source_path": "mock.parquet"}),
    )

    def fake_build_lag_candidates_with_metadata(*, leader_markets, polymarket_universe, **_kwargs):
        observed["leader_ids"] = [leader["market_key"] for leader in leader_markets]
        return [], {
            "leaders_fetched": len(leader_markets),
            "leaders_evaluated": len(leader_markets),
            "remaining_leaders_not_evaluated": 0,
            "leaders_skipped": 0,
            "skip_reasons_count": {},
            "valid_candidates_found": 0,
        }

    monkeypatch.setattr(api_server, "build_lag_candidates_with_metadata", fake_build_lag_candidates_with_metadata)

    resp = client.get("/lag-candidates?leader_offset=2&leader_page_size=3&leader_fetch_limit=8")

    assert resp.status_code == 200
    assert observed["leader_ids"] == ["leader-2", "leader-3", "leader-4"]
    meta = resp.json()["metadata"]
    assert meta["leader_offset"] == 2
    assert meta["leader_page_size"] == 3
    assert meta["leader_slice_start"] == 2
    assert meta["leader_slice_end"] == 5
    assert meta["total_leaders_available"] == 8
    assert meta["next_leader_offset"] == 5


def test_lag_candidates_last_leader_page_has_no_next_offset(monkeypatch) -> None:
    leaders = [
        {
            "title": f"Leader {idx}",
            "market_key": f"leader-{idx}",
            "one_hour_price_change": 0.12,
            "one_day_price_change": 0.20,
        }
        for idx in range(8)
    ]

    monkeypatch.setattr(api_server, "get_leader_markets", lambda **_kw: leaders)
    monkeypatch.setattr(
        api_server,
        "load_local_polymarket_universe_with_metadata",
        lambda _root: ([], {"row_count": 0, "source_path": "mock.parquet"}),
    )
    monkeypatch.setattr(
        api_server,
        "build_lag_candidates_with_metadata",
        lambda *, leader_markets, polymarket_universe, **_kwargs: (
            [],
            {
                "leaders_fetched": len(leader_markets),
                "leaders_evaluated": len(leader_markets),
                "remaining_leaders_not_evaluated": 0,
                "leaders_skipped": 0,
                "skip_reasons_count": {},
                "valid_candidates_found": 0,
            },
        ),
    )

    resp = client.get("/lag-candidates?leader_offset=6&leader_page_size=25&leader_fetch_limit=8")

    assert resp.status_code == 200
    meta = resp.json()["metadata"]
    assert meta["leader_slice_start"] == 6
    assert meta["leader_slice_end"] == 8
    assert meta["total_leaders_available"] == 8
    assert meta["next_leader_offset"] is None


def test_lag_candidates_sorted_by_divergence_descending(monkeypatch) -> None:
    monkeypatch.setattr(
        api_server,
        "get_leader_markets",
        lambda **_kw: [
            {
                "title": "Will Bitcoin rally in 2026?",
                "market_key": "leader-btc",
                "one_hour_price_change": 0.10,
                "one_day_price_change": 0.30,
            }
        ],
    )
    markets = [
            {
                "venue": "polymarket",
                "market_key": "small-gap",
                "title": "Will Bitcoin ETF rally in 2026?",
                "one_hour_price_change": 0.08,
                "one_day_price_change": 0.20,
            },
            {
                "venue": "polymarket",
                "market_key": "large-gap",
                "title": "Will Bitcoin price rally in 2026?",
                "one_hour_price_change": 0.0,
                "one_day_price_change": 0.0,
            },
        ]
    monkeypatch.setattr(
        api_server,
        "load_local_polymarket_universe_with_metadata",
        lambda _root: (markets, {"row_count": len(markets), "source_path": "mock.parquet"}),
    )

    resp = client.get("/lag-candidates?min_similarity=0.1&min_divergence=0")

    assert resp.status_code == 200
    scores = [item["divergence_score"] for item in resp.json()["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_lag_candidates_debug_reports_rejection_reasons(monkeypatch) -> None:
    monkeypatch.setattr(
        api_server,
        "get_leader_markets",
        lambda **_kw: [
            {
                "title": "Will Bitcoin rally in 2026?",
                "market_key": "leader-btc",
                "one_hour_price_change": 0.10,
                "one_day_price_change": 0.30,
            }
        ],
    )
    markets = [
            {
                "venue": "polymarket",
                "market_key": "leader-btc",
                "title": "Will Bitcoin rally in 2026?",
                "slug": "btc-self",
            },
            {
                "venue": "polymarket",
                "market_key": "weak-match",
                "title": "Will Ethereum rally in 2026?",
                "slug": "eth-rally",
                "one_hour_price_change": 0.0,
                "one_day_price_change": 0.0,
            },
            {
                "venue": "polymarket",
                "market_key": "dup-match",
                "title": "Will Ethereum rally in 2026?",
                "slug": "eth-rally",
                "one_hour_price_change": 0.0,
                "one_day_price_change": 0.0,
            },
        ]
    monkeypatch.setattr(
        api_server,
        "load_local_polymarket_universe_with_metadata",
        lambda _root: (
            markets,
            {
                "source_path": "mock.parquet",
                "row_count": len(markets),
            },
        ),
    )

    resp = client.get("/lag-candidates/debug?leader_limit=1&min_similarity=0.8")

    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload) == 1
    row = payload[0]
    assert row["leader_market_title"] == "Will Bitcoin rally in 2026?"
    assert row["universe_source_path"] == "mock.parquet"
    assert row["universe_row_count"] == 3
    assert row["number_of_local_universe_markets_searched"] == 3
    candidates = row["top_related_candidates_before_threshold_filtering"]
    assert len(candidates) == 3
    assert any(candidate["excluded_as_self_market"] for candidate in candidates)
    assert any(candidate["excluded_as_duplicate_bucket"] for candidate in candidates)
    assert any("similarity below threshold" in (candidate["rejected_reason"] or "") for candidate in candidates)
    assert "valid_candidate_count" in row
    assert "rejected_candidate_count" in row
    assert "top_rejection_reasons" in row
    assert "leader_skipped" in row
    assert "best_valid_candidates" in row


def test_lag_candidates_skips_leader_with_no_valid_candidates_and_uses_next(monkeypatch) -> None:
    """A leader whose only related markets are rejected should be skipped; the next
    leader that does have valid candidates should supply the output."""
    useless_leader = {
        "title": "Will Japan win the 2026 FIFA World Cup?",
        "market_key": "japan-wc",
        "one_hour_price_change": -0.9,
        "one_day_price_change": -0.9,
    }
    good_leader = {
        "title": "Will a Gaza ceasefire deal happen by June 2026?",
        "market_key": "gaza-ceasefire",
        "one_hour_price_change": 0.4,
        "one_day_price_change": 0.5,
    }
    monkeypatch.setattr(
        api_server,
        "get_leader_markets",
        lambda **_kw: [useless_leader, good_leader],
    )
    markets = [
        {
            "venue": "polymarket",
            "market_key": "taylor-halftime",
            "title": "Will Taylor Swift perform at the 2026 World Cup halftime show?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        },
        {
            "venue": "polymarket",
            "market_key": "gaza-hostages",
            "title": "Will Gaza hostages be released by June 2026?",
            "one_hour_price_change": 0.01,
            "one_day_price_change": 0.02,
        },
    ]
    monkeypatch.setattr(
        api_server,
        "load_local_polymarket_universe_with_metadata",
        lambda _root: (markets, {"row_count": len(markets), "source_path": "mock.parquet"}),
    )

    resp = client.get(
        "/lag-candidates?max_leaders_evaluated=10&min_similarity=0&min_divergence=0"
        "&include_review_only=true&max_abs_leader_move=1"
    )

    assert resp.status_code == 200
    payload = resp.json()
    meta = payload["metadata"]
    assert meta["leaders_skipped"] >= 1
    assert meta["skip_reasons_count"].get("no_valid_candidates", 0) >= 1

    candidates = payload["candidates"]
    assert len(candidates) >= 1
    assert candidates[0]["leader_market_id"] == "gaza-ceasefire"


def test_lag_candidates_returns_empty_with_metadata_when_all_leaders_skipped(monkeypatch) -> None:
    monkeypatch.setattr(
        api_server,
        "get_leader_markets",
        lambda **_kw: [
            {
                "title": "Will Japan win the 2026 FIFA World Cup?",
                "market_key": "japan-wc",
                "one_hour_price_change": -0.9,
                "one_day_price_change": -0.9,
            }
        ],
    )
    monkeypatch.setattr(
        api_server,
        "load_local_polymarket_universe_with_metadata",
        lambda _root: (
            [
                {
                    "venue": "polymarket",
                    "market_key": "taylor-halftime",
                    "title": "Will Taylor Swift perform at the 2026 World Cup halftime show?",
                    "one_hour_price_change": 0.0,
                    "one_day_price_change": 0.0,
                }
            ],
            {"row_count": 1, "source_path": "mock.parquet"},
        ),
    )

    resp = client.get("/lag-candidates?min_similarity=0&min_divergence=0&max_abs_leader_move=1")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["candidates"] == []
    meta = payload["metadata"]
    assert meta["leaders_skipped"] == 1
    assert meta["leaders_evaluated"] == 1
    assert meta["valid_candidates_found"] == 0
    assert "no_valid_candidates" in meta["skip_reasons_count"]


def test_lag_candidates_endpoint_returns_world_cup_same_option_set_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        api_server,
        "get_leader_markets",
        lambda **_kw: [
            {
                "title": "Will Japan win the 2026 FIFA World Cup?",
                "market_key": "japan-wc",
                "one_hour_price_change": -0.9,
                "one_day_price_change": -0.9,
            }
        ],
    )
    markets = [
        {
            "venue": "polymarket",
            "market_key": "brazil-wc",
            "title": "Will Brazil win the 2026 FIFA World Cup?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        },
        {
            "venue": "polymarket",
            "market_key": "france-wc",
            "title": "Will France win the 2026 FIFA World Cup?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        },
        {
            "venue": "polymarket",
            "market_key": "england-wc",
            "title": "Will England win the 2026 FIFA World Cup?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        },
        {
            "venue": "polymarket",
            "market_key": "japan-group-f",
            "title": "Will Japan win Group F at the 2026 FIFA World Cup?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        },
    ]
    monkeypatch.setattr(
        api_server,
        "load_local_polymarket_universe_with_metadata",
        lambda _root: (markets, {"row_count": len(markets), "source_path": "mock.parquet"}),
    )

    resp = client.get(
        "/lag-candidates?min_similarity=0&min_divergence=0&include_review_only=true"
        "&max_abs_leader_move=1&max_candidates_per_leader_output=10&max_candidates_per_event_output=10"
        "&min_related_abs_move_24h=0&min_similarity_for_trade=0"
    )

    assert resp.status_code == 200
    candidates = resp.json()["candidates"]
    candidate_ids = {candidate["related_market_id"] for candidate in candidates}
    assert {"brazil-wc", "france-wc", "england-wc"}.issubset(candidate_ids)
    assert "japan-group-f" not in candidate_ids
    assert all(
        candidate["relationship_type"] == "SAME_OPTION_SET_NEGATIVE_CORRELATION"
        and candidate["expected_correlation_direction"] == "negative"
        for candidate in candidates
    )

    debug_resp = client.get("/lag-candidates/debug?leader_limit=1&min_similarity=0&min_divergence=0&max_abs_leader_move=1")
    assert debug_resp.status_code == 200
    debug_candidates = debug_resp.json()[0]["top_related_candidates_before_threshold_filtering"]
    group_f = next(
        candidate
        for candidate in debug_candidates
        if candidate["related_market_title"] == "Will Japan win Group F at the 2026 FIFA World Cup?"
    )
    assert group_f["relationship_type"] == "SAME_EVENT_OUTCOME"
    assert "SAME_EVENT_OUTCOME" in group_f["rejected_reason"]


def test_lag_candidates_debug_shows_skip_info(monkeypatch) -> None:
    monkeypatch.setattr(
        api_server,
        "get_leader_markets",
        lambda **_kw: [
            {
                "title": "Will Japan win the 2026 FIFA World Cup?",
                "market_key": "japan-wc",
                "one_hour_price_change": -0.9,
                "one_day_price_change": -0.9,
            }
        ],
    )
    monkeypatch.setattr(
        api_server,
        "load_local_polymarket_universe_with_metadata",
        lambda _root: (
            [
                {
                    "venue": "polymarket",
                    "market_key": "taylor-halftime",
                    "title": "Will Taylor Swift perform at the 2026 World Cup halftime show?",
                    "one_hour_price_change": 0.0,
                    "one_day_price_change": 0.0,
                }
            ],
            {"row_count": 1, "source_path": "mock.parquet"},
        ),
    )

    resp = client.get("/lag-candidates/debug?leader_limit=1&min_similarity=0&min_divergence=0&max_abs_leader_move=1")

    assert resp.status_code == 200
    row = resp.json()[0]
    assert row["leader_skipped"] is True
    assert row["skip_reason"] == "no_valid_candidates"
    assert row["valid_candidate_count"] == 0
    assert row["option_set_type"] == "winner"


def test_resolve_market_accepts_side_case_insensitive() -> None:
    body_upper = {"query": "Will CPI YoY be above 3.0%", "side": "YES", "notional_usd": 500}
    body_lower = {"query": "Will CPI YoY be above 3.0%", "side": "yes", "notional_usd": 500}

    r1 = client.post("/resolve_market?mode=fixture", json=body_upper)
    r2 = client.post("/resolve_market?mode=fixture", json=body_lower)

    assert r1.status_code == 200
    assert r2.status_code == 200


def test_resolve_market_returns_not_found_payload_when_no_match() -> None:
    body = {"query": "quantum potato moon price", "side": "yes", "notional_usd": 100}
    resp = client.post("/resolve_market", json=body)

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["resolution_status"] == "not_found"
    assert "candidates" in payload
    assert payload["cross_venue"]["matched_pairs"] == []


def test_resolve_market_returns_503_when_both_venues_fail(monkeypatch) -> None:
    def mock_fetch_poly(*_args, **_kwargs):
        return [], {
            "fetched_candidates": 0,
            "eligible_two_sided": 0,
            "selected": 0,
            "excluded": 0,
            "exclusion_reasons": {"mock": 1},
        }, "none", "live_error: mock poly failure"

    def mock_fetch_kalshi(*_args, **_kwargs):
        return [], {
            "fetched_candidates": 0,
            "eligible_two_sided": 0,
            "selected": 0,
            "excluded": 0,
            "exclusion_reasons": {"mock": 1},
        }, "none", "live_error: mock kalshi failure"

    monkeypatch.setattr(resolver_module, "find_candidates_polymarket", mock_fetch_poly)
    monkeypatch.setattr(resolver_module, "find_candidates_kalshi", mock_fetch_kalshi)

    body = {"query": "Will CPI YoY be above 3.0%", "side": "yes", "notional_usd": 100}
    resp = client.post("/resolve_market", json=body)

    assert resp.status_code == 503
    payload = resp.json()
    assert payload["error_type"] == "venues_unavailable"
    assert "venue_status" in payload
