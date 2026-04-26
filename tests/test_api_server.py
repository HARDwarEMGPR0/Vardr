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


def test_leader_markets_returns_list_with_required_keys() -> None:
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


def test_leader_markets_limit_parameter_works() -> None:
    resp = client.get("/leader-markets?limit=1&min_abs_move=0")

    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_leader_markets_min_abs_move_filters_low_move_leaders() -> None:
    resp = client.get("/leader-markets?min_abs_move=0.06")

    assert resp.status_code == 200
    assert resp.json() == []


def test_leader_markets_fallback_demo_leader_appears() -> None:
    resp = client.get("/leader-markets")

    assert resp.status_code == 200
    titles = {item["title"] for item in resp.json()}
    assert "Will bitcoin hit $1m before GTA VI?" in titles


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
    assert payload[0]["market_key"].startswith("vardr_")


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
