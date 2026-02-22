from __future__ import annotations

from fastapi.testclient import TestClient

import src.market_resolver as resolver_module
from api.server import app


client = TestClient(app)


def test_health_ok() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


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
