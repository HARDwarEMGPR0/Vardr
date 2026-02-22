from __future__ import annotations

from fastapi.testclient import TestClient

from api.server import app


client = TestClient(app)


def test_resolve_market_fixture_mode_returns_expected_keys() -> None:
    payload = {
        "query": "Will CPI YoY be above 3.0%",
        "side": "yes",
        "notional_usd": 250,
    }
    resp = client.post("/resolve_market?mode=fixture", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "venue_status" in body
    assert ("polymarket_snapshots" in body) or ("kalshi_snapshot" in body)
    assert "scores" in body
