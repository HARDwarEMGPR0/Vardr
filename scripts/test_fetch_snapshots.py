from __future__ import annotations

import argparse
from pathlib import Path

import requests

from scripts.fetch_snapshots import determine_exit_code, run_pipeline


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def _args(mode: str, allow_fallback: bool, n_markets: int = 5) -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        fixtures_dir=str(FIXTURES),
        poly_limit=200,
        kalshi_limit=200,
        n_markets=n_markets,
        mapping=None,
        save_fixtures=False,
        allow_fallback=allow_fallback,
    )


def test_fixture_mode_has_rich_snapshots() -> None:
    n = 5
    payload = run_pipeline(_args(mode="fixture", allow_fallback=False, n_markets=n))

    assert determine_exit_code(payload, "fixture") == 0
    assert payload["venue_status"]["polymarket"]["source"] == "fixture"
    assert payload["venue_status"]["kalshi"]["source"] == "fixture"

    poly = payload["polymarket_snapshots"]
    kalshi = payload["kalshi_snapshots"]

    assert len(poly) >= n
    assert len(kalshi) >= n

    poly_good = [s for s in poly if s.get("best_yes_bid") is not None and s.get("best_yes_ask") is not None and s.get("spread_reported") is not None]
    assert len(poly_good) >= 3

    kalshi_good = [s for s in kalshi if s.get("best_yes_bid") is not None or s.get("best_no_bid") is not None]
    assert len(kalshi_good) >= 3

    for s in kalshi:
        if s.get("best_yes_bid") is not None and s.get("best_no_bid") is not None:
            assert s.get("spread") is not None

    cross = payload["cross_venue"]
    assert len(cross["matched_pairs"]) <= 2
    assert len(cross["kalshi_only_examples"]) == 2
    assert len(cross["polymarket_only_examples"]) == 2


def test_live_no_fallback_network_failure(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise requests.ConnectionError("[WinError 10013] blocked")

    monkeypatch.setattr(requests, "get", _raise)

    payload = run_pipeline(_args(mode="live", allow_fallback=False, n_markets=5))
    assert determine_exit_code(payload, "live") == 1
    assert payload["venue_status"]["polymarket"]["source"] == "none"
    assert payload["venue_status"]["kalshi"]["source"] == "none"


def test_live_with_fallback_network_failure(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise requests.ConnectionError("[WinError 10013] blocked")

    monkeypatch.setattr(requests, "get", _raise)

    payload = run_pipeline(_args(mode="live", allow_fallback=True, n_markets=5))
    assert determine_exit_code(payload, "live") == 0
    assert payload["venue_status"]["polymarket"]["source"] == "fixture"
    assert payload["venue_status"]["kalshi"]["source"] == "fixture"
