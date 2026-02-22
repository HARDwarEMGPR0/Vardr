from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.fetch_snapshots import build_cross_venue, run_pipeline
from src.connectors.kalshi_public import fetch_markets_fixture as fetch_kalshi_fixture


ROOT = Path(__file__).resolve().parents[1]


class FetchSnapshotTests(unittest.TestCase):
    def test_a_fixture_mode_quality(self) -> None:
        args = argparse.Namespace(
            mode="fixture",
            fixtures_dir=str(ROOT / "fixtures"),
            poly_limit=200,
            kalshi_limit=200,
            n_markets=5,
            n_kalshi=5,
            n_polymarket=5,
            include_mve=False,
            mapping=None,
            save_fixtures=False,
            allow_fallback=False,
        )
        payload = run_pipeline(args)

        poly = payload["polymarket_snapshots"]
        kalshi = payload["kalshi_snapshots"]

        poly_good = [
            s
            for s in poly
            if s.get("best_yes_bid") is not None
            and s.get("best_yes_ask") is not None
            and s.get("best_no_bid") is not None
            and s.get("best_no_ask") is not None
            and s.get("spread") is not None
        ]
        kalshi_good = [
            s
            for s in kalshi
            if s.get("best_yes_bid") is not None
            and s.get("best_yes_ask") is not None
            and s.get("best_no_bid") is not None
            and s.get("best_no_ask") is not None
            and s.get("spread") is not None
        ]

        self.assertGreaterEqual(len(poly_good), 2)
        self.assertGreaterEqual(len(kalshi_good), 2)

    def test_b_selection_logic_excludes_mve_and_one_sided(self) -> None:
        selection_dir = ROOT / "tests" / "fixtures" / "selection"
        selected, debug, _, _ = fetch_kalshi_fixture(
            fixtures_dir=selection_dir,
            n_markets=5,
            include_mve=False,
        )

        tickers = {item.get("ticker") for item in selected}
        self.assertNotIn("KXMVESPORTSMULTIGAMEEXTENDED-ABC", tickers)
        self.assertNotIn("KX.REGULAR.2", tickers)  # one-sided no book side
        self.assertGreaterEqual(debug.get("exclusion_reasons", {}).get("excluded_mve_bundle", 0), 1)
        self.assertGreaterEqual(debug.get("exclusion_reasons", {}).get("not_two_sided", 0), 1)

    def test_c_matching_world_cup_style(self) -> None:
        kalshi = [
            {
                "market_key": "KX.WC.GROUPA",
                "title": "FIFA World Cup Group A Winner",
            },
            {
                "market_key": "KX.WC.GROUPB",
                "title": "FIFA World Cup Group B Winner",
            },
        ]
        poly = [
            {
                "market_key": "0xwc_a",
                "title": "World Cup Group A winner market",
            },
            {
                "market_key": "0xother",
                "title": "Fed rate decision in June",
            },
        ]

        matched, _, _ = build_cross_venue(kalshi, poly, threshold=0.35)
        self.assertGreaterEqual(len(matched), 1)


if __name__ == "__main__":
    unittest.main()
