from __future__ import annotations

import unittest
from pathlib import Path

from src.market_resolver import MarketCandidate, TradeIntent, candidate_score, resolve_trade


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = str(ROOT / "fixtures")


class MarketResolverTests(unittest.TestCase):
    def test_super_bowl_query_prefers_sports(self) -> None:
        intent = TradeIntent(query="Seahawks win the Super Bowl", side="YES", notional_usd=500)
        sports = MarketCandidate(
            venue="kalshi",
            market_key="K.SPORTS",
            title="Will Seahawks win Super Bowl this season?",
            liquidity_proxy=1000,
            volume_proxy=500,
            raw={},
        )
        cpi = MarketCandidate(
            venue="kalshi",
            market_key="K.CPI",
            title="Will CPI YoY be above 3.0%?",
            liquidity_proxy=2000,
            volume_proxy=800,
            raw={},
        )
        self.assertGreater(candidate_score(intent, sports), candidate_score(intent, cpi))

    def test_cpi_query_prefers_cpi_market(self) -> None:
        intent = TradeIntent(query="CPI inflation next print above 3 percent", side="YES", notional_usd=200)
        sports = MarketCandidate(
            venue="polymarket",
            market_key="P.SPORTS",
            title="Will Chiefs make playoffs?",
            liquidity_proxy=1200,
            volume_proxy=700,
            raw={},
        )
        cpi = MarketCandidate(
            venue="polymarket",
            market_key="P.CPI",
            title="Will CPI YoY be above 3.0% in next print?",
            liquidity_proxy=1200,
            volume_proxy=700,
            raw={},
        )
        self.assertGreater(candidate_score(intent, cpi), candidate_score(intent, sports))

    def test_category_hint_shifts_ranking(self) -> None:
        base_intent = TradeIntent(query="winner market", side="YES", notional_usd=150)
        hinted_intent = TradeIntent(query="winner market", side="YES", notional_usd=150, category_hint="sports")

        macro_market = MarketCandidate(
            venue="kalshi",
            market_key="K.MACRO",
            title="Will the Fed cut rates by June 2026?",
            liquidity_proxy=1000,
            volume_proxy=400,
            raw={},
        )
        sports_market = MarketCandidate(
            venue="kalshi",
            market_key="K.SPORTS",
            title="Will team win playoffs?",
            liquidity_proxy=1000,
            volume_proxy=400,
            raw={},
        )

        base_gap = candidate_score(base_intent, sports_market) - candidate_score(base_intent, macro_market)
        hinted_gap = candidate_score(hinted_intent, sports_market) - candidate_score(hinted_intent, macro_market)
        self.assertGreater(hinted_gap, base_gap)

    def test_low_match_quality(self) -> None:
        intent = TradeIntent(query="quantum potatoes over moon", side="YES", notional_usd=100)
        resolved = resolve_trade(intent=intent, mode="fixture", fixtures_dir=FIXTURES, allow_fallback=True)
        self.assertEqual(resolved.match_quality, "LOW")

    def test_deterministic_ordering(self) -> None:
        intent = TradeIntent(query="Will CPI be above 3.0", side="YES", notional_usd=100)
        r1 = resolve_trade(intent=intent, mode="fixture", fixtures_dir=FIXTURES, allow_fallback=True)
        r2 = resolve_trade(intent=intent, mode="fixture", fixtures_dir=FIXTURES, allow_fallback=True)

        self.assertEqual(r1.best_kalshi, r2.best_kalshi)
        self.assertEqual(r1.best_polymarket, r2.best_polymarket)
        self.assertEqual(r1.candidates, r2.candidates)

    def test_score_fields_present(self) -> None:
        intent = TradeIntent(query="Will CPI YoY be above 3.0%", side="YES", notional_usd=250)
        resolved = resolve_trade(intent=intent, mode="fixture", fixtures_dir=FIXTURES, allow_fallback=True)

        self.assertIsNotNone(resolved.best_kalshi)
        self.assertIsNotNone(resolved.best_polymarket)
        self.assertIsNotNone(resolved.best_kalshi.get("execution_risk_index"))
        self.assertIsNotNone(resolved.best_polymarket.get("execution_risk_index"))


if __name__ == "__main__":
    unittest.main()
