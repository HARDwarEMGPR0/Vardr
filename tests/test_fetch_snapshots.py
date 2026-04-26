from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


    def _fixture_args(self, **overrides) -> argparse.Namespace:
        base = dict(
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
            leader_markets_json=None,
            vardr_leader_api_url=None,
            history_jsonl=None,
            debug_intelligence=False,
            summary=False,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def _fixture_payload(self) -> dict:
        args = self._fixture_args()
        return run_pipeline(args)

    def test_d_intelligence_block_has_all_keys(self) -> None:
        intel = self._fixture_payload()["intelligence"]
        self.assertIn("opportunities", intel)
        self.assertIn("reference_event_clusters", intel)
        self.assertIn("inconsistencies", intel)
        self.assertIn("lag_signals", intel)
        self.assertIn("review_candidates", intel)
        self.assertIsInstance(intel["opportunities"], list)
        self.assertIsInstance(intel["reference_event_clusters"], list)
        self.assertIsInstance(intel["inconsistencies"], list)
        self.assertIsInstance(intel["lag_signals"], list)
        self.assertIsInstance(intel["review_candidates"], list)

    def test_f_pipeline_accepts_leader_markets_json(self) -> None:
        leader = {
            "title": "Will Bitcoin hit 100k before GTA VI?",
            "market_key": "btc-100k-leader",
            "reference_event": None,
            "one_hour_price_change": 0.06,
            "one_day_price_change": 0.10,
            "mid": 0.70,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump([leader], f)
            tmp_path = f.name

        try:
            payload = run_pipeline(self._fixture_args(leader_markets_json=tmp_path))
        finally:
            Path(tmp_path).unlink()

        intel = payload["intelligence"]
        self.assertIn("lag_signals", intel)
        self.assertIsInstance(intel["lag_signals"], list)

    def test_f_pipeline_accepts_vardr_leader_api_url(self) -> None:
        leader = {
            "title": "Will Bitcoin hit 100k before GTA VI?",
            "market_key": "btc-api-leader",
            "one_hour_price_change": 0.06,
        }
        with patch("scripts.fetch_snapshots.load_leader_markets_from_vardr_api", return_value=[leader]) as loader:
            payload = run_pipeline(
                self._fixture_args(vardr_leader_api_url="http://localhost:8000/leader-markets")
            )

        loader.assert_called_once_with("http://localhost:8000/leader-markets")
        self.assertIn("lag_signals", payload["intelligence"])
        self.assertIn("review_candidates", payload["intelligence"])

    def test_g_history_jsonl_writes_one_line(self) -> None:
        from scripts.fetch_snapshots import _append_run_to_jsonl

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp = f.name
        Path(tmp).unlink()
        try:
            _append_run_to_jsonl(tmp, {"run_id": "r1"})
            lines = [l for l in Path(tmp).read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["run_id"], "r1")
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_h_history_jsonl_appends_not_overwrites(self) -> None:
        from scripts.fetch_snapshots import _append_run_to_jsonl

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp = f.name
        Path(tmp).unlink()
        try:
            _append_run_to_jsonl(tmp, {"run_id": "first"})
            _append_run_to_jsonl(tmp, {"run_id": "second"})
            lines = [l for l in Path(tmp).read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["run_id"], "first")
            self.assertEqual(json.loads(lines[1])["run_id"], "second")
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_i_computed_mid_delta_present_with_prior_snapshot(self) -> None:
        # First run: build a payload to act as the "previous" snapshot
        first_payload = self._fixture_payload()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            json.dump(first_payload, f)
            f.write("\n")
            tmp = f.name

        try:
            second_payload = run_pipeline(self._fixture_args(history_jsonl=tmp))
            for snap in second_payload["polymarket_snapshots"]:
                self.assertIn("computed_mid_delta", snap)
            # Same fixture data → deltas are 0.0 for matching markets (not None)
            non_none = [
                s["computed_mid_delta"]
                for s in second_payload["polymarket_snapshots"]
                if s["computed_mid_delta"] is not None
            ]
            self.assertGreater(len(non_none), 0)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_j_summary_helper_includes_intelligence_summary(self) -> None:
        import io
        from scripts.fetch_snapshots import _print_intelligence_summary

        payload = self._fixture_payload()
        buf = io.StringIO()
        _print_intelligence_summary(payload, file=buf)
        output = buf.getvalue()
        self.assertIn("INTELLIGENCE SUMMARY", output)
        self.assertIn("Opportunities", output)
        self.assertIn("Lag signals", output)
        self.assertIn("Review candidates", output)

    def test_e_stale_reason_text_absent(self) -> None:
        import json
        payload_str = json.dumps(self._fixture_payload())
        self.assertNotIn("Unrelated events priced similarly", payload_str)
        self.assertNotIn("narrative distortion", payload_str)


if __name__ == "__main__":
    unittest.main()
