import json
import tempfile
from pathlib import Path

import market_fetcher.intelligence as intelligence
from market_fetcher.intelligence import (
    find_opportunities,
    detect_inconsistencies,
    detect_reference_event_clusters,
    detect_lagging_correlated_markets,
    detect_lagging_correlated_markets_with_review,
    score_market_relationship,
)
from market_fetcher.leader_markets import load_leader_markets, load_leader_markets_from_vardr_api
import market_fetcher.lag_candidates as lag_candidates_module
from market_fetcher.lag_candidates import (
    build_lag_candidates,
    build_lag_candidates_debug,
    build_lag_candidates_with_metadata,
    enrich_leader_markets_with_universe_movement,
    load_local_polymarket_universe_with_metadata,
)
from market_fetcher.vardr1_client import fetch_vardr1_leader_markets, vardr1_last_fetch_metadata
from market_fetcher.vardr1_client import map_vardr1_market_to_leader
from src.normalize import normalize_execution_market, normalize_kalshi_market
from src.scoring import compute_execution_risk_details


def _market(title, spread, depth_top5, mid, market_key="key-1"):
    return {
        "title": title,
        "market_key": market_key,
        "spread": spread,
        "depth_top5": depth_top5,
        "mid": mid,
    }


def _gta_market(title, mid, market_key):
    return {
        "title": title,
        "market_key": market_key,
        "mid": mid,
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
        "volume_proxy": 1000,
        "liquidity_proxy": 500,
    }


# ---------------------------------------------------------------------------
# find_opportunities
# ---------------------------------------------------------------------------

def test_find_opportunities_flags_qualifying_market():
    markets = [_market("Will X happen?", spread=0.005, depth_top5=100_000, mid=0.5)]
    results = find_opportunities(markets)
    assert len(results) == 1
    r = results[0]
    assert r["type"] == "high_quality_market"
    assert r["market_key"] == "key-1"
    assert r["spread"] == 0.005
    assert r["depth_top5"] == 100_000
    assert r["mid"] == 0.5
    assert "reason" in r


def test_find_opportunities_skips_missing_fields():
    markets = [
        {"title": "No spread", "market_key": "a", "depth_top5": 100_000, "mid": 0.5},
        {"title": "No depth", "market_key": "b", "spread": 0.005, "mid": 0.5},
        {"title": "No mid", "market_key": "c", "spread": 0.005, "depth_top5": 100_000},
    ]
    assert find_opportunities(markets) == []


def test_normalize_execution_market_uses_common_schema_for_polymarket():
    normalized = normalize_execution_market(
        {
            "venue": "polymarket",
            "market_key": "pm-1",
            "last_trade_price": 0.51,
            "best_yes_bid": 0.50,
            "best_yes_ask": 0.54,
            "mid": 0.52,
            "spread": 0.04,
            "depth_top5": 1200,
        }
    )

    assert normalized == {
        "market_id": "pm-1",
        "platform": "polymarket",
        "price": 0.51,
        "best_bid": 0.5,
        "best_ask": 0.54,
        "midpoint": 0.52,
        "spread": 0.04,
        "depth_top5": 1200.0,
    }


def test_normalize_execution_market_reconstructs_kalshi_yes_ask_from_no_bid():
    kalshi = normalize_kalshi_market(
        "KX.TEST",
        {"title": "Will test happen?"},
        {"orderbook": {"yes": [[45, 10]], "no": [[52, 12]]}},
        {"trades": []},
    )
    normalized = normalize_execution_market(kalshi)

    assert normalized["market_id"] == "KX.TEST"
    assert normalized["platform"] == "kalshi"
    assert normalized["best_bid"] == 45.0
    assert normalized["best_ask"] == 48.0
    assert normalized["midpoint"] == 46.5
    assert normalized["spread"] == 3.0


def test_compute_execution_risk_details_returns_product_label():
    risk = compute_execution_risk_details({"venue": "polymarket", "spread": 0.20, "depth_top5": 0})

    assert 0 <= risk["risk_score"] <= 1
    assert risk["risk_label"] in {"LOW", "MEDIUM", "HIGH"}
    assert "drivers" in risk


def test_vardr1_leader_mapping_outputs_standard_schema_and_score():
    leader = map_vardr1_market_to_leader(
        {
            "market_id": "m1",
            "market_title": "Will test happen?",
            "current_price": 0.42,
            "price_change_1h": 0.12,
            "price_change_24h": -0.2,
            "liquidity": 50_000,
        }
    )

    assert leader["market_id"] == "m1"
    assert leader["market_title"] == "Will test happen?"
    assert leader["current_price"] == 0.42
    assert leader["price_change_1h"] == 0.12
    assert leader["price_change_24h"] == -0.2
    assert leader["leader_score"] > 0


def test_vardr1_leader_mapping_accepts_current_suspicious_schema():
    leader = map_vardr1_market_to_leader(
        {
            "market_id": "0xabc",
            "market_title": "Will test happen?",
            "price": 0.0868,
            "risk_score": 0.334,
            "anomaly_score": 0.605,
            "trade_size": 121.1,
            "platform": "polymarket",
            "_window": "24h",
        }
    )

    assert leader["market_id"] == "0xabc"
    assert leader["market_title"] == "Will test happen?"
    assert leader["current_price"] == 0.0868
    assert leader["price_change_1h"] == 0.0
    assert leader["price_change_24h"] == 0.0
    assert leader["leader_score"] == 0.605
    assert leader["platform"] == "polymarket"
    assert leader["recent_volume"] == 121.1
    assert leader["source"] == "vardr1_anomaly_fallback"
    assert leader["_has_movement_fields"] is False
    assert leader["movement_source"] == "anomaly_score"


def test_vardr1_leader_mapping_accepts_market_movement_schema():
    leader = map_vardr1_market_to_leader(
        {
            "market_id": "0xmove",
            "market_title": "Will movement happen?",
            "current_price": 0.44,
            "price_change_1h": 0.07,
            "one_day_price_change": -0.12,
            "leader_score": 0.9,
            "platform": "polymarket",
            "recent_volume": 5000,
            "stale_or_resolved": False,
        }
    )

    assert leader["market_id"] == "0xmove"
    assert leader["current_price"] == 0.44
    assert leader["price_change_1h"] == 0.07
    assert leader["price_change_24h"] == -0.12
    assert leader["leader_score"] == 0.9
    assert leader["source"] == "vardr1_market_movement"
    assert leader["stale_or_resolved"] is False


def test_suspicious_leader_gets_universe_price_movement_by_market_id():
    leader = map_vardr1_market_to_leader(
        {
            "market_id": "fed-cut-leader",
            "market_title": "Will the Fed cut rates by June 2026?",
            "price": 0.58,
            "anomaly_score": 0.81,
            "risk_score": 0.31,
        }
    )
    universe = [
        {
            "venue": "polymarket",
            "market_id": "fed-cut-leader",
            "market_key": "fed-cut-leader",
            "title": "Will the Fed cut rates by June 2026?",
            "current_price": 0.61,
            "price_change_1h": 0.04,
            "price_change_24h": 0.21,
            "volume": 12_500,
            "liquidity": 80_000,
            "active": True,
            "closed": False,
            "end_date": "2026-06-30T00:00:00Z",
        }
    ]

    enriched, meta = enrich_leader_markets_with_universe_movement([leader], universe)

    enriched_leader = enriched[0]
    assert enriched_leader["price_change_1h"] == 0.04
    assert enriched_leader["price_change_24h"] == 0.21
    assert enriched_leader["current_price"] == 0.61
    assert enriched_leader["volume"] == 12_500
    assert enriched_leader["liquidity"] == 80_000
    assert enriched_leader["active"] is True
    assert enriched_leader["closed"] is False
    assert enriched_leader["end_date"] == "2026-06-30T00:00:00Z"
    assert enriched_leader["movement_source"] == "universe_match"
    assert enriched_leader["enrichment_source"] == "universe_match"
    assert enriched_leader["anomaly_score"] == 0.81
    assert enriched_leader["leader_score"] == 0.81
    assert meta["leaders_enriched_with_universe_movement"] == 1
    assert meta["leaders_missing_real_movement"] == 0
    assert meta["movement_source_counts"]["universe_match"] == 1


def test_suspicious_leader_derives_movement_from_snapshot_history_when_universe_lacks_changes():
    leader = map_vardr1_market_to_leader(
        {
            "market_id": "history-leader",
            "market_title": "Will historical movement happen?",
            "price": 0.70,
            "anomaly_score": 0.77,
        }
    )
    universe = [
        {
            "venue": "polymarket",
            "market_id": "history-leader",
            "market_key": "history-leader",
            "title": "Will historical movement happen?",
            "current_price": 0.70,
        }
    ]
    snapshot_history = [
        {
            "ts_utc": "2026-04-28T00:00:00+00:00",
            "polymarket_snapshots": [
                {"market_key": "history-leader", "title": "Will historical movement happen?", "mid": 0.50}
            ],
        },
        {
            "ts_utc": "2026-04-28T23:00:00+00:00",
            "polymarket_snapshots": [
                {"market_key": "history-leader", "title": "Will historical movement happen?", "mid": 0.65}
            ],
        },
        {
            "ts_utc": "2026-04-29T00:00:00+00:00",
            "polymarket_snapshots": [
                {"market_key": "history-leader", "title": "Will historical movement happen?", "mid": 0.70}
            ],
        },
    ]

    enriched, meta = enrich_leader_markets_with_universe_movement(
        [leader],
        universe,
        snapshot_history=snapshot_history,
    )

    enriched_leader = enriched[0]
    assert round(enriched_leader["price_change_1h"], 6) == 0.05
    assert round(enriched_leader["price_change_24h"], 6) == 0.20
    assert enriched_leader["movement_source"] == "snapshot_history"
    assert enriched_leader["enrichment_source"] == "snapshot_history"
    assert enriched_leader["anomaly_score"] == 0.77
    assert meta["leaders_enriched_with_snapshot_movement"] == 1
    assert meta["leaders_missing_real_movement"] == 0


def test_enriched_suspicious_causal_pair_can_be_primary_trade():
    leader = map_vardr1_market_to_leader(
        {
            "market_id": "gaza-ceasefire",
            "market_title": "Will a Gaza ceasefire deal happen?",
            "price": 0.58,
            "anomaly_score": 0.81,
            "risk_score": 0.31,
        }
    )
    leader["_lag_tokens"] = {"gaza", "ceasefire", "deal", "hostages", "released"}
    self_row = {
        "venue": "polymarket",
        "market_id": "gaza-ceasefire",
        "market_key": "gaza-ceasefire",
        "title": "Will a Gaza ceasefire deal happen?",
        "current_price": 0.61,
        "price_change_1h": 0.16,
        "price_change_24h": 0.21,
        "active": True,
        "closed": False,
    }
    related = {
        "venue": "polymarket",
        "market_key": "gaza-hostages",
        "title": "Will Gaza hostages be released?",
        "one_hour_price_change": 0.01,
        "one_day_price_change": 0.02,
        "current_price": 0.44,
        "spread": 0.01,
        "depth_top5": 100_000,
        "_lag_tokens": {"gaza", "ceasefire", "deal", "hostages", "released"},
    }

    candidates, meta = build_lag_candidates_with_metadata(
        [leader],
        [self_row, related],
        min_similarity=0,
        min_divergence=0,
        min_similarity_for_trade=0,
        min_related_abs_move_24h=0.01,
    )

    assert candidates
    candidate = candidates[0]
    assert candidate["leader_movement_source"] == "universe_match"
    assert candidate["leader_anomaly_score"] == 0.81
    assert candidate["tradable_signal"] is True
    assert candidate["trade_bucket"] == "primary"
    assert candidate["suggested_trade_direction"] == "buy_yes_related"
    assert meta["leaders_enriched_with_universe_movement"] == 1
    assert meta["leaders_missing_real_movement"] == 0


def test_lag_candidates_does_not_low_info_skip_suspicious_anomaly_leader():
    leader = map_vardr1_market_to_leader(
        {
            "ts": "2026-04-26 18:23:44+00:00",
            "platform": "polymarket",
            "market_id": "0x9c1a",
            "market_title": "Russia-Ukraine Ceasefire before GTA VI?",
            "price": 0.47,
            "trade_size": 970.17,
            "anomaly_score": 0.7181171196558715,
            "raw_risk": 0.7181171196558715,
            "risk_score": 0.30879036145202476,
            "_window": "24h",
            "_source_file": "suspicious_24h.csv",
        }
    )
    universe = [
        {
            "venue": "polymarket",
            "market_key": "russia-sanctions",
            "title": "Will Russia sanctions happen before GTA VI?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
            "current_price": 0.31,
            "liquidity_proxy": 1000,
        }
    ]

    candidates, meta = build_lag_candidates_with_metadata(
        [leader],
        universe,
        exploratory=True,
        min_similarity=0,
        min_divergence=0,
    )

    assert meta["skip_reasons_count"].get("low_information_leader", 0) == 0
    assert meta["leaders_skipped"] == 0
    assert candidates
    assert candidates[0]["leader_movement_source"] == "anomaly_score"
    assert candidates[0]["trade_bucket"] == "review"
    assert "anomaly/risk score" in candidates[0]["review_reason"]


def test_vardr1_leader_mapping_zero_valued_movement_fields_classified_as_anomaly_source():
    """Zero-valued price_change fields (e.g. from /api/suspicious) should not count as movement."""
    leader = map_vardr1_market_to_leader(
        {
            "market_id": "0xsus1",
            "market_title": "Will Russia impose new sanctions before April 2026?",
            "price": 0.42,
            "price_change_1h": 0.0,
            "price_change_24h": 0.0,
            "anomaly_score": 0.71,
            "risk_score": 0.31,
            "platform": "polymarket",
        }
    )
    assert leader["source"] == "vardr1_anomaly_fallback", f"got source={leader['source']!r}"
    assert leader["_has_movement_fields"] is False
    assert leader["movement_source"] == "anomaly_score"
    assert leader["leader_score"] == 0.71
    assert leader["price_change_1h"] == 0.0
    assert leader["price_change_24h"] == 0.0


def test_vardr1_leader_mapping_zero_score_anomaly_source_still_valid():
    """Zero anomaly_score from /api/suspicious should not prevent processing (score is absent/zero, not negative)."""
    leader = map_vardr1_market_to_leader(
        {
            "market_id": "0xzero",
            "market_title": "Will EU sanctions on Russia be expanded?",
            "price": 0.35,
            "price_change_1h": 0.0,
            "price_change_24h": 0.0,
            "anomaly_score": 0.0,
            "risk_score": 0.0,
            "platform": "polymarket",
        }
    )
    assert leader["source"] == "vardr1_anomaly_fallback"
    assert leader["_has_movement_fields"] is False
    # leader_score comes from anomaly_score=0.0, so compute_leader_score is NOT called
    assert leader["leader_score"] == 0.0
    universe = [
        {
            "venue": "polymarket",
            "market_key": "eu-russia-sanctions",
            "title": "Will EU impose new Russia sanctions before June 2026?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
            "current_price": 0.48,
        }
    ]
    candidates, meta = build_lag_candidates_with_metadata(
        [leader],
        universe,
        exploratory=True,
        min_similarity=0,
        min_divergence=0,
    )
    assert meta["skip_reasons_count"].get("low_information_leader", 0) == 0, (
        f"zero-score anomaly leader was skipped: {meta['skip_reasons_count']}"
    )
    assert meta["leaders_skipped"] == 0


def test_lag_candidates_exploratory_returns_review_candidate_for_suspicious_leaders():
    """Realistic /api/suspicious payload: leader with zero movement but valid title+score
    should produce review_candidates when exploratory=True and a related market exists."""
    leaders = [
        map_vardr1_market_to_leader(
            {
                "market_id": "0xsanction",
                "market_title": "Will the US impose new tariffs on Chinese goods?",
                "price": 0.67,
                "price_change_1h": 0.0,
                "price_change_24h": 0.0,
                "anomaly_score": 0.82,
                "risk_score": 0.41,
                "trade_size": 500.0,
                "platform": "polymarket",
                "_window": "24h",
                "_source_file": "suspicious_24h.csv",
            }
        )
    ]
    universe = [
        {
            "venue": "polymarket",
            "market_key": "china-tariff-impact",
            "market_id": "china-tariff-impact",
            "title": "Will US tariffs on China exceed 50% by end of 2026?",
            "question": "Will US tariffs on China exceed 50% by end of 2026?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
            "current_price": 0.55,
            "liquidity": 50_000,
            "active": True,
            "closed": False,
        },
        {
            "venue": "polymarket",
            "market_key": "china-retaliation",
            "market_id": "china-retaliation",
            "title": "Will China retaliate against US tariffs with its own trade restrictions?",
            "question": "Will China retaliate against US tariffs with its own trade restrictions?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
            "current_price": 0.60,
            "liquidity": 30_000,
            "active": True,
            "closed": False,
        },
    ]
    candidates, meta = build_lag_candidates_with_metadata(
        leaders,
        universe,
        exploratory=True,
        min_similarity=0.0,
        min_divergence=0.0,
        min_similarity_for_trade=0.35,
        min_related_abs_move_24h=0.0,
    )
    assert meta["skip_reasons_count"].get("low_information_leader", 0) == 0, (
        f"suspicious leaders with nonzero anomaly_score should not be low_information_leader: "
        f"{meta['skip_reasons_count']}"
    )
    assert meta["leaders_skipped"] == 0
    assert candidates, (
        f"expected at least one review candidate for tariff leader vs tariff lag market; "
        f"top_rejection_reasons={meta.get('top_candidate_rejection_reasons')}"
    )
    candidate = candidates[0]
    assert candidate["trade_bucket"] == "review"
    assert candidate["leader_movement_source"] == "anomaly_score"
    assert candidate["leader_score"] == 0.82


def test_lag_similarity_uses_text_vector_beyond_token_overlap():
    leader = {"title": "Will the Federal Reserve cut interest rates by June 2026?"}
    related = {"title": "Fed rate reduction before June 2026?"}

    score = lag_candidates_module._similarity_score(leader, related)

    assert score > 0


# ---------------------------------------------------------------------------
# detect_inconsistencies (legacy shape — backward compat)
# ---------------------------------------------------------------------------

def test_detect_inconsistencies_clustered_gta_markets():
    markets = [
        _market("Will Y happen before GTA VI?", spread=0.01, depth_top5=1000, mid=0.50, market_key="g1"),
        _market("Will Z happen before GTA VI?", spread=0.01, depth_top5=1000, mid=0.52, market_key="g2"),
        _market("Will W happen before GTA VI?", spread=0.01, depth_top5=1000, mid=0.54, market_key="g3"),
    ]
    signals = detect_inconsistencies(markets)
    assert len(signals) == 1
    s = signals[0]
    assert s["type"] == "clustered_probabilities"
    assert s["theme"] == "GTA VI reference markets"
    assert s["price_range"] == 0.54 - 0.50
    assert len(s["markets"]) == 3
    assert all("market_key" in m for m in s["markets"])
    assert "reason" in s


def test_detect_inconsistencies_empty_when_fewer_than_three():
    markets = [
        _market("Will Y happen before GTA VI?", spread=0.01, depth_top5=1000, mid=0.50, market_key="g1"),
        _market("Will Z happen before GTA VI?", spread=0.01, depth_top5=1000, mid=0.51, market_key="g2"),
    ]
    assert detect_inconsistencies(markets) == []


# ---------------------------------------------------------------------------
# detect_reference_event_clusters
# ---------------------------------------------------------------------------

def test_detect_reference_event_clusters_returns_linked_signal():
    markets = [
        _gta_market("Will Y happen before GTA VI?", mid=0.50, market_key="g1"),
        _gta_market("Will Z happen before GTA VI?", mid=0.52, market_key="g2"),
        _gta_market("Will W happen before GTA VI?", mid=0.54, market_key="g3"),
    ]
    signals = detect_reference_event_clusters(markets)
    assert len(signals) == 1
    s = signals[0]
    assert s["type"] == "linked_reference_markets"
    assert s["reference_event"] == "GTA VI"
    assert s["rule_type"] == "before_reference_event"
    assert len(s["markets"]) == 3
    for m in s["markets"]:
        assert "market_key" in m
        assert "one_hour_price_change" in m
        assert "one_day_price_change" in m
        assert "volume_proxy" in m
        assert "liquidity_proxy" in m


def test_detect_reference_event_clusters_reason_mentions_resolution_rule():
    markets = [
        _gta_market("Will Y happen before GTA VI?", mid=0.50, market_key="g1"),
        _gta_market("Will Z happen before GTA VI?", mid=0.52, market_key="g2"),
        _gta_market("Will W happen before GTA VI?", mid=0.54, market_key="g3"),
    ]
    signals = detect_reference_event_clusters(markets)
    assert signals
    reason = signals[0]["reason"].lower()
    assert "resolution" in reason


def test_detect_inconsistencies_backward_compat_shape():
    markets = [
        _gta_market("Will Y happen before GTA VI?", mid=0.50, market_key="g1"),
        _gta_market("Will Z happen before GTA VI?", mid=0.52, market_key="g2"),
        _gta_market("Will W happen before GTA VI?", mid=0.54, market_key="g3"),
    ]
    signals = detect_inconsistencies(markets)
    assert len(signals) == 1
    s = signals[0]
    assert s["type"] == "clustered_probabilities"
    assert s["theme"] == "GTA VI reference markets"
    assert "price_range" in s
    assert "reason" in s


# ---------------------------------------------------------------------------
# detect_lagging_correlated_markets
# ---------------------------------------------------------------------------

def test_detect_lagging_correlated_markets_flags_laggard():
    leader = {
        "title": "Will Bitcoin hit 100k this year?",
        "market_key": "btc-100k",
        "mid": 0.60,
        "one_hour_price_change": 0.08,
        "one_day_price_change": 0.10,
    }
    laggard = {
        "title": "Will Bitcoin ETF get approval this year?",
        "market_key": "btc-etf",
        "mid": 0.55,
        "one_hour_price_change": 0.001,
        "one_day_price_change": 0.002,
    }
    signals = detect_lagging_correlated_markets([laggard], [leader])
    assert len(signals) == 1
    s = signals[0]
    assert s["type"] == "lagging_correlated_market"
    assert s["leader_market_key"] == "btc-100k"
    assert s["laggard_market_key"] == "btc-etf"
    assert s["relationship"] == "same_semantic_group"
    assert s["relationship_score"] >= 0.45
    assert s["relationship_reasons"]
    assert s["expected_direction"] == "same"
    assert "reason" in s


def test_self_market_key_does_not_emit_lag_signal():
    leader = {
        "title": "Will Bitcoin hit 100k this year?",
        "market_key": "btc-same",
        "one_hour_price_change": 0.08,
    }
    laggard = {
        "title": "Will Bitcoin hit 100k this year?",
        "market_key": "btc-same",
        "one_hour_price_change": 0.0,
    }

    assert detect_lagging_correlated_markets([laggard], [leader]) == []


def test_same_normalized_title_does_not_emit_lag_signal():
    leader = {
        "title": "Will Bitcoin hit 100k this year?",
        "market_key": "btc-leader",
        "one_hour_price_change": 0.08,
    }
    laggard = {
        "title": "Will Bitcoin hit 100k this year!!!",
        "market_key": "btc-laggard",
        "one_hour_price_change": 0.0,
    }

    assert detect_lagging_correlated_markets([laggard], [leader]) == []


def test_reference_event_only_match_does_not_emit_lag_signal():
    leader = {
        "title": "Will Bitcoin reach a new high before GTA VI?",
        "market_key": "btc-gta",
        "one_hour_price_change": 0.08,
    }
    laggard = {
        "title": "Will Jesus return before GTA VI?",
        "market_key": "jesus-gta",
        "reference_event": "GTA VI",
        "one_hour_price_change": 0.0,
    }

    signals = detect_lagging_correlated_markets([laggard], [leader])
    assert signals == []
    relationship = score_market_relationship(leader, laggard)
    assert relationship["relationship"] == "reference_event_only"
    assert relationship["expected_direction"] == "unknown"


def test_crypto_leader_emits_signal_for_unmoved_crypto_laggard():
    leader = {
        "title": "Will Bitcoin rally before GTA VI?",
        "market_key": "btc-gta",
        "one_hour_price_change": 0.08,
    }
    laggard = {
        "title": "Will MicroStrategy stock rise with Bitcoin?",
        "market_key": "mstr-btc",
        "one_hour_price_change": 0.0,
    }

    signals = detect_lagging_correlated_markets([laggard], [leader])
    assert len(signals) == 1
    assert signals[0]["relationship"] == "same_semantic_group"


def test_crypto_leader_does_not_emit_for_geopolitics_laggard_with_same_gta_reference():
    leader = {
        "title": "Will Bitcoin rally before GTA VI?",
        "market_key": "btc-gta",
        "reference_event": "GTA VI",
        "one_hour_price_change": 0.08,
    }
    laggard = {
        "title": "Will China invade Taiwan before GTA VI?",
        "market_key": "china-taiwan-gta",
        "reference_event": "GTA VI",
        "one_hour_price_change": 0.0,
    }

    assert detect_lagging_correlated_markets([laggard], [leader]) == []


def test_btc_up_microstrategy_sells_bitcoin_becomes_review_candidate():
    leader = {
        "title": "Will Bitcoin hit $1m before GTA VI?",
        "market_key": "btc-leader",
        "computed_mid_delta": 0.05,
    }
    laggard = {
        "title": "MicroStrategy sells any Bitcoin by June 30, 2026?",
        "market_key": "mstr-sells-btc",
        "computed_mid_delta": 0.0,
        "spread": 0.002,
        "depth_top5": 60_000,
    }

    result = detect_lagging_correlated_markets_with_review([laggard], [leader])
    assert result["lag_signals"] == []
    assert len(result["review_candidates"]) == 1
    assert "does not clearly imply" in result["review_candidates"][0]["reason"]


def test_reference_event_only_pair_becomes_review_candidate_not_lag_signal():
    leader = {
        "title": "Will Bitcoin rally before GTA VI?",
        "market_key": "btc-gta",
        "reference_event": "GTA VI",
        "computed_mid_delta": 0.05,
    }
    laggard = {
        "title": "Will China invade Taiwan before GTA VI?",
        "market_key": "china-taiwan-gta",
        "reference_event": "GTA VI",
        "computed_mid_delta": 0.0,
    }

    result = detect_lagging_correlated_markets_with_review([laggard], [leader])
    assert result["lag_signals"] == []
    assert len(result["review_candidates"]) == 1
    assert result["review_candidates"][0]["relationship"] == "reference_event_only"


def test_low_confidence_reference_markets_never_produce_lag_signals():
    leader = {
        "title": "GTA VI released before June 2026?",
        "market_key": "gta-release",
        "computed_mid_delta": 0.05,
        "resolution_meta": {
            "reference_event": "GTA VI",
            "rule_type": "release_timing",
            "deadline_confidence": 0.3,
            "resolution_deadline_utc": "2026-06-01T00:00:00Z",
        },
    }
    laggard = {
        "title": "Will Jesus Christ return before GTA VI?",
        "market_key": "jesus-gta",
        "computed_mid_delta": 0.0,
        "resolution_meta": {
            "reference_event": "GTA VI",
            "rule_type": "before_reference_event",
            "deadline_confidence": 0.3,
            "resolution_deadline_utc": None,
        },
    }

    result = detect_lagging_correlated_markets_with_review([laggard], [leader])
    assert result["lag_signals"] == []
    assert len(result["review_candidates"]) == 1
    assert result["review_candidates"][0]["reason"] == "insufficient resolution-rule confidence"


def test_deadline_aware_causal_direction_uses_opposite_logic():
    leader = {
        "title": "GTA VI released before June 30, 2026?",
        "market_key": "gta-release",
        "computed_mid_delta": 0.05,
        "resolution_meta": {
            "reference_event": "GTA VI",
            "rule_type": "release_timing",
            "deadline_confidence": 0.85,
            "resolution_deadline_utc": "2026-06-30T00:00:00Z",
        },
    }
    laggard = {
        "title": "Will Jesus Christ return before GTA VI?",
        "market_key": "jesus-gta",
        "computed_mid_delta": 0.0,
        "resolution_meta": {
            "reference_event": "GTA VI",
            "rule_type": "before_reference_event",
            "deadline_confidence": 0.85,
            "resolution_deadline_utc": None,
        },
    }

    result = detect_lagging_correlated_markets_with_review([laggard], [leader])
    assert result["review_candidates"] == []
    assert len(result["lag_signals"]) == 1
    signal = result["lag_signals"][0]
    assert signal["expected_direction"] == "opposite"
    assert signal["action"] == "buy_no_laggard"
    assert signal["relationship"] == "resolution_aware_reference_event"
    assert "deadline compression" in signal["relationship_reasons"][0]
    assert signal["deadline_confidence"] == 0.85
    assert signal["rule_type"] == "before_reference_event"


def test_no_emitted_signal_uses_shared_keyword_theme_relationship():
    leader = {
        "title": "Will Bitcoin ETF hit record high?",
        "market_key": "btc-leader",
        "computed_mid_delta": 0.05,
    }
    laggard = {
        "title": "Will Bitcoin ETF get approval?",
        "market_key": "btc-laggard",
        "computed_mid_delta": 0.0,
    }

    signals = detect_lagging_correlated_markets([laggard], [leader])
    assert signals
    assert all(s["relationship"] != "shared_keyword_theme" for s in signals)


# ---------------------------------------------------------------------------
# load_leader_markets
# ---------------------------------------------------------------------------

def test_load_leader_markets_none_returns_empty():
    assert load_leader_markets(None) == []


def test_load_leader_markets_from_json_file():
    leaders = [
        {
            "title": "Will Fed cut rates in Q3?",
            "market_key": "fed-cut-q3",
            "reference_event": "FOMC",
            "one_hour_price_change": 0.05,
            "one_day_price_change": 0.08,
            "mid": 0.65,
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(leaders, f)
        tmp_path = f.name

    loaded = load_leader_markets(tmp_path)
    assert loaded == leaders
    assert loaded[0]["market_key"] == "fed-cut-q3"
    Path(tmp_path).unlink()


def test_vardr_api_loader_accepts_list_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "title": "Will Bitcoin rally?",
                    "market_key": "btc-rally",
                    "one_hour_price_change": 0.06,
                }
            ]

    monkeypatch.setattr("market_fetcher.leader_markets.requests.get", lambda *args, **kwargs: Response())
    loaded = load_leader_markets_from_vardr_api("http://localhost:8000/leader-markets")
    assert loaded == [
        {
            "title": "Will Bitcoin rally?",
            "market_key": "btc-rally",
            "reference_event": None,
            "mid": None,
            "computed_mid_delta": None,
            "one_hour_price_change": 0.06,
            "one_day_price_change": None,
            "source": "vardr_api",
        }
    ]


def test_vardr_api_loader_accepts_dict_leader_markets_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "leader_markets": [
                    {
                        "question": "Will Trump win the election?",
                        "id": "trump-election",
                        "mid": 0.52,
                    }
                ]
            }

    monkeypatch.setattr("market_fetcher.leader_markets.requests.get", lambda *args, **kwargs: Response())
    loaded = load_leader_markets_from_vardr_api("http://localhost:8000/leader-markets")
    assert loaded[0]["title"] == "Will Trump win the election?"
    assert loaded[0]["market_key"] == "trump-election"
    assert loaded[0]["mid"] == 0.52


def test_vardr_api_loader_handles_failures_safely(monkeypatch):
    import requests

    def fail(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("market_fetcher.leader_markets.requests.get", fail)
    assert load_leader_markets_from_vardr_api("http://localhost:8000/leader-markets") == []


def test_vardr1_client_fetches_suspicious_endpoint_first(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        text = (
            '[{"platform":"polymarket","market_id":"0x0189df05",'
            '"market_title":"Will Japan win the 2026 FIFA World Cup?",'
            '"price":0.021,"anomaly_score":0.72,'
            '"raw_risk":0.66,"trade_size":145476.8280440001}]'
        )

        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "platform": "polymarket",
                    "market_id": "0x0189df05",
                    "market_title": "Will Japan win the 2026 FIFA World Cup?",
                    "price": 0.021,
                    "anomaly_score": 0.72,
                    "raw_risk": 0.66,
                    "trade_size": 145476.8280440001,
                }
            ]

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("market_fetcher.vardr1_client.requests.get", fake_get)
    leaders = fetch_vardr1_leader_markets(base_url="http://localhost:9002 ", window="24h", limit=25)

    assert calls[0][0] == "http://localhost:9002/api/suspicious"
    assert calls[0][1]["params"] == {"window": "24h", "limit": 25}
    assert len(calls) == 1
    assert leaders[0]["title"] == "Will Japan win the 2026 FIFA World Cup?"
    assert leaders[0]["market_key"] == "0x0189df05"
    assert leaders[0]["computed_mid_delta"] is None
    assert leaders[0]["one_hour_price_change"] == 0.0
    assert leaders[0]["one_day_price_change"] == 0.0
    assert leaders[0]["mid"] == 0.021
    assert leaders[0]["leader_score"] == 0.72
    assert leaders[0]["raw_risk"] == 0.66
    assert leaders[0]["recent_volume"] == 145476.8280440001
    assert leaders[0]["source"] == "vardr1_anomaly_fallback"
    assert leaders[0]["movement_source"] == "anomaly_score"
    assert leaders[0]["reason"] == 0.72
    metadata = vardr1_last_fetch_metadata()
    assert metadata["leader_rows_raw_loaded"] == 1
    assert metadata["leader_source_endpoint"] == "http://localhost:9002/api/suspicious?window=24h&limit=25"
    assert metadata["leader_fetch_endpoint_attempts"][0]["status_code"] == 200
    assert metadata["leader_fetch_endpoint_attempts"][0]["raw_count"] == 1


def test_vardr1_client_falls_back_to_suspicious_markets_when_suspicious_empty(monkeypatch):
    calls = []

    class Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload
            self.text = json.dumps(payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/api/suspicious"):
            return Response([])
        if url.endswith("/api/suspicious-markets"):
            return Response(
                {
                    "data": [
                        {
                            "platform": "polymarket",
                            "market_id": "fallback",
                            "market_title": "Fallback market aggregate",
                            "max_raw_risk": 0.7,
                            "max_risk_score": 0.3,
                        }
                    ]
                }
            )
        return Response([])

    monkeypatch.setattr("market_fetcher.vardr1_client.requests.get", fake_get)
    leaders = fetch_vardr1_leader_markets(base_url="http://localhost:9002", window="24h", limit=25)

    assert [call[0] for call in calls] == [
        "http://localhost:9002/api/suspicious",
        "http://localhost:9002/api/suspicious-markets",
    ]
    assert calls[1][1]["params"] == {"window": "24h", "limit": 100}
    assert leaders[0]["market_id"] == "fallback"
    assert leaders[0]["leader_score"] == 0.7
    assert leaders[0]["source"] == "vardr1_anomaly_fallback"
    metadata = vardr1_last_fetch_metadata()
    assert metadata["leader_source_endpoint"] == "http://localhost:9002/api/suspicious-markets?window=24h&limit=100"
    assert metadata["leader_rows_raw_loaded"] == 1
    assert [attempt["raw_count"] for attempt in metadata["leader_fetch_endpoint_attempts"]] == [0, 1]


def test_vardr1_client_metadata_explains_zero_loaded_leaders(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        text = "[]"

        def raise_for_status(self):
            return None

        def json(self):
            return []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("market_fetcher.vardr1_client.requests.get", fake_get)

    leaders = fetch_vardr1_leader_markets(base_url="http://localhost:9002", window="24h", limit=25)

    assert leaders == []
    assert [call[0] for call in calls] == [
        "http://localhost:9002/api/suspicious",
        "http://localhost:9002/api/suspicious-markets",
    ]
    metadata = vardr1_last_fetch_metadata()
    assert metadata["leader_rows_raw_loaded"] == 0
    assert "No Vardr-1 leaders loaded" in metadata["leader_fetch_error"]
    assert "http://localhost:9002/api/suspicious?window=24h&limit=25" in metadata["leader_fetch_error"]
    assert "http://localhost:9002/api/suspicious-markets?window=24h&limit=100" in metadata["leader_fetch_error"]
    assert metadata["leader_fetch_response_length"] == 2
    assert metadata["leader_fetch_response_preview"] == "[]"


def test_lag_universe_loader_prefers_vardr1_parquet(monkeypatch, tmp_path):
    import pandas as pd

    parquet_path = tmp_path / "polymarket_markets.parquet"
    pd.DataFrame(
        [
            {
                "conditionid": "0xactive",
                "question": "Will Bitcoin rally in 2026?",
                "slug": "bitcoin-rally",
                "active": True,
                "closed": False,
                "archived": False,
                "lasttradeprice": 0.42,
                "onehourpricechange": 0.03,
                "onedaypricechange": 0.07,
                "volume24hr": 1234.0,
                "liquiditynum": 5678.0,
            },
            {
                "conditionid": "0xclosed",
                "question": "Closed market",
                "slug": "closed-market",
                "active": True,
                "closed": True,
                "archived": False,
            },
        ]
    ).to_parquet(parquet_path)

    monkeypatch.setenv("VARDR1_MARKETS_PATH", str(parquet_path))

    markets, metadata = load_local_polymarket_universe_with_metadata(tmp_path)

    assert metadata["source"] == "vardr1_parquet"
    assert metadata["source_path"] == str(parquet_path)
    assert metadata["raw_row_count"] == 2
    assert metadata["row_count"] == 1
    assert len(markets) == 1
    assert markets[0]["market_id"] == "0xactive"
    assert markets[0]["market_key"] == "0xactive"
    assert markets[0]["market_title"] == "Will Bitcoin rally in 2026?"
    assert markets[0]["current_price"] == 0.42


def test_lag_universe_loader_falls_back_to_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "polymarket_markets.csv"
    csv_path.write_text(
        "market_id,market_title,current_price,bestbid,bestask,active,closed,price_change_1h,price_change_24h,volume24hr,liquiditynum\n"
        "m1,Will CSV market happen?,0.45,0.44,0.46,True,False,0.03,0.07,1234,5678\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VARDR1_MARKETS_PATH", str(tmp_path / "missing.parquet"))
    monkeypatch.setenv("VARDR_MARKETS_CSV_PATH", str(csv_path))

    markets, metadata = load_local_polymarket_universe_with_metadata(tmp_path)

    assert len(markets) == 1
    assert markets[0]["market_key"] == "m1"
    assert metadata["source"] == "csv"
    assert metadata["source_path"] == str(csv_path)
    assert markets[0]["one_hour_price_change"] == 0.03
    assert markets[0]["one_day_price_change"] == 0.07
    assert markets[0]["volume_proxy"] == 1234.0
    assert markets[0]["liquidity_proxy"] == 5678.0


def test_lag_candidates_world_cup_winner_vs_winner_is_same_option_set():
    leader = {
        "title": "Will Japan win the 2026 FIFA World Cup?",
        "market_key": "japan-world-cup",
        "one_hour_price_change": -0.9,
        "one_day_price_change": -0.9,
    }
    related = {
        "venue": "polymarket",
        "market_key": "switzerland-world-cup",
        "title": "Will Switzerland win the 2026 FIFA World Cup?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
    }

    candidates = build_lag_candidates(
        [leader], [related], min_similarity=0, min_divergence=0, include_review_only=True, max_abs_leader_move=1.0
    )
    assert len(candidates) == 1
    assert candidates[0]["relationship_type"] == "SAME_OPTION_SET_NEGATIVE_CORRELATION"
    assert candidates[0]["expected_correlation_direction"] == "negative"
    assert candidates[0]["option_set_key"] is not None

    debug = build_lag_candidates_debug([leader], [related], min_similarity=0, min_divergence=0, max_abs_leader_move=1.0)
    candidate = debug[0]["top_related_candidates_before_threshold_filtering"][0]
    assert candidate["relationship_type"] == "SAME_OPTION_SET_NEGATIVE_CORRELATION"
    assert candidate["rejected_reason"] is None


def test_crypto_fdv_different_assets_are_same_template_review_only():
    leader = {
        "title": "MegaETH market cap (FDV) > $2B one day after launch?",
        "market_key": "megaeth-2b",
        "one_hour_price_change": 0.12,
        "one_day_price_change": 0.24,
        "movement_source": "universe_match",
    }
    related = {
        "venue": "polymarket",
        "market_key": "abstract-1-5b",
        "title": "Abstract FDV above $1.5B one day after launch?",
        "one_hour_price_change": 0.01,
        "one_day_price_change": 0.02,
        "depth_top5": 100_000,
    }

    candidates, meta = build_lag_candidates_with_metadata(
        [leader],
        [related],
        exploratory=True,
        min_similarity=0,
        min_divergence=0,
        min_similarity_for_trade=0,
    )

    assert candidates
    candidate = candidates[0]
    assert candidate["relationship_type"] == "SAME_TEMPLATE_DIFFERENT_ASSET"
    assert candidate["relationship_type"] != "CAUSALLY_LINKED"
    assert candidate["trade_bucket"] == "review"
    assert candidate["tradable_signal"] is False
    assert candidate["suggested_trade_direction"] == "review_only"
    assert meta["same_template_different_asset_examples"]
    assert "same market template but different asset" in candidate["review_reason"]


def test_crypto_fdv_same_asset_different_threshold_is_threshold_bucket_not_causal():
    leader = {
        "title": "MegaETH market cap (FDV) > $2B one day after launch?",
        "market_key": "megaeth-2b",
        "one_hour_price_change": 0.12,
        "one_day_price_change": 0.24,
        "movement_source": "universe_match",
    }
    related = {
        "venue": "polymarket",
        "market_key": "megaeth-1-5b",
        "title": "MegaETH FDV above $1.5B one day after launch?",
        "one_hour_price_change": 0.01,
        "one_day_price_change": 0.02,
        "depth_top5": 100_000,
    }

    candidates, meta = build_lag_candidates_with_metadata(
        [leader],
        [related],
        exploratory=True,
        min_similarity=0,
        min_divergence=0,
        min_similarity_for_trade=0,
    )

    # Threshold-bucket pairs route to threshold_violation_signals, not candidates
    assert candidates == []
    assert meta["same_event_threshold_bucket_examples"]
    # No price data in this test → no price_monotonicity_violation; same-direction movement → no movement_divergence
    assert meta["threshold_violation_signals_count"] == 0


def test_spurs_nuggets_wcf_same_option_set_can_be_primary():
    leader = {
        "title": "Will the Spurs win the NBA Western Conference Finals?",
        "market_key": "spurs-wcf",
        "one_hour_price_change": 0.14,
        "one_day_price_change": 0.22,
        "movement_source": "universe_match",
    }
    related = {
        "venue": "polymarket",
        "market_key": "nuggets-wcf",
        "title": "Will the Nuggets win the NBA Western Conference Finals?",
        "one_hour_price_change": 0.01,
        "one_day_price_change": 0.03,
        "depth_top5": 100_000,
        "spread": 0.01,
    }

    candidates, meta = build_lag_candidates_with_metadata(
        [leader],
        [related],
        min_similarity=0,
        min_divergence=0,
        min_similarity_for_trade=0,
        min_related_abs_move_24h=0.01,
    )

    assert candidates
    candidate = candidates[0]
    assert candidate["relationship_type"] == "SAME_OPTION_SET_NEGATIVE_CORRELATION"
    assert candidate["expected_correlation_direction"] == "negative"
    assert candidate["tradable_signal"] is True
    assert candidate["trade_bucket"] == "primary"
    assert candidate["suggested_trade_direction"] == "buy_no_related"
    assert meta["same_option_set_examples"]


def test_lag_candidates_world_cup_option_set_valid_and_group_winner_rejected():
    leader = {
        "title": "Will Japan win the 2026 FIFA World Cup?",
        "market_key": "japan-world-cup",
        "one_hour_price_change": -0.9,
        "one_day_price_change": -0.9,
    }
    universe = [
        {
            "venue": "polymarket",
            "market_key": "brazil-world-cup",
            "title": "Will Brazil win the 2026 FIFA World Cup?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        },
        {
            "venue": "polymarket",
            "market_key": "france-world-cup",
            "title": "Will France win the 2026 FIFA World Cup?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        },
        {
            "venue": "polymarket",
            "market_key": "england-world-cup",
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

    candidates = build_lag_candidates(
        [leader],
        universe,
        min_similarity=0,
        min_divergence=0,
        include_review_only=True,
        max_abs_leader_move=1.0,
        max_candidates_per_leader_output=10,
        max_candidates_per_event_output=10,
        min_related_abs_move_24h=0,
        min_similarity_for_trade=0,
    )
    candidate_ids = {candidate["related_market_id"] for candidate in candidates}
    assert {"brazil-world-cup", "france-world-cup", "england-world-cup"}.issubset(candidate_ids)
    assert "japan-group-f" not in candidate_ids
    for candidate in candidates:
        if candidate["related_market_id"] in {"brazil-world-cup", "france-world-cup", "england-world-cup"}:
            assert candidate["relationship_type"] == "SAME_OPTION_SET_NEGATIVE_CORRELATION"
            assert candidate["expected_correlation_direction"] == "negative"

    debug = build_lag_candidates_debug([leader], universe, min_similarity=0, min_divergence=0, max_abs_leader_move=1.0)
    by_title = {
        candidate["related_market_title"]: candidate
        for candidate in debug[0]["top_related_candidates_before_threshold_filtering"]
    }
    group_f = by_title["Will Japan win Group F at the 2026 FIFA World Cup?"]
    assert group_f["relationship_type"] == "SAME_EVENT_OUTCOME"
    assert "SAME_EVENT_OUTCOME" in group_f["rejected_reason"]


def test_lag_candidates_skips_resolved_or_bad_tick_leaders():
    leaders = [
        {
            "title": "Will Glenn Youngkin win the 2028 US President?",
            "market_key": "youngkin-bad-tick",
            "one_hour_price_change": 0.981,
            "one_day_price_change": 0.981,
        },
        {
            "title": "Will Japan win the 2026 FIFA World Cup?",
            "market_key": "japan-bad-tick",
            "one_hour_price_change": -0.958,
            "one_day_price_change": -0.958,
        },
    ]
    universe = [
        {
            "venue": "polymarket",
            "market_key": "republicans-president",
            "title": "Will Republicans win the 2028 US President?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        },
        {
            "venue": "polymarket",
            "market_key": "asia-world-cup",
            "title": "Will Asia win the 2026 FIFA World Cup?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        },
    ]

    candidates, meta = build_lag_candidates_with_metadata(
        leaders,
        universe,
        min_similarity=0,
        min_divergence=0,
        max_leaders_evaluated=10,
    )

    assert candidates == []
    assert meta["resolved_or_bad_tick_leaders_skipped"] == 2
    assert meta["skip_reasons_count"]["resolved_or_bad_tick_candidate"] == 2
    assert len(meta["resolved_or_bad_tick_skip_examples"]) == 2

    debug = build_lag_candidates_debug(leaders, universe, leader_limit=2, min_similarity=0, min_divergence=0)
    assert [row["resolved_or_bad_tick_candidate"] for row in debug] == [True, True]
    assert [row["skip_reason"] for row in debug] == [
        "resolved_or_bad_tick_candidate",
        "resolved_or_bad_tick_candidate",
    ]


def test_lag_candidates_same_option_set_subset_positive_party_and_region():
    cases = [
        (
            "Will Glenn Youngkin win the 2028 US President?",
            "Will Republicans win the 2028 US President?",
            "youngkin",
            "republicans",
        ),
        (
            "Will Marco Rubio win the 2028 US President?",
            "Will Republicans win the 2028 US President?",
            "rubio",
            "republicans",
        ),
        (
            "Will AOC win the 2028 US President?",
            "Will Democrats win the 2028 US President?",
            "aoc",
            "democrats",
        ),
        (
            "Will Gretchen Whitmer win the 2028 US President?",
            "Will Democrats win the 2028 US President?",
            "whitmer",
            "democrats",
        ),
        (
            "Will Japan win the 2026 FIFA World Cup?",
            "Will Asia win the 2026 FIFA World Cup?",
            "japan",
            "asia",
        ),
    ]

    for leader_title, related_title, leader_key, related_key in cases:
        leader = {
            "title": leader_title,
            "market_key": leader_key,
            "one_hour_price_change": 0.3,
            "one_day_price_change": 0.4,
        }
        related = {
            "venue": "polymarket",
            "market_key": related_key,
            "title": related_title,
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        }

        candidates = build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0, include_review_only=True)

        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["relationship_type"] == "SAME_OPTION_SET_POSITIVE_CORRELATION"
        assert candidate["expected_correlation_direction"] == "positive"
        assert candidate["subset_relationship"] is True
        assert candidate["subset_relationship_reason"]
        assert candidate["resolved_or_bad_tick_candidate"] is False
        assert candidate["suggested_trade_direction"] != "buy_no_related"


def test_lag_candidates_same_option_set_competitors_remain_negative():
    cases = [
        (
            "Will Japan win the 2026 FIFA World Cup?",
            "Will Brazil win the 2026 FIFA World Cup?",
            "japan",
            "brazil",
        ),
        (
            "Will Japan win the 2026 FIFA World Cup?",
            "Will France win the 2026 FIFA World Cup?",
            "japan",
            "france",
        ),
        (
            "Will Glenn Youngkin win the 2028 US President?",
            "Will Democrats win the 2028 US President?",
            "youngkin",
            "democrats",
        ),
        (
            "Will Marco Rubio win the 2028 US President?",
            "Will Democrats win the 2028 US President?",
            "rubio",
            "democrats",
        ),
        (
            "Will AOC win the 2028 US President?",
            "Will Republicans win the 2028 US President?",
            "aoc",
            "republicans",
        ),
    ]

    for leader_title, related_title, leader_key, related_key in cases:
        leader = {
            "title": leader_title,
            "market_key": leader_key,
            "one_hour_price_change": 0.3,
            "one_day_price_change": 0.4,
        }
        related = {
            "venue": "polymarket",
            "market_key": related_key,
            "title": related_title,
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        }

        candidates = build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0, include_review_only=True)

        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["relationship_type"] == "SAME_OPTION_SET_NEGATIVE_CORRELATION"
        assert candidate["expected_correlation_direction"] == "negative"
        assert candidate["subset_relationship"] is False
        assert candidate["suggested_trade_direction"] == "review_only"
        assert candidate["low_related_activity"] is True
        assert candidate["tradable_signal"] is False


def test_lag_candidates_subset_down_move_is_not_treated_as_competitor_negative():
    leader = {
        "title": "Will Japan win the 2026 FIFA World Cup?",
        "market_key": "japan",
        "one_hour_price_change": -0.3,
        "one_day_price_change": -0.4,
    }
    related = {
        "venue": "polymarket",
        "market_key": "asia",
        "title": "Will Asia win the 2026 FIFA World Cup?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
    }

    candidates = build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0, include_review_only=True)

    assert len(candidates) == 1
    assert candidates[0]["relationship_type"] == "SAME_OPTION_SET_POSITIVE_CORRELATION"
    assert candidates[0]["expected_correlation_direction"] == "positive"
    assert candidates[0]["suggested_trade_direction"] != "buy_yes_related"


def test_lag_candidates_same_party_positive_trade_direction_is_not_buy_no():
    leader = {
        "title": "Will Marco Rubio win the 2028 US President?",
        "market_key": "rubio",
        "one_hour_price_change": 0.3,
        "one_day_price_change": 0.4,
    }
    related = {
        "venue": "polymarket",
        "market_key": "republicans",
        "title": "Will Republicans win the 2028 US President?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.02,
    }

    candidates = build_lag_candidates(
        [leader],
        [related],
        min_similarity=0,
        min_divergence=0,
        min_similarity_for_trade=0,
    )

    assert len(candidates) == 1
    assert candidates[0]["relationship_type"] == "SAME_OPTION_SET_POSITIVE_CORRELATION"
    assert candidates[0]["suggested_trade_direction"] == "buy_yes_related"
    assert candidates[0]["suggested_trade_direction"] != "buy_no_related"


def test_lag_candidates_execution_filters_allow_only_tradable_buy_actions():
    leader = {
        "title": "Will a Gaza ceasefire deal happen by June 2026?",
        "market_key": "gaza-ceasefire",
        "one_hour_price_change": 0.4,
        "one_day_price_change": 0.5,
        "_lag_tokens": {"gaza", "ceasefire", "deal", "june"},
    }
    related = {
        "venue": "polymarket",
        "market_key": "gaza-hostages",
        "title": "Will Gaza hostages be released by June 2026?",
        "one_hour_price_change": 0.01,
        "one_day_price_change": 0.02,
        "_lag_tokens": {"gaza", "ceasefire", "deal", "june"},
    }

    candidates = build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0, include_review_only=True)

    assert len(candidates) == 1
    assert candidates[0]["suggested_trade_direction"] == "buy_yes_related"
    assert candidates[0]["low_related_activity"] is False
    assert candidates[0]["below_similarity_threshold"] is False
    assert candidates[0]["tradable_signal"] is True


def test_lag_candidates_low_related_activity_downgrades_trade_action():
    leader = {
        "title": "Will a Gaza ceasefire deal happen by June 2026?",
        "market_key": "gaza-ceasefire",
        "one_hour_price_change": 0.4,
        "one_day_price_change": 0.5,
        "_lag_tokens": {"gaza", "ceasefire", "deal", "june"},
    }
    related = {
        "venue": "polymarket",
        "market_key": "gaza-hostages",
        "title": "Will Gaza hostages be released by June 2026?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
        "_lag_tokens": {"gaza", "ceasefire", "deal", "june"},
    }

    candidates = build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0, include_review_only=True)

    assert len(candidates) == 1
    assert candidates[0]["suggested_trade_direction"] == "review_only"
    assert candidates[0]["low_related_activity"] is True
    assert candidates[0]["below_similarity_threshold"] is False
    assert candidates[0]["tradable_signal"] is False


def test_lag_candidates_zero_24h_leader_and_related_removed():
    leader = {
        "title": "Will a Gaza ceasefire deal happen by June 2026?",
        "market_key": "gaza-ceasefire",
        "one_hour_price_change": 0.4,
        "one_day_price_change": 0.0,
    }
    related = {
        "venue": "polymarket",
        "market_key": "gaza-hostages",
        "title": "Will Gaza hostages be released by June 2026?",
        "one_hour_price_change": 0.1,
        "one_day_price_change": 0.0,
    }

    assert build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0) == []


def test_lag_candidates_exclude_duplicate_buckets():
    leader = {
        "title": "Will SBF be sentenced to 10-20 years?",
        "market_key": "sbf-10-20",
        "one_hour_price_change": 0.3,
        "one_day_price_change": 0.3,
    }
    related = {
        "venue": "polymarket",
        "market_key": "sbf-20-30",
        "title": "Will SBF be sentenced to 20-30 years?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
    }

    assert build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0) == []
    debug = build_lag_candidates_debug([leader], [related], min_similarity=0, min_divergence=0)
    candidate = debug[0]["top_related_candidates_before_threshold_filtering"][0]
    assert candidate["relationship_type"] == "DUPLICATE_BUCKET"
    assert "DUPLICATE_BUCKET" in candidate["rejected_reason"]


def test_lag_candidates_include_causally_linked_positive_lag():
    leader = {
        "title": "Will a Gaza ceasefire deal happen by June 2026?",
        "market_key": "gaza-ceasefire",
        "one_hour_price_change": 0.4,
        "one_day_price_change": 0.5,
    }
    related = {
        "venue": "polymarket",
        "market_key": "gaza-hostages-released",
        "title": "Will Gaza hostages be released by June 2026?",
        "one_hour_price_change": 0.01,
        "one_day_price_change": 0.02,
    }

    candidates = build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0, include_review_only=True)

    assert len(candidates) == 1
    assert candidates[0]["relationship_type"] == "CAUSALLY_LINKED"
    assert candidates[0]["expected_correlation_direction"] == "positive"
    assert candidates[0]["suggested_trade_direction"] == "review_only"
    assert candidates[0]["below_similarity_threshold"] is True
    assert candidates[0]["tradable_signal"] is False


def test_lag_candidates_correlation_direction_negative():
    leader = {
        "title": "Will a Gaza ceasefire deal happen by June 2026?",
        "market_key": "gaza-ceasefire",
        "one_hour_price_change": 0.4,
        "one_day_price_change": 0.5,
    }
    related = {
        "venue": "polymarket",
        "market_key": "gaza-sanctions",
        "title": "Will Gaza sanctions happen by June 2026?",
        "one_hour_price_change": 0.01,
        "one_day_price_change": 0.02,
    }

    candidates = build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0, include_review_only=True)

    assert len(candidates) == 1
    assert candidates[0]["relationship_type"] == "CAUSALLY_LINKED"
    assert candidates[0]["expected_correlation_direction"] == "negative"
    assert candidates[0]["suggested_trade_direction"] == "review_only"
    assert candidates[0]["below_similarity_threshold"] is True
    assert candidates[0]["tradable_signal"] is False


def test_lag_candidates_exploratory_allows_weak_correlated_candidates():
    leader = {
        "title": "Will Bitcoin hit 100k by end of 2026?",
        "market_key": "btc-100k",
        "one_hour_price_change": 0.2,
        "one_day_price_change": 0.3,
    }
    related = {
        "venue": "polymarket",
        "market_key": "btc-etf",
        "title": "Will Bitcoin ETF get approval by end of 2026?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
    }

    assert build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0) == []
    exploratory = build_lag_candidates(
        [leader],
        [related],
        min_similarity=0,
        min_divergence=0,
        exploratory=True,
    )

    assert len(exploratory) == 1
    assert exploratory[0]["relationship_type"] == "CORRELATED_BUT_WEAK"
    assert exploratory[0]["strong_match"] is False
    assert exploratory[0]["exploratory_match"] is True


def test_lag_candidates_reject_world_cup_winner_vs_halftime_performer_as_shared_event_only():
    leader = {
        "title": "Will Japan win the 2026 FIFA World Cup?",
        "market_key": "japan-world-cup",
        "one_hour_price_change": -0.9,
        "one_day_price_change": -0.9,
    }
    related = {
        "venue": "polymarket",
        "market_key": "world-cup-halftime",
        "title": "Will Taylor Swift perform at the 2026 World Cup halftime show?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
    }

    assert build_lag_candidates(
        [leader], [related], min_similarity=0, min_divergence=0, exploratory=True, max_abs_leader_move=1.0
    ) == []
    debug = build_lag_candidates_debug(
        [leader], [related], min_similarity=0, min_divergence=0, exploratory=True, max_abs_leader_move=1.0
    )
    candidate = debug[0]["top_related_candidates_before_threshold_filtering"][0]
    assert candidate["relationship_type"] == "SHARED_EVENT_ONLY"
    assert "SHARED_EVENT_ONLY" in candidate["rejected_reason"]


def test_lag_candidates_cross_role_same_entity_accepted_as_review_only():
    leader = {
        "title": "Will Ted Cruz win the 2028 Republican presidential nomination?",
        "market_key": "ted-president",
        "one_hour_price_change": -0.9,
        "one_day_price_change": -0.9,
    }
    related = {
        "venue": "polymarket",
        "market_key": "ted-vp",
        "title": "Will Ted Cruz be the 2028 Republican Vice-Presidential nominee?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
    }

    candidates = build_lag_candidates(
        [leader], [related], min_similarity=0, min_divergence=0, include_review_only=True, max_abs_leader_move=1.0
    )
    assert len(candidates) == 1
    assert candidates[0]["relationship_type"] == "CROSS_ROLE_SAME_ENTITY"
    assert candidates[0]["suggested_trade_direction"] == "review_only"

    debug = build_lag_candidates_debug([leader], [related], min_similarity=0, min_divergence=0, max_abs_leader_move=1.0)
    candidate = debug[0]["top_related_candidates_before_threshold_filtering"][0]
    assert candidate["relationship_type"] == "CROSS_ROLE_SAME_ENTITY"
    assert candidate["rejected_reason"] is None


def test_lag_signals_from_external_leader_markets():
    leader = {
        "title": "Will Trump win the election?",
        "market_key": "trump-election",
        "one_hour_price_change": 0.07,
        "one_day_price_change": 0.09,
        "mid": 0.65,
    }
    laggard = {
        "title": "Will Republicans control Congress after the election?",
        "market_key": "congress-election",
        "one_hour_price_change": 0.001,
        "one_day_price_change": 0.002,
        "mid": 0.30,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump([leader], f)
        tmp_path = f.name

    loaded_leaders = load_leader_markets(tmp_path)
    signals = detect_lagging_correlated_markets(markets=[laggard], leader_markets=loaded_leaders)
    Path(tmp_path).unlink()

    assert len(signals) == 1
    assert signals[0]["leader_market_key"] == "trump-election"
    assert signals[0]["laggard_market_key"] == "congress-election"


# ---------------------------------------------------------------------------
# detect_lagging_correlated_markets — new signal fields
# ---------------------------------------------------------------------------

def _lag_pair():
    """Return a (leader, laggard) pair that always produces a signal."""
    leader = {
        "title": "Will Bitcoin ETF hit record high?",
        "market_key": "btc-leader",
        "mid": 0.70,
        "computed_mid_delta": 0.05,
        "one_hour_price_change": 0.03,
    }
    laggard = {
        "title": "Will Bitcoin ETF get approval?",
        "market_key": "btc-laggard",
        "mid": 0.50,
        "computed_mid_delta": 0.001,
        "spread": 0.02,
        "depth_top5": 20_000,
    }
    return leader, laggard


def test_lag_uses_computed_mid_delta_before_1h_change():
    leader, laggard = _lag_pair()
    signals = detect_lagging_correlated_markets([laggard], [leader])
    assert len(signals) == 1
    # leader_move should be the delta (0.05), not 1h change (0.03)
    assert signals[0]["leader_move"] == 0.05


def test_action_is_buy_yes_when_leader_move_positive():
    leader, laggard = _lag_pair()
    signals = detect_lagging_correlated_markets([laggard], [leader])
    assert len(signals) == 1
    assert signals[0]["action"] == "buy_yes_laggard"


def test_action_is_buy_no_when_leader_move_negative():
    leader, laggard = _lag_pair()
    leader["computed_mid_delta"] = -0.05
    signals = detect_lagging_correlated_markets([laggard], [leader])
    assert len(signals) == 1
    assert signals[0]["action"] == "buy_no_laggard"


def test_expected_direction_controls_opposite_action(monkeypatch):
    leader, laggard = _lag_pair()

    def fake_score(_leader, _laggard):
        return {
            "score": 0.8,
            "relationship": "same_semantic_group",
            "reasons": ["forced opposite direction for test"],
            "expected_direction": "opposite",
        }

    monkeypatch.setattr(intelligence, "score_market_relationship", fake_score)
    signals = intelligence.detect_lagging_correlated_markets([laggard], [leader])
    assert len(signals) == 1
    assert signals[0]["expected_direction"] == "opposite"
    assert signals[0]["action"] == "buy_no_laggard"


def test_lag_candidates_halftime_artist_vs_artist_is_same_option_set():
    leader = {
        "title": "Will Beyoncé perform at the 2026 World Cup halftime show?",
        "market_key": "beyonce-halftime",
        "one_hour_price_change": 0.3,
        "one_day_price_change": 0.4,
    }
    related = {
        "venue": "polymarket",
        "market_key": "taylor-halftime",
        "title": "Will Taylor Swift perform at the 2026 World Cup halftime show?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
    }

    candidates = build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0, include_review_only=True)
    assert len(candidates) == 1
    assert candidates[0]["relationship_type"] == "SAME_OPTION_SET_NEGATIVE_CORRELATION"
    assert candidates[0]["expected_correlation_direction"] == "negative"
    assert candidates[0]["suggested_trade_direction"] == "review_only"
    assert candidates[0]["low_related_activity"] is True
    assert candidates[0]["tradable_signal"] is False


def test_lag_candidates_japan_winner_vs_calvin_harris_halftime_rejected_as_shared_event_only():
    leader = {
        "title": "Will Japan win the 2026 FIFA World Cup?",
        "market_key": "japan-world-cup",
        "one_hour_price_change": 0.3,
        "one_day_price_change": 0.4,
    }
    related = {
        "venue": "polymarket",
        "market_key": "calvin-halftime",
        "title": "Will Calvin Harris perform at the 2026 World Cup halftime show?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
    }

    assert build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0) == []
    debug = build_lag_candidates_debug([leader], [related], min_similarity=0, min_divergence=0)
    candidate = debug[0]["top_related_candidates_before_threshold_filtering"][0]
    assert candidate["relationship_type"] == "SHARED_EVENT_ONLY"
    assert "SHARED_EVENT_ONLY" in candidate["rejected_reason"]


def test_lag_candidates_ted_cruz_presidential_vs_vp_is_cross_role_same_entity():
    leader = {
        "title": "Will Ted Cruz win the 2028 Republican presidential nomination?",
        "market_key": "ted-president",
        "one_hour_price_change": 0.3,
        "one_day_price_change": 0.4,
    }
    related = {
        "venue": "polymarket",
        "market_key": "ted-vp",
        "title": "Will Ted Cruz be the 2028 Republican Vice-Presidential nominee?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
    }

    candidates = build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0, include_review_only=True)
    assert len(candidates) == 1
    c = candidates[0]
    assert c["relationship_type"] == "CROSS_ROLE_SAME_ENTITY"
    assert c["suggested_trade_direction"] == "review_only"
    assert c["expected_correlation_direction"] == "unclear"


def test_lag_candidates_world_cup_team_vs_team_is_same_option_set_not_causally_linked():
    leader = {
        "title": "Will Brazil win the 2026 FIFA World Cup?",
        "market_key": "brazil-world-cup",
        "one_hour_price_change": 0.3,
        "one_day_price_change": 0.4,
    }
    related = {
        "venue": "polymarket",
        "market_key": "france-world-cup",
        "title": "Will France win the 2026 FIFA World Cup?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
    }

    candidates = build_lag_candidates([leader], [related], min_similarity=0, min_divergence=0, include_review_only=True)
    assert len(candidates) == 1
    assert candidates[0]["relationship_type"] == "SAME_OPTION_SET_NEGATIVE_CORRELATION"
    assert candidates[0]["relationship_type"] != "CAUSALLY_LINKED"
    assert candidates[0]["expected_correlation_direction"] == "negative"
    assert candidates[0]["suggested_trade_direction"] == "review_only"
    assert candidates[0]["low_related_activity"] is True
    assert candidates[0]["tradable_signal"] is False


def test_signal_strength_is_numeric():
    leader, laggard = _lag_pair()
    signals = detect_lagging_correlated_markets([laggard], [leader])
    assert len(signals) == 1
    strength = signals[0]["signal_strength"]
    assert isinstance(strength, (int, float))
    assert strength > 0


def test_build_lag_candidates_with_metadata_returns_metadata():
    leader = {
        "title": "Will a Gaza ceasefire deal happen by June 2026?",
        "market_key": "gaza-ceasefire",
        "one_hour_price_change": 0.4,
        "one_day_price_change": 0.5,
    }
    related = {
        "venue": "polymarket",
        "market_key": "gaza-hostages",
        "title": "Will Gaza hostages be released by June 2026?",
        "one_hour_price_change": 0.01,
        "one_day_price_change": 0.02,
    }

    candidates, meta = build_lag_candidates_with_metadata(
        [leader], [related], min_similarity=0, min_divergence=0, max_leaders_evaluated=10, include_review_only=True
    )

    assert len(candidates) == 1
    assert meta["leaders_fetched"] == 1
    assert meta["leaders_evaluated"] == 1
    assert meta["remaining_leaders_not_evaluated"] == 0
    assert meta["leaders_skipped"] == 0
    assert meta["valid_candidates_found"] == 1
    assert meta["partial_results_due_to_timeout"] is False


def test_lag_candidates_evaluates_minimum_leaders_before_timeout(monkeypatch):
    leaders = [
        {
            "title": f"Leader {idx}",
            "market_key": f"leader-{idx}",
            "one_hour_price_change": 0.1,
            "one_day_price_change": 0.1,
        }
        for idx in range(10)
    ]

    ticks = iter([idx * 0.2 for idx in range(20)])
    monkeypatch.setattr(lag_candidates_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(lag_candidates_module, "_leader_candidate_evaluations", lambda *args, **kwargs: [])

    candidates, meta = build_lag_candidates_with_metadata(
        leaders,
        [],
        max_leaders_evaluated=10,
        timeout_seconds=0.1,
        min_leaders_before_timeout=5,
    )

    assert candidates == []
    assert meta["partial_results_due_to_timeout"] is True
    assert meta["leaders_evaluated"] == 5
    assert meta["remaining_leaders_not_evaluated"] == 5
    assert meta["leaders_evaluated_before_timeout"] == 5
    assert meta["elapsed_time_total"] > 0


def test_lag_candidates_evaluates_minimum_leaders_before_limit_short_circuit(monkeypatch):
    leaders = [
        {
            "title": f"Leader {idx}",
            "market_key": f"leader-{idx}",
            "one_hour_price_change": 0.1,
            "one_day_price_change": 0.1,
        }
        for idx in range(10)
    ]

    def fake_evaluations(leader, *_args, **_kwargs):
        return [
            {
                "leader_market_title": leader["title"],
                "leader_market_id": leader["market_key"],
                "related_market_title": f"Related {leader['market_key']}",
                "related_market_id": f"related-{leader['market_key']}",
                "relationship_type": "SAME_OPTION_SET_NEGATIVE_CORRELATION",
                "expected_correlation_direction": "negative",
                "divergence_score": 1.0,
                "rejected_reason": None,
                "excluded_as_self_market": False,
                "excluded_as_duplicate_bucket": False,
            }
        ]

    monkeypatch.setattr(lag_candidates_module, "_leader_candidate_evaluations", fake_evaluations)

    candidates, meta = build_lag_candidates_with_metadata(
        leaders,
        [],
        limit=1,
        max_leaders_evaluated=10,
        min_leaders_before_timeout=5,
        timeout_seconds=60,
        include_review_only=True,
    )

    assert len(candidates) == 1
    assert meta["leaders_evaluated"] == 5
    assert meta["partial_results_due_to_timeout"] is False


def test_lag_candidates_output_caps_each_leader_and_continues(monkeypatch):
    leaders = [
        {
            "title": f"Leader {idx}",
            "market_key": f"leader-{idx}",
            "one_hour_price_change": 0.1,
            "one_day_price_change": 0.1,
        }
        for idx in range(3)
    ]

    def fake_evaluations(leader, *_args, **_kwargs):
        return [
            {
                "leader_market_title": leader["title"],
                "leader_market_id": leader["market_key"],
                "related_market_title": f"Related {leader['market_key']}-{idx}",
                "related_market_id": f"related-{leader['market_key']}-{idx}",
                "relationship_type": "CAUSALLY_LINKED",
                "expected_correlation_direction": "positive",
                "divergence_score": 1.0 - (idx / 10),
                "tradable_signal": True,
                "rejected_reason": None,
                "excluded_as_self_market": False,
                "excluded_as_duplicate_bucket": False,
            }
            for idx in range(4)
        ]

    monkeypatch.setattr(lag_candidates_module, "_leader_candidate_evaluations", fake_evaluations)

    candidates, meta = build_lag_candidates_with_metadata(
        leaders,
        [],
        limit=5,
        max_leaders_evaluated=3,
        max_candidates_per_leader_output=2,
        min_leaders_before_timeout=0,
    )

    counts_by_leader = {}
    for candidate in candidates:
        counts_by_leader[candidate["leader_market_id"]] = counts_by_leader.get(candidate["leader_market_id"], 0) + 1

    assert len(candidates) == 5
    assert max(counts_by_leader.values()) <= 2
    assert len(counts_by_leader) >= 3
    assert meta["candidates_suppressed_by_leader_cap"] >= 3
    assert meta["max_candidates_per_leader_output"] == 2


def test_lag_candidates_output_caps_each_event_and_continues(monkeypatch):
    leaders = [
        {
            "title": f"Election Leader {idx}",
            "market_key": f"leader-{idx}",
            "one_hour_price_change": 0.1,
            "one_day_price_change": 0.1,
        }
        for idx in range(4)
    ]

    def fake_evaluations(leader, *_args, **_kwargs):
        return [
            {
                "leader_market_title": leader["title"],
                "leader_market_id": leader["market_key"],
                "related_market_title": f"Related {leader['market_key']}",
                "related_market_id": f"related-{leader['market_key']}",
                "relationship_type": "SAME_OPTION_SET_NEGATIVE_CORRELATION",
                "expected_correlation_direction": "negative",
                "divergence_score": 1.0,
                "tradable_signal": True,
                "option_set_key": "('us presidential election', '2028', 'winner')",
                "rejected_reason": None,
                "excluded_as_self_market": False,
                "excluded_as_duplicate_bucket": False,
            }
        ]

    monkeypatch.setattr(lag_candidates_module, "_leader_candidate_evaluations", fake_evaluations)

    candidates, meta = build_lag_candidates_with_metadata(
        leaders,
        [],
        limit=10,
        max_leaders_evaluated=4,
        max_candidates_per_leader_output=5,
        max_candidates_per_event_output=2,
        min_leaders_before_timeout=0,
    )

    assert len(candidates) == 2
    assert meta["candidates_suppressed_by_event_cap"] == 2
    assert meta["events_represented_count"] == 1
    assert meta["unique_leaders_in_output"] == 2
    assert meta["tradable_signals_count"] == 2
    assert meta["max_candidates_per_event_output"] == 2


def test_lag_candidates_leader_cap_allows_extra_slots_only_for_tradable(monkeypatch):
    leader = {
        "title": "Leader",
        "market_key": "leader",
        "one_hour_price_change": 0.1,
        "one_day_price_change": 0.1,
    }

    def fake_evaluations(_leader, *_args, **_kwargs):
        rows = []
        for idx, tradable in enumerate([False, False, True, True, False]):
            rows.append(
                {
                    "leader_market_title": "Leader",
                    "leader_market_id": "leader",
                    "related_market_title": f"Related {idx}",
                    "related_market_id": f"related-{idx}",
                    "relationship_type": "CAUSALLY_LINKED",
                    "expected_correlation_direction": "positive",
                    "divergence_score": 1.0 - (idx / 10),
                    "tradable_signal": tradable,
                    "rejected_reason": None,
                    "excluded_as_self_market": False,
                    "excluded_as_duplicate_bucket": False,
                }
            )
        return rows

    monkeypatch.setattr(lag_candidates_module, "_leader_candidate_evaluations", fake_evaluations)

    candidates, meta = build_lag_candidates_with_metadata(
        [leader],
        [],
        limit=10,
        max_leaders_evaluated=1,
        include_review_only=True,
        min_leaders_before_timeout=0,
    )

    assert len(candidates) == 4
    assert [candidate["related_market_id"] for candidate in candidates] == [
        "related-0",
        "related-1",
        "related-2",
        "related-3",
    ]
    assert sum(1 for candidate in candidates if not candidate["tradable_signal"]) == 2
    assert meta["max_candidates_per_leader_output"] == 4
    assert meta["candidates_suppressed_by_leader_cap"] == 1


def test_lag_candidates_prefilters_low_quality_leaders_before_evaluation(monkeypatch):
    leaders = [
        {
            "title": "Inconsistent spike leader",
            "market_key": "inconsistent",
            "one_hour_price_change": 0.6,
            "one_day_price_change": 0.01,
        },
        {
            "title": "Low information leader",
            "market_key": "low-info",
            "one_hour_price_change": 0.01,
            "one_day_price_change": 0.005,
        },
        {
            "title": "Near boundary leader",
            "market_key": "boundary",
            "one_hour_price_change": 0.1,
            "one_day_price_change": 0.1,
            "current_price": 0.99,
        },
    ]

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("prefiltered leaders should not be evaluated")

    monkeypatch.setattr(lag_candidates_module, "_leader_candidate_evaluations", fail_if_called)

    candidates, meta = build_lag_candidates_with_metadata(
        leaders,
        [],
        limit=10,
        max_leaders_evaluated=3,
        min_leaders_before_timeout=0,
    )

    assert candidates == []
    assert meta["leaders_filtered_before_evaluation"] == 3
    assert meta["leader_prefilter_skip_reasons_count"] == {
        "inconsistent_spike_candidate": 1,
        "low_information_leader": 1,
        "near_boundary_probability": 1,
    }
    assert meta["leaders_skipped"] == 3


def test_lag_candidates_metadata_counts_output_quality(monkeypatch):
    leaders = [
        {
            "title": "World Cup leader",
            "market_key": "wc-leader",
            "one_hour_price_change": 0.1,
            "one_day_price_change": 0.1,
        },
        {
            "title": "Election leader",
            "market_key": "election-leader",
            "one_hour_price_change": 0.1,
            "one_day_price_change": 0.1,
        },
    ]

    def fake_evaluations(leader, *_args, **_kwargs):
        if leader["market_key"] == "wc-leader":
            option_set_key = "('fifa world cup', '2026', 'winner')"
        else:
            option_set_key = "('us presidential election', '2028', 'winner')"
        return [
            {
                "leader_market_title": leader["title"],
                "leader_market_id": leader["market_key"],
                "related_market_title": f"Tradable {leader['market_key']}",
                "related_market_id": f"tradable-{leader['market_key']}",
                "relationship_type": "SAME_OPTION_SET_NEGATIVE_CORRELATION",
                "expected_correlation_direction": "negative",
                "divergence_score": 1.0,
                "tradable_signal": True,
                "option_set_key": option_set_key,
                "rejected_reason": None,
                "excluded_as_self_market": False,
                "excluded_as_duplicate_bucket": False,
            },
            {
                "leader_market_title": leader["title"],
                "leader_market_id": leader["market_key"],
                "related_market_title": f"Review {leader['market_key']}",
                "related_market_id": f"review-{leader['market_key']}",
                "relationship_type": "SAME_OPTION_SET_NEGATIVE_CORRELATION",
                "expected_correlation_direction": "negative",
                "divergence_score": 0.9,
                "tradable_signal": False,
                "option_set_key": option_set_key,
                "rejected_reason": None,
                "excluded_as_self_market": False,
                "excluded_as_duplicate_bucket": False,
            },
        ]

    monkeypatch.setattr(lag_candidates_module, "_leader_candidate_evaluations", fake_evaluations)

    candidates, meta = build_lag_candidates_with_metadata(
        leaders,
        [],
        limit=10,
        max_leaders_evaluated=2,
        min_leaders_before_timeout=0,
    )

    assert len(candidates) == 2
    assert meta["events_represented_count"] == 2
    assert meta["unique_leaders_in_output"] == 2
    assert meta["tradable_signals_count"] == 2
    assert meta["non_tradable_filtered_count"] == 2


def test_lag_candidates_default_suppresses_review_only_candidates():
    leader = {
        "title": "Will a Gaza ceasefire deal happen by June 2026?",
        "market_key": "gaza-ceasefire",
        "one_hour_price_change": 0.4,
        "one_day_price_change": 0.5,
        "_lag_tokens": {"gaza", "ceasefire", "deal", "june"},
    }
    related = {
        "venue": "polymarket",
        "market_key": "gaza-hostages",
        "title": "Will Gaza hostages be released by June 2026?",
        "one_hour_price_change": 0.0,
        "one_day_price_change": 0.0,
        "_lag_tokens": {"gaza", "ceasefire", "deal", "june"},
    }

    candidates, meta = build_lag_candidates_with_metadata(
        [leader], [related], min_similarity=0, min_divergence=0
    )

    assert candidates == []
    assert meta["review_only_candidates_suppressed"] == 1
    assert meta["valid_candidates_found"] == 0


def test_lag_candidates_default_bad_tick_threshold_skips_aoc_size_move():
    leader = {
        "title": "Will Alexandria Ocasio-Cortez win the 2028 US Presidential Election?",
        "market_key": "aoc-2028",
        "one_hour_price_change": 0.882,
        "one_day_price_change": 0.882,
    }
    related = {
        "venue": "polymarket",
        "market_key": "democrats-2028",
        "title": "Will the Democrats win the 2028 US Presidential Election?",
        "one_hour_price_change": 0.02,
        "one_day_price_change": 0.02,
    }

    candidates, meta = build_lag_candidates_with_metadata([leader], [related], min_similarity=0, min_divergence=0)

    assert candidates == []
    assert meta["resolved_or_bad_tick_leaders_skipped"] == 1
    assert meta["max_abs_leader_move"] == 0.8


def test_lag_candidates_skips_leader_with_no_valid_candidates_and_continues():
    """A Japan World Cup leader only finds halftime performers → gets skipped.
    The next leader (ceasefire) finds hostage-release → supplies the output."""
    wc_leader = {
        "title": "Will Japan win the 2026 FIFA World Cup?",
        "market_key": "japan-wc",
        "one_hour_price_change": -0.9,
        "one_day_price_change": -0.9,
    }
    ceasefire_leader = {
        "title": "Will a Gaza ceasefire deal happen by June 2026?",
        "market_key": "gaza-ceasefire",
        "one_hour_price_change": 0.4,
        "one_day_price_change": 0.5,
    }
    universe = [
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

    candidates, meta = build_lag_candidates_with_metadata(
        [wc_leader, ceasefire_leader],
        universe,
        min_similarity=0,
        min_divergence=0,
        max_leaders_evaluated=10,
        include_review_only=True,
        max_abs_leader_move=1.0,
    )

    assert meta["leaders_skipped"] >= 1
    assert meta["skip_reasons_count"].get("no_valid_candidates", 0) >= 1
    assert len(candidates) >= 1
    assert candidates[0]["leader_market_id"] == "gaza-ceasefire"


def test_lag_candidates_with_metadata_all_leaders_skipped_returns_empty():
    leader = {
        "title": "Will Japan win the 2026 FIFA World Cup?",
        "market_key": "japan-wc",
        "one_hour_price_change": -0.9,
        "one_day_price_change": -0.9,
    }
    universe = [
        {
            "venue": "polymarket",
            "market_key": "taylor-halftime",
            "title": "Will Taylor Swift perform at the 2026 World Cup halftime show?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        }
    ]

    candidates, meta = build_lag_candidates_with_metadata(
        [leader], universe, min_similarity=0, min_divergence=0, max_abs_leader_move=1.0
    )

    assert candidates == []
    assert meta["leaders_skipped"] == 1
    assert meta["leaders_evaluated"] == 1
    assert meta["valid_candidates_found"] == 0
    assert "no_valid_candidates" in meta["skip_reasons_count"]


def test_lag_candidates_debug_shows_per_leader_skip_fields():
    leader = {
        "title": "Will Japan win the 2026 FIFA World Cup?",
        "market_key": "japan-wc",
        "one_hour_price_change": -0.9,
        "one_day_price_change": -0.9,
    }
    universe = [
        {
            "venue": "polymarket",
            "market_key": "taylor-halftime",
            "title": "Will Taylor Swift perform at the 2026 World Cup halftime show?",
            "one_hour_price_change": 0.0,
            "one_day_price_change": 0.0,
        }
    ]

    debug = build_lag_candidates_debug([leader], universe, min_similarity=0, min_divergence=0, max_abs_leader_move=1.0)

    assert len(debug) == 1
    row = debug[0]
    assert row["leader_skipped"] is True
    assert row["skip_reason"] == "no_valid_candidates"
    assert row["valid_candidate_count"] == 0
    assert row["rejected_candidate_count"] >= 1
    assert isinstance(row["top_rejection_reasons"], list)
    assert row["best_valid_candidates"] == []
    assert row["option_set_type"] == "winner"
    assert row["leader_option_set_key"] is not None


def test_lag_candidates_debug_evaluates_minimum_leaders_before_timeout(monkeypatch):
    leaders = [
        {
            "title": f"Leader {idx}",
            "market_key": f"leader-{idx}",
            "one_hour_price_change": 0.1,
            "one_day_price_change": 0.1,
        }
        for idx in range(10)
    ]

    ticks = iter([idx * 0.2 for idx in range(30)])
    monkeypatch.setattr(lag_candidates_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(lag_candidates_module, "_leader_candidate_evaluations", lambda *args, **kwargs: [])

    rows = build_lag_candidates_debug(
        leaders,
        [],
        leader_limit=10,
        timeout_seconds=0.1,
        min_leaders_before_timeout=5,
    )

    assert len(rows) == 5
    assert rows[0]["partial_results_due_to_timeout"] is True
    assert rows[0]["leaders_evaluated_before_timeout"] == 5
    assert rows[0]["elapsed_time_total"] > 0


def test_debug_output_goes_to_stderr_not_stdout():
    import io
    import logging

    logger = logging.getLogger("market_fetcher.intelligence")
    capture = io.StringIO()
    handler = logging.StreamHandler(capture)
    handler.setLevel(logging.DEBUG)
    old_level = logger.level
    old_handlers = logger.handlers[:]
    old_propagate = logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    try:
        leader, laggard = _lag_pair()
        signals = detect_lagging_correlated_markets([laggard], [leader])
        debug_text = capture.getvalue()
        assert debug_text, "expected debug output when logger is at DEBUG"
        assert len(signals) == 1, "function must still produce correct signals"
    finally:
        logger.setLevel(old_level)
        logger.handlers = old_handlers
        logger.propagate = old_propagate


def test_threshold_violation_signals_detect_fdv_monotonicity_and_movement_divergence():
    # Leader market has higher FDV threshold (1B) but higher market price (0.55) — structural violation:
    # P(FDV > 1B) should be ≤ P(FDV > 600M), but here 0.55 > 0.40 (inverted).
    # Also: leader falling 15% while related rising 12% — movement divergence.
    leader = {
        "title": "Will MegaETH FDV exceed 1B on launch?",
        "market_key": "megaeth-1b",
        "mid": 0.55,
        "one_hour_price_change": -0.08,
        "one_day_price_change": -0.15,
        "movement_source": "universe_match",
    }
    related = {
        "venue": "polymarket",
        "market_key": "megaeth-600m",
        "title": "Will MegaETH FDV exceed 600M on launch?",
        "mid": 0.40,
        "one_hour_price_change": 0.05,
        "one_day_price_change": 0.12,
        "depth_top5": 100_000,
    }

    candidates, meta = build_lag_candidates_with_metadata(
        [leader],
        [related],
        exploratory=True,
        min_similarity=0,
        min_divergence=0,
        min_similarity_for_trade=0,
    )

    # Threshold-bucket pairs are excluded from candidates and routed to threshold_violation_signals
    assert candidates == []

    signals = meta["threshold_violation_signals"]
    assert meta["threshold_violation_signals_count"] == 1
    assert len(signals) == 1

    signal = signals[0]
    assert signal["high_threshold"] == 1_000_000_000.0
    assert signal["low_threshold"] == 600_000_000.0
    assert "price_monotonicity_violation" in signal["violations"]
    assert "movement_divergence" in signal["violations"]
    assert signal["violation_type"] == "price_monotonicity_violation + movement_divergence"
    assert abs(signal["high_price"] - 0.55) < 1e-6
    assert abs(signal["low_price"] - 0.40) < 1e-6
    assert "cannot have higher probability" in signal["explanation"]
    assert "24h divergence" in signal["explanation"]
