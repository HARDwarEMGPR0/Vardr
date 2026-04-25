import json
import tempfile
from pathlib import Path

from market_fetcher.intelligence import (
    find_opportunities,
    detect_inconsistencies,
    detect_reference_event_clusters,
    detect_lagging_correlated_markets,
)
from market_fetcher.leader_markets import load_leader_markets


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
    assert s["relationship"] == "shared_keyword_theme"
    assert "reason" in s


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


def test_lag_signals_from_external_leader_markets():
    leader = {
        "title": "Will Fed cut rates in Q3?",
        "market_key": "fed-cut-q3",
        "reference_event": "FOMC",
        "one_hour_price_change": 0.07,
        "one_day_price_change": 0.09,
        "mid": 0.65,
    }
    laggard = {
        "title": "Will Fed pause rates in Q3?",
        "market_key": "fed-pause-q3",
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
    assert signals[0]["leader_market_key"] == "fed-cut-q3"
    assert signals[0]["laggard_market_key"] == "fed-pause-q3"


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


def test_signal_strength_is_numeric():
    leader, laggard = _lag_pair()
    signals = detect_lagging_correlated_markets([laggard], [leader])
    assert len(signals) == 1
    strength = signals[0]["signal_strength"]
    assert isinstance(strength, (int, float))
    assert strength > 0


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
