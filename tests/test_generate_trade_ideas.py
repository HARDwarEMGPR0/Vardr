import json
import tempfile
from pathlib import Path

from scripts.generate_trade_ideas import (
    build_prompt,
    get_execution_risk,
    get_market_context,
    load_snapshot,
    sort_lag_signals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _lag_signal(market_key: str, strength: float, title: str = "Will X happen?") -> dict:
    return {
        "type": "lagging_correlated_market",
        "leader_market_key": f"{market_key}-leader",
        "laggard_market_key": market_key,
        "leader_title": f"Leader for {market_key}",
        "laggard_title": title,
        "leader_move": 0.07,
        "laggard_move": 0.001,
        "signal_strength": strength,
        "action": "buy_yes_laggard",
        "relationship": "same_semantic_group",
        "relationship_score": 0.50,
        "relationship_reasons": ["same semantic group: crypto"],
        "expected_direction": "same",
        "reason": f"Leader moved, laggard did not (strength={strength}).",
    }


def _make_snapshot(lag_signals=None, opportunities=None, ref_clusters=None,
                   poly_scores=None, poly_snapshots=None, review_candidates=None) -> dict:
    return {
        "run_id": "test-run",
        "intelligence": {
            "lag_signals": lag_signals if lag_signals is not None else [
                _lag_signal("btc-laggard", 0.08, "Will Bitcoin ETF get approved?"),
                _lag_signal("eth-laggard", 0.04, "Will Ethereum ETF launch?"),
            ],
            "opportunities": opportunities if opportunities is not None else [
                {
                    "type": "high_quality_market",
                    "title": "Will X win?",
                    "market_key": "x-win",
                    "mid": 0.65,
                    "spread": 0.005,
                    "depth_top5": 80_000,
                    "reason": "Tight spread + sufficient depth",
                }
            ],
            "reference_event_clusters": ref_clusters if ref_clusters is not None else [
                {
                    "type": "linked_reference_markets",
                    "reference_event": "GTA VI",
                    "rule_type": "before_reference_event",
                    "price_range": 0.04,
                    "markets": [
                        {"title": "Will Y happen before GTA VI?", "mid": 0.50, "market_key": "g1"},
                        {"title": "Will Z happen before GTA VI?", "mid": 0.52, "market_key": "g2"},
                        {"title": "Will W happen before GTA VI?", "mid": 0.54, "market_key": "g3"},
                    ],
                    "reason": "Markets share a reference-event resolution rule.",
                }
            ],
            "inconsistencies": [],
            "review_candidates": review_candidates if review_candidates is not None else [],
        },
        "scores": {
            "polymarket": poly_scores if poly_scores is not None else [
                {"market_key": "btc-laggard", "execution_risk_index": 0.23},
            ],
        },
        "polymarket_snapshots": poly_snapshots if poly_snapshots is not None else [
            {
                "market_key": "btc-laggard",
                "title": "Will Bitcoin ETF get approved?",
                "mid": 0.55,
                "spread": 0.02,
                "depth_top5": 25_000,
            }
        ],
    }


# ---------------------------------------------------------------------------
# load_snapshot
# ---------------------------------------------------------------------------

def test_load_snapshot_from_file():
    snap = _make_snapshot()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(snap, f)
        tmp = f.name
    try:
        loaded = load_snapshot(tmp)
        assert loaded["run_id"] == "test-run"
        assert "intelligence" in loaded
        assert "lag_signals" in loaded["intelligence"]
    finally:
        Path(tmp).unlink()


# ---------------------------------------------------------------------------
# sort_lag_signals
# ---------------------------------------------------------------------------

def test_sort_lag_signals_descending_by_strength():
    signals = [
        _lag_signal("a", 0.03),
        _lag_signal("b", 0.09),
        _lag_signal("c", 0.06),
    ]
    sorted_signals = sort_lag_signals(signals)
    strengths = [s["signal_strength"] for s in sorted_signals]
    assert strengths == sorted(strengths, reverse=True)
    assert strengths[0] == 0.09


def test_sort_lag_signals_empty_returns_empty():
    assert sort_lag_signals([]) == []


# ---------------------------------------------------------------------------
# get_execution_risk
# ---------------------------------------------------------------------------

def test_get_execution_risk_returns_matching_score():
    scores = [{"market_key": "m1", "execution_risk_index": 0.30, "spread_score": 0.8}]
    risk = get_execution_risk("m1", scores)
    assert risk is not None
    assert risk["execution_risk_index"] == 0.30
    assert "market_key" not in risk


def test_get_execution_risk_returns_none_for_missing_key():
    scores = [{"market_key": "m1", "execution_risk_index": 0.30}]
    assert get_execution_risk("m2", scores) is None


def test_get_execution_risk_returns_none_for_none_key():
    assert get_execution_risk(None, [{"market_key": "m1"}]) is None


# ---------------------------------------------------------------------------
# get_market_context
# ---------------------------------------------------------------------------

def test_get_market_context_returns_spread_depth_mid():
    snaps = [{"market_key": "m1", "mid": 0.55, "spread": 0.02, "depth_top5": 25_000, "extra": "x"}]
    ctx = get_market_context("m1", snaps)
    assert ctx == {"mid": 0.55, "spread": 0.02, "depth_top5": 25_000}


def test_get_market_context_returns_none_for_missing():
    snaps = [{"market_key": "other", "mid": 0.5}]
    assert get_market_context("m1", snaps) is None


# ---------------------------------------------------------------------------
# build_prompt — structure and content
# ---------------------------------------------------------------------------

def test_prompt_includes_system_instruction():
    prompt = build_prompt(_make_snapshot())
    assert "SYSTEM INSTRUCTION" in prompt
    assert "prediction-market trade idea analyst" in prompt


def test_prompt_sorts_lag_signals_by_strength():
    snap = _make_snapshot(lag_signals=[
        _lag_signal("weak", 0.02),
        _lag_signal("strong", 0.10),
    ])
    prompt = build_prompt(snap)
    idx_strong = prompt.index("strong")
    idx_weak = prompt.index("weak")
    assert idx_strong < idx_weak, "stronger signal should appear first"


def test_prompt_includes_laggard_execution_risk():
    snap = _make_snapshot(
        lag_signals=[_lag_signal("btc-laggard", 0.08)],
        poly_scores=[{"market_key": "btc-laggard", "execution_risk_index": 0.42}],
    )
    prompt = build_prompt(snap)
    assert "Execution risk" in prompt
    assert "0.42" in prompt


def test_prompt_includes_laggard_spread_depth_mid():
    snap = _make_snapshot(
        lag_signals=[_lag_signal("btc-laggard", 0.08)],
        poly_snapshots=[{
            "market_key": "btc-laggard",
            "mid": 0.55,
            "spread": 0.02,
            "depth_top5": 25_000,
        }],
    )
    prompt = build_prompt(snap)
    assert "Laggard market" in prompt
    assert "0.55" in prompt
    assert "0.02" in prompt
    assert "25000" in prompt


def test_reference_event_clusters_labeled_as_context_only():
    prompt = build_prompt(_make_snapshot())
    assert "context only" in prompt.lower()
    assert "not standalone trade recommendations" in prompt.lower()


def test_empty_lag_signals_produces_no_strong_trade_ideas():
    snap = _make_snapshot(lag_signals=[])
    prompt = build_prompt(snap)
    assert "No lag signals detected" in prompt
    assert "No strong trade ideas" in prompt


def test_prompt_includes_review_candidates_as_not_trade_recommendations():
    snap = _make_snapshot(
        lag_signals=[],
        review_candidates=[
            {
                "type": "causal_review_candidate",
                "leader_title": "Will Bitcoin hit $1m before GTA VI?",
                "laggard_title": "MicroStrategy sells any Bitcoin by June 30, 2026?",
                "leader_move": 0.05,
                "laggard_move": 0.0,
                "relationship_score": 0.50,
                "reason": "BTC bullishness does not clearly imply MicroStrategy is more likely to sell Bitcoin.",
                "review_question": (
                    "Does movement in the leader market imply same-direction, opposite-direction, "
                    "or no clear movement in the laggard market?"
                ),
            }
        ],
    )
    prompt = build_prompt(snap)
    assert "REVIEW CANDIDATES - not trade recommendations" in prompt
    assert "Do not convert review candidates into trades" in prompt
    assert "If only review candidates exist, say no strong trade exists" in prompt
    assert "MicroStrategy sells any Bitcoin" in prompt
    assert "BTC bullishness does not clearly imply" in prompt
    assert "same-direction, opposite-direction, or no clear movement" in prompt


def test_prompt_includes_opportunities_section():
    prompt = build_prompt(_make_snapshot())
    assert "OPPORTUNITIES" in prompt
    assert "Will X win?" in prompt


def test_prompt_omits_opportunities_section_when_empty():
    snap = _make_snapshot(opportunities=[])
    prompt = build_prompt(snap)
    assert "OPPORTUNITIES" not in prompt


def test_prompt_omits_clusters_section_when_empty():
    snap = _make_snapshot(ref_clusters=[])
    prompt = build_prompt(snap)
    assert "REFERENCE-EVENT CLUSTERS" not in prompt
