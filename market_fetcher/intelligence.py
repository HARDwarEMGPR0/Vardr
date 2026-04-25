from __future__ import annotations

import re

_STOPWORDS = {
    "will", "the", "a", "an", "in", "on", "by", "of", "to", "be", "is",
    "before", "after", "when", "how", "what", "who", "this", "that",
}

_MEANINGFUL_MOVE = 0.03   # leader must move at least this much (absolute)
_LAGGARD_CAP = 0.005      # laggard must be moving less than this (absolute)


def _title_tokens(title: str | None) -> set[str]:
    if not title:
        return set()
    return set(re.sub(r"[^a-z0-9]", " ", title.lower()).split()) - _STOPWORDS


def find_opportunities(markets):
    opportunities = []

    for m in markets:
        spread = m.get("spread")
        depth = m.get("depth_top5")
        mid = m.get("mid")

        if spread is None or depth is None or mid is None:
            continue

        if spread <= 0.01 and depth >= 50_000:
            opportunities.append({
                "type": "high_quality_market",
                "title": m.get("title"),
                "market_key": m.get("market_key"),
                "mid": mid,
                "spread": spread,
                "depth_top5": depth,
                "reason": "Tight spread + sufficient depth",
            })

    return opportunities


def detect_reference_event_clusters(markets):
    """Return linked_reference_markets signals for 'before GTA VI' markets with clustered prices."""
    signals = []

    gta_markets = [
        m for m in markets
        if "before gta vi" in (m.get("title") or "").lower()
    ]

    if len(gta_markets) < 3:
        return signals

    mids = [m["mid"] for m in gta_markets if m.get("mid") is not None]
    if len(mids) < 3:
        return signals

    price_range = max(mids) - min(mids)
    if price_range > 0.05:
        return signals

    signals.append({
        "type": "linked_reference_markets",
        "reference_event": "GTA VI",
        "rule_type": "before_reference_event",
        "price_range": price_range,
        "markets": [
            {
                "title": m.get("title"),
                "mid": m.get("mid"),
                "market_key": m.get("market_key"),
                "one_hour_price_change": m.get("one_hour_price_change"),
                "one_day_price_change": m.get("one_day_price_change"),
                "volume_proxy": m.get("volume_proxy"),
                "liquidity_proxy": m.get("liquidity_proxy"),
            }
            for m in gta_markets
        ],
        "reason": (
            "Markets share a reference-event resolution rule, so clustering may reflect "
            "common timing exposure rather than independent event probabilities."
        ),
    })

    return signals


def detect_inconsistencies(markets):
    """Backward-compatible wrapper; maps detect_reference_event_clusters to the legacy shape."""
    signals = []
    for s in detect_reference_event_clusters(markets):
        signals.append({
            "type": "clustered_probabilities",
            "theme": "GTA VI reference markets",
            "price_range": s["price_range"],
            "markets": [
                {
                    "title": m["title"],
                    "mid": m["mid"],
                    "market_key": m["market_key"],
                }
                for m in s["markets"]
            ],
            "reason": s["reason"],
        })
    return signals


def detect_lagging_correlated_markets(markets, leader_markets):
    """Flag markets that have not repriced after a related leader market moved significantly."""
    signals = []

    for leader in leader_markets:
        lh = leader.get("one_hour_price_change") or 0.0
        ld = leader.get("one_day_price_change") or 0.0
        if abs(lh) < _MEANINGFUL_MOVE and abs(ld) < _MEANINGFUL_MOVE:
            continue

        leader_tokens = _title_tokens(leader.get("title"))
        leader_ref = (leader.get("reference_event") or "").lower()
        # Pick the larger move for display
        representative_move = lh if abs(lh) >= abs(ld) else ld

        for laggard in markets:
            if laggard.get("market_key") == leader.get("market_key"):
                continue

            gh = laggard.get("one_hour_price_change") or 0.0
            gd = laggard.get("one_day_price_change") or 0.0
            if abs(gh) > _LAGGARD_CAP or abs(gd) > _LAGGARD_CAP:
                continue

            relationship: str | None = None
            laggard_ref = (laggard.get("reference_event") or "").lower()
            if leader_ref and laggard_ref and leader_ref == laggard_ref:
                relationship = "shared_reference_event"
            else:
                overlap = leader_tokens & _title_tokens(laggard.get("title"))
                if len(overlap) >= 2:
                    relationship = "shared_keyword_theme"

            if relationship is None:
                continue

            signals.append({
                "type": "lagging_correlated_market",
                "leader_market_key": leader.get("market_key"),
                "laggard_market_key": laggard.get("market_key"),
                "leader_title": leader.get("title"),
                "laggard_title": laggard.get("title"),
                "leader_price_change": representative_move,
                "laggard_price_change": gh if abs(gh) >= abs(gd) else gd,
                "relationship": relationship,
                "reason": (
                    f"Leader market moved {representative_move:+.3f} while related laggard "
                    f"shows no significant price movement ({gh:+.3f} 1h / {gd:+.3f} 1d). "
                    "This may indicate delayed price discovery or insufficient liquidity "
                    "in the laggard market."
                ),
            })

    return signals
