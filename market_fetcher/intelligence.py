from __future__ import annotations

import logging
import re

LOGGER = logging.getLogger(__name__)

_STOPWORDS = {
    "will", "the", "a", "an", "in", "on", "by", "of", "to", "be", "is",
    "before", "after", "when", "how", "what", "who", "this", "that",
    "for", "with", "yes", "no", "market", "markets", "and", "or", "it",
    "hit", "get", "have", "has", "from", "at", "as", "than", "over",
}

_GROUP_TERMS = {
    "crypto": {
        "bitcoin", "btc", "ethereum", "eth", "solana", "crypto",
        "microstrategy", "strategy", "mstr",
    },
    "geopolitics": {
        "china", "taiwan", "war", "invasion", "russia", "ukraine",
        "israel", "iran", "nato",
    },
    "politics": {
        "election", "president", "prime", "minister", "macron", "starmer",
        "trump", "biden", "congress", "parliament",
    },
    "reference_event": {"gta", "gta vi", "gta 6"},
}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _text_tokens(text: str | None) -> set[str]:
    return set(normalize_text(text).split()) - _STOPWORDS


def _title_tokens(title: str | None) -> set[str]:
    return _text_tokens(title)


def _market_text(market: dict) -> str:
    parts = [
        market.get("title"),
        market.get("question"),
        market.get("slug"),
        market.get("reference_event"),
    ]
    return normalize_text(" ".join(str(part) for part in parts if part))


def infer_market_group(market: dict) -> str | None:
    text = _market_text(market)
    if not text:
        return None

    for group in ("crypto", "geopolitics", "politics", "reference_event"):
        for term in _GROUP_TERMS[group]:
            if re.search(rf"\b{re.escape(term)}\b", text):
                return group
    return None


def _same_market(leader: dict, laggard: dict) -> bool:
    leader_key = leader.get("market_key")
    laggard_key = laggard.get("market_key")
    if leader_key and laggard_key and str(leader_key) == str(laggard_key):
        return True

    leader_title = normalize_text(leader.get("title"))
    laggard_title = normalize_text(laggard.get("title"))
    return bool(leader_title and laggard_title and leader_title == laggard_title)


def _reference_event_label(market: dict) -> str | None:
    meta = market.get("resolution_meta") or {}
    meta_ref = normalize_text(meta.get("reference_event"))
    if meta_ref:
        return meta_ref
    ref = normalize_text(market.get("reference_event"))
    text = _market_text(market)
    if ref:
        return ref
    if re.search(r"\bgta\s*(vi|6)?\b", text):
        return "gta vi"
    return None


def _resolution_meta(market: dict) -> dict:
    return market.get("resolution_meta") or {}


def _deadline_confidence(market: dict) -> float:
    try:
        return float(_resolution_meta(market).get("deadline_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resolution_rule_type(market: dict) -> str:
    return str(_resolution_meta(market).get("rule_type") or "unknown")


def _reference_event_causal_direction(leader: dict, laggard: dict, leader_move: float | None = None) -> tuple[str, str] | None:
    leader_meta = _resolution_meta(leader)
    laggard_meta = _resolution_meta(laggard)
    leader_ref = normalize_text(leader_meta.get("reference_event"))
    laggard_ref = normalize_text(laggard_meta.get("reference_event"))
    if not leader_ref or leader_ref != laggard_ref:
        return None

    leader_rule = _resolution_rule_type(leader)
    laggard_rule = _resolution_rule_type(laggard)
    if leader_rule not in {"fixed_date", "release_timing"} or laggard_rule != "before_reference_event":
        return None

    if min(_deadline_confidence(leader), _deadline_confidence(laggard)) < 0.6:
        return None

    if leader_move is not None and leader_move > 0:
        return "opposite", "deadline compression reduces probability of X occurring before event"
    return "unknown", "reference-event timing relationship is known but leader move direction is not usable"


def score_market_relationship(leader: dict, laggard: dict) -> dict:
    if _same_market(leader, laggard):
        return {
            "score": 0.0,
            "relationship": "self",
            "reasons": ["same market_key or normalized title"],
            "expected_direction": "unknown",
        }

    score = 0.0
    reasons: list[str] = []
    expected_direction = "unknown"
    leader_ref = _reference_event_label(leader)
    laggard_ref = _reference_event_label(laggard)
    same_ref = bool(leader_ref and laggard_ref and leader_ref == laggard_ref)

    directional_ref = _reference_event_causal_direction(leader, laggard, _best_move(leader))
    if directional_ref:
        direction, direction_reason = directional_ref
        score += 0.65
        reasons.append(f"resolution-aware reference-event link: {direction_reason}")
        expected_direction = direction
    elif same_ref:
        score += 0.25
        reasons.append(f"shared reference_event: {leader_ref}")

    leader_group = infer_market_group(leader)
    laggard_group = infer_market_group(laggard)
    same_semantic_group = (
        bool(leader_group and laggard_group and leader_group == laggard_group)
        and leader_group != "reference_event"
    )
    if same_semantic_group:
        score += 0.45
        reasons.append(f"same semantic group: {leader_group}")
        expected_direction = "same"

    overlap = _title_tokens(leader.get("title")) & _title_tokens(laggard.get("title"))
    if overlap:
        keyword_score = min(0.20, len(overlap) * 0.05)
        score += keyword_score
        reasons.append("meaningful title keyword overlap: " + ", ".join(sorted(overlap)))

    score = min(1.0, score)

    if directional_ref and same_semantic_group:
        relationship = "same_semantic_group_and_reference_event"
    elif directional_ref:
        relationship = "resolution_aware_reference_event"
    elif same_semantic_group and same_ref:
        relationship = "same_semantic_group_and_reference_event"
    elif same_semantic_group:
        relationship = "same_semantic_group"
    elif same_ref or (leader_group == laggard_group == "reference_event"):
        relationship = "reference_event_only"
        expected_direction = "unknown"
    else:
        relationship = "weak_or_unknown"

    return {
        "score": round(score, 4),
        "relationship": relationship,
        "reasons": reasons,
        "expected_direction": expected_direction,
    }


def is_ambiguous_causal_pair(leader: dict, laggard: dict) -> tuple[bool, str]:
    relationship_score = score_market_relationship(leader, laggard)
    leader_move = _best_move(leader)
    leader_text = _market_text(leader)
    laggard_text = _market_text(laggard)
    same_ref = bool(_reference_event_label(leader) and _reference_event_label(leader) == _reference_event_label(laggard))

    if same_ref and min(_deadline_confidence(leader), _deadline_confidence(laggard)) < 0.5:
        return True, "insufficient resolution-rule confidence"

    btc_leader = bool(re.search(r"\b(bitcoin|btc)\b", leader_text))
    btc_price_up = bool(
        leader_move is not None
        and leader_move > 0
        and (
            re.search(r"\b(rally|bull|bullish|rise|rises|up|higher|high|record|1m|100k)\b", leader_text)
            or btc_leader
        )
    )
    mstr_sells_btc = bool(
        re.search(r"\b(microstrategy|strategy|mstr)\b", laggard_text)
        and re.search(r"\b(sell|sells|selling|sold)\b", laggard_text)
        and re.search(r"\b(bitcoin|btc)\b", laggard_text)
    )
    if btc_leader and btc_price_up and mstr_sells_btc:
        return True, "BTC bullishness does not clearly imply MicroStrategy is more likely to sell Bitcoin."

    if relationship_score["relationship"] == "reference_event_only":
        return True, "Shared reference-event exposure does not establish a causal lag relationship."

    if same_ref and relationship_score["relationship"] != "resolution_aware_reference_event":
        return True, "Shared reference-event exposure is not tradable without high-confidence causal timing rules."

    if relationship_score["expected_direction"] == "unknown":
        return True, "Expected causal direction is unknown."

    if relationship_score["score"] <= 0.45:
        return True, "Relationship score is borderline for a direct trade signal."

    return False, ""


def _lag_action(leader_move: float, expected_direction: str) -> str:
    if expected_direction == "same":
        return "buy_yes_laggard" if leader_move > 0 else "buy_no_laggard"
    return "buy_no_laggard" if leader_move > 0 else "buy_yes_laggard"


def _signal_strength(leader_move: float, laggard_move: float | None, laggard: dict, relationship_score: dict) -> float:
    signal_strength = (
        abs(leader_move) - abs(laggard_move or 0.0)
    ) * relationship_score["score"]
    if laggard.get("spread") is not None and laggard["spread"] <= 0.05:
        signal_strength += 0.05
    if laggard.get("depth_top5") is not None and laggard["depth_top5"] >= 10_000:
        signal_strength += 0.05
    return round(signal_strength, 4)


def _review_candidate(
    leader: dict,
    laggard: dict,
    leader_move: float,
    laggard_move: float | None,
    relationship_score: dict,
    reason: str,
) -> dict:
    meta = _resolution_meta(laggard)
    return {
        "type": "causal_review_candidate",
        "leader_title": leader.get("title"),
        "laggard_title": laggard.get("title"),
        "leader_market_key": leader.get("market_key"),
        "laggard_market_key": laggard.get("market_key"),
        "leader_move": leader_move,
        "laggard_move": laggard_move,
        "relationship": relationship_score["relationship"],
        "relationship_score": relationship_score["score"],
        "resolution_deadline_utc": meta.get("resolution_deadline_utc"),
        "deadline_confidence": meta.get("deadline_confidence"),
        "rule_type": meta.get("rule_type"),
        "reason": reason,
        "review_question": (
            "Does movement in the leader market imply same-direction, opposite-direction, "
            "or no clear movement in the laggard market?"
        ),
    }


def _lag_signal(
    leader: dict,
    laggard: dict,
    leader_move: float,
    laggard_move: float | None,
    relationship_score: dict,
) -> dict:
    expected_direction = relationship_score["expected_direction"]
    action = _lag_action(leader_move, expected_direction)
    laggard_move_display = laggard_move if laggard_move is not None else 0.0
    signal_strength = _signal_strength(leader_move, laggard_move, laggard, relationship_score)
    meta = _resolution_meta(laggard)
    return {
        "type": "lagging_correlated_market",
        "leader_market_key": leader.get("market_key"),
        "laggard_market_key": laggard.get("market_key"),
        "leader_title": leader.get("title"),
        "laggard_title": laggard.get("title"),
        "leader_move": leader_move,
        "laggard_move": laggard_move,
        "signal_strength": signal_strength,
        "action": action,
        "relationship": relationship_score["relationship"],
        "relationship_score": relationship_score["score"],
        "relationship_reasons": relationship_score["reasons"],
        "expected_direction": expected_direction,
        "resolution_deadline_utc": meta.get("resolution_deadline_utc"),
        "deadline_confidence": meta.get("deadline_confidence"),
        "rule_type": meta.get("rule_type"),
        "reason": (
            f"Leader market moved {leader_move:+.3f} while related laggard "
            f"shows no significant price movement ({laggard_move_display:+.3f}). "
            f"Relationship '{relationship_score['relationship']}' scored "
            f"{relationship_score['score']:.2f}, suggesting delayed price "
            "discovery may be tradable in the laggard market."
        ),
    }


def _best_move(m: dict) -> float | None:
    """Return the best available price movement: delta > 1h change > 1d change."""
    delta = m.get("computed_mid_delta")
    if delta is not None:
        return delta
    h = m.get("one_hour_price_change")
    if h is not None:
        return h
    d = m.get("one_day_price_change")
    if d is not None:
        return d
    return None


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


def _detect_lagging_correlated_markets_legacy(
    markets,
    leader_markets,
    leader_threshold: float = 0.02,
    laggard_threshold: float = 0.01,
):
    """Flag markets that have not repriced after a related leader market moved significantly.

    Movement priority per market: computed_mid_delta > one_hour_price_change > one_day_price_change.
    A leader qualifies if abs(move) >= leader_threshold.
    A laggard qualifies if move is None or abs(move) <= laggard_threshold.
    """
    signals = []

    for leader in leader_markets:
        leader_move = _best_move(leader)
        LOGGER.debug("leader %s | move=%s", leader.get("market_key"), leader_move)

        if leader_move is None or abs(leader_move) < leader_threshold:
            LOGGER.debug("  skipped — move below threshold")
            continue

        leader_tokens = _title_tokens(leader.get("title"))
        leader_ref = (leader.get("reference_event") or "").lower()

        for laggard in markets:
            if laggard.get("market_key") == leader.get("market_key"):
                continue

            laggard_move = _best_move(laggard)
            if laggard_move is not None and abs(laggard_move) > laggard_threshold:
                LOGGER.debug(
                    "  laggard %s skipped — move=%s above threshold",
                    laggard.get("market_key"), laggard_move,
                )
                continue

            relationship: str | None = None
            laggard_ref = (laggard.get("reference_event") or "").lower()
            if leader_ref and laggard_ref and leader_ref == laggard_ref:
                relationship = "legacy_reference_event"
            else:
                overlap = leader_tokens & _title_tokens(laggard.get("title"))
                if len(overlap) >= 2:
                    relationship = "legacy_keyword_theme"

            if relationship is None:
                LOGGER.debug(
                    "  laggard %s skipped — no relationship with leader",
                    laggard.get("market_key"),
                )
                continue

            # Signal strength: base = leader move advantage, bonuses for tight/deep laggard
            signal_strength = abs(leader_move) - abs(laggard_move or 0.0)
            if laggard.get("spread") is not None and laggard["spread"] <= 0.05:
                signal_strength += 0.05
            if laggard.get("depth_top5") is not None and laggard["depth_top5"] >= 10_000:
                signal_strength += 0.05

            action = "buy_yes_laggard" if leader_move > 0 else "buy_no_laggard"
            laggard_move_display = laggard_move if laggard_move is not None else 0.0

            LOGGER.debug(
                "  → signal emitted | laggard=%s strength=%.4f action=%s",
                laggard.get("market_key"), signal_strength, action,
            )

            signals.append({
                "type": "lagging_correlated_market",
                "leader_market_key": leader.get("market_key"),
                "laggard_market_key": laggard.get("market_key"),
                "leader_title": leader.get("title"),
                "laggard_title": laggard.get("title"),
                "leader_move": leader_move,
                "laggard_move": laggard_move,
                "signal_strength": round(signal_strength, 4),
                "action": action,
                "relationship": relationship,
                "reason": (
                    f"Leader market moved {leader_move:+.3f} while related laggard "
                    f"shows no significant price movement ({laggard_move_display:+.3f}). "
                    "This may indicate delayed price discovery or insufficient liquidity "
                    "in the laggard market."
                ),
            })

    return signals


def detect_lagging_correlated_markets(
    markets,
    leader_markets,
    leader_threshold: float = 0.02,
    laggard_threshold: float = 0.01,
    relationship_threshold: float = 0.45,
):
    """Return only direct lag signals, preserving the historical list return type."""
    return detect_lagging_correlated_markets_with_review(
        markets=markets,
        leader_markets=leader_markets,
        leader_threshold=leader_threshold,
        laggard_threshold=laggard_threshold,
        relationship_threshold=relationship_threshold,
    )["lag_signals"]


def detect_lagging_correlated_markets_with_review(
    markets,
    leader_markets,
    leader_threshold: float = 0.02,
    laggard_threshold: float = 0.01,
    relationship_threshold: float = 0.45,
) -> dict:
    """Split correlated lag opportunities into direct signals and review candidates."""
    lag_signals = []
    review_candidates = []

    for leader in leader_markets:
        leader_move = _best_move(leader)
        LOGGER.debug("leader %s | move=%s", leader.get("market_key"), leader_move)

        if leader_move is None or abs(leader_move) < leader_threshold:
            LOGGER.debug(
                "  skipped leader below threshold | leader=%s move=%s threshold=%s",
                leader.get("market_key"), leader_move, leader_threshold,
            )
            continue

        for laggard in markets:
            relationship_score = score_market_relationship(leader, laggard)
            if relationship_score["relationship"] == "self":
                LOGGER.debug(
                    "  skipped self | leader=%s laggard=%s",
                    leader.get("market_key"), laggard.get("market_key"),
                )
                continue

            laggard_move = _best_move(laggard)
            if laggard_move is not None and abs(laggard_move) > laggard_threshold:
                LOGGER.debug(
                    "  skipped laggard already moved | laggard=%s move=%s threshold=%s",
                    laggard.get("market_key"), laggard_move, laggard_threshold,
                )
                continue

            if relationship_score["score"] < relationship_threshold:
                ambiguous, ambiguous_reason = is_ambiguous_causal_pair(leader, laggard)
                if ambiguous and relationship_score["score"] > 0:
                    review_candidates.append(
                        _review_candidate(
                            leader,
                            laggard,
                            leader_move,
                            laggard_move,
                            relationship_score,
                            ambiguous_reason,
                        )
                    )
                    LOGGER.debug(
                        "  review candidate | laggard=%s relationship=%s score=%s reason=%s",
                        laggard.get("market_key"),
                        relationship_score["relationship"],
                        relationship_score["score"],
                        ambiguous_reason,
                    )
                    continue
                LOGGER.debug(
                    "  skipped weak relationship score | laggard=%s score=%s threshold=%s relationship=%s",
                    laggard.get("market_key"),
                    relationship_score["score"],
                    relationship_threshold,
                    relationship_score["relationship"],
                )
                continue

            expected_direction = relationship_score["expected_direction"]
            ambiguous, ambiguous_reason = is_ambiguous_causal_pair(leader, laggard)
            if ambiguous:
                review_candidates.append(
                    _review_candidate(
                        leader,
                        laggard,
                        leader_move,
                        laggard_move,
                        relationship_score,
                        ambiguous_reason,
                    )
                )
                LOGGER.debug(
                    "  review candidate | laggard=%s relationship=%s score=%s reason=%s",
                    laggard.get("market_key"),
                    relationship_score["relationship"],
                    relationship_score["score"],
                    ambiguous_reason,
                )
                continue

            if expected_direction == "unknown":
                LOGGER.debug(
                    "  skipped unknown expected direction | laggard=%s relationship=%s reasons=%s",
                    laggard.get("market_key"),
                    relationship_score["relationship"],
                    relationship_score["reasons"],
                )
                continue

            signal = _lag_signal(leader, laggard, leader_move, laggard_move, relationship_score)

            LOGGER.debug(
                "  signal emitted | laggard=%s strength=%.4f action=%s relationship=%s score=%s",
                laggard.get("market_key"), signal["signal_strength"], signal["action"],
                relationship_score["relationship"], relationship_score["score"],
            )

            lag_signals.append(signal)

    return {"lag_signals": lag_signals, "review_candidates": review_candidates}
