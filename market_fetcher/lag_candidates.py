from __future__ import annotations

import json
import logging
import os
import re
import time
import hashlib
from ast import literal_eval
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.normalize import normalize_polymarket_market
from src.scoring import compute_execution_risk_details

LOGGER = logging.getLogger(__name__)

DEFAULT_MIN_SIMILARITY = 0.05
DEFAULT_MIN_DIVERGENCE = 0.0
STRONG_MIN_SIMILARITY = 0.18
STRONG_MIN_DIVERGENCE = 0.02
DEFAULT_TIMEOUT_SECONDS = 45.0
MIN_LEADER_MOVE = 0.02
DEFAULT_MIN_RELATED_ABS_MOVE_24H = 0.01
DEFAULT_MIN_SIMILARITY_FOR_TRADE = 0.60
NEAR_ZERO_MOVE = 1e-9
DEFAULT_MAX_ABS_LEADER_MOVE = 0.80
DEFAULT_MAX_CANDIDATES_PER_LEADER_OUTPUT = 4
DEFAULT_MAX_CANDIDATES_PER_EVENT_OUTPUT = 2
DEFAULT_LEADER_FETCH_LIMIT = 100
DEFAULT_MAX_LEADERS_EVALUATED = 25
DEFAULT_MIN_VALID_CANDIDATES = 1
DEFAULT_VARDR1_MARKETS_PATH = (
    r"C:\Users\trace\market_fetcher_project\market_fetcher\Vardr-1-sandbox\data\raw\polymarket_markets.parquet"
)

_STOPWORDS = {
    "will",
    "the",
    "a",
    "an",
    "in",
    "on",
    "by",
    "of",
    "to",
    "be",
    "is",
    "for",
    "with",
    "and",
    "or",
    "this",
    "that",
    "market",
    "markets",
    "win",
    "winner",
    "2024",
    "2025",
    "2026",
    "2027",
    "2028",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

_ALIASES = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "gta": "gta",
    "vi": "6",
    "fifa": "world",
    "soccer": "world",
}

REL_SAME_EVENT_OUTCOME = "SAME_EVENT_OUTCOME"  # legacy; no longer emitted
REL_DUPLICATE_BUCKET = "DUPLICATE_BUCKET"
REL_CAUSALLY_LINKED = "CAUSALLY_LINKED"
REL_CORRELATED_BUT_WEAK = "CORRELATED_BUT_WEAK"
REL_UNRELATED = "UNRELATED"
REL_SAME_OPTION_SET_NEGATIVE_CORRELATION = "SAME_OPTION_SET_NEGATIVE_CORRELATION"
REL_SAME_OPTION_SET_POSITIVE_CORRELATION = "SAME_OPTION_SET_POSITIVE_CORRELATION"
REL_CROSS_ROLE_SAME_ENTITY = "CROSS_ROLE_SAME_ENTITY"
REL_SHARED_EVENT_ONLY = "SHARED_EVENT_ONLY"

DIR_POSITIVE = "positive"
DIR_NEGATIVE = "negative"
DIR_UNCLEAR = "unclear"

# Relationship type categories for validity classification
STRICTLY_VALID_RELATIONSHIPS = frozenset({
    REL_CAUSALLY_LINKED,
    REL_SAME_OPTION_SET_NEGATIVE_CORRELATION,
    REL_SAME_OPTION_SET_POSITIVE_CORRELATION,
})
EXPLORATORY_OPTIONAL_RELATIONSHIPS = frozenset({
    REL_CROSS_ROLE_SAME_ENTITY,
})
ALWAYS_INVALID_RELATIONSHIPS = frozenset({
    REL_CORRELATED_BUT_WEAK,
    REL_SHARED_EVENT_ONLY,
    REL_UNRELATED,
    REL_DUPLICATE_BUCKET,
    REL_SAME_EVENT_OUTCOME,
})

_ENTITY_GROUP_MAP: dict[str, tuple[str, ...]] = {
    "glenn youngkin": ("republican", "republicans", "gop"),
    "marco rubio": ("republican", "republicans", "gop"),
    "ted cruz": ("republican", "republicans", "gop"),
    "aoc": ("democrat", "democrats", "democratic"),
    "alexandria ocasio cortez": ("democrat", "democrats", "democratic"),
    "gretchen whitmer": ("democrat", "democrats", "democratic"),
    "pete buttigieg": ("democrat", "democrats", "democratic"),
    "buttigieg": ("democrat", "democrats", "democratic"),
    "raphael warnock": ("democrat", "democrats", "democratic"),
    "jon ossoff": ("democrat", "democrats", "democratic"),
    "ossoff": ("democrat", "democrats", "democratic"),
    "tim walz": ("democrat", "democrats", "democratic"),
    "walz": ("democrat", "democrats", "democratic"),
    "japan": ("asia", "asian"),
    "south korea": ("asia", "asian"),
    "brazil": ("south america", "south american"),
    "argentina": ("south america", "south american"),
    "france": ("europe", "european"),
    "england": ("europe", "european"),
    "spain": ("europe", "european"),
    "switzerland": ("europe", "european"),
    "ghana": ("africa", "african"),
    "haiti": ("north america", "north american", "concacaf"),
    "new zealand": ("oceania", "oceanian"),
}


def _is_valid_relationship_type(
    relationship_type: str,
    *,
    exploratory: bool,
    include_weak_similarity: bool,
) -> bool:
    return (
        relationship_type in STRICTLY_VALID_RELATIONSHIPS
        or relationship_type == REL_CROSS_ROLE_SAME_ENTITY
        or (include_weak_similarity and relationship_type == REL_CORRELATED_BUT_WEAK)
        or (exploratory and relationship_type in EXPLORATORY_OPTIONAL_RELATIONSHIPS)
    )

_PERFORMER_PATTERNS = (
    r"\b(perform|performs|performing|performer)\b",
    r"\bhalftime\b",
    r"\bhalf.time\b",
)

_POLITICAL_ROLE_TOKENS: frozenset[str] = frozenset({
    "republican", "democratic", "democrat", "gop", "party",
    "presidential", "vice", "vp", "nominee", "nomination",
    "president", "primary", "candidate",
})

_MULTI_OUTCOME_PATTERNS = (
    r"\bwin(s|ning)?\b.*\b(world cup|nba finals|super bowl|championship|election|nomination|primary|mvp|award|ballon d or)\b",
    r"\b(world cup|nba finals|super bowl|championship|election|nomination|primary|mvp|award|ballon d or)\b.*\bwin(s|ner)?\b",
    r"\bbe (the )?(next )?(president|nominee|winner|mvp|champion)\b",
)

_CAUSAL_TERMS = {
    "ceasefire",
    "truce",
    "peace",
    "deal",
    "agreement",
    "withdraw",
    "withdrawal",
    "capture",
    "control",
    "sanction",
    "sanctions",
    "launch",
    "approve",
    "approved",
    "approval",
    "resign",
    "sentenced",
    "sentence",
    "confirmed",
    "released",
    "release",
    "airdrop",
    "lawsuit",
    "etf",
    "rate",
    "cut",
    "hike",
}

_BINARY_EVENT_TERMS = {
    "happen",
    "occur",
    "pass",
    "launch",
    "resign",
    "approved",
    "approve",
    "confirmed",
    "released",
    "sentenced",
    "capture",
    "withdraw",
    "ceasefire",
    "airdrop",
}

_BUCKET_TERMS = {
    "between",
    "under",
    "over",
    "above",
    "below",
    "range",
    "less",
    "more",
    "at least",
    "by",
    "before",
    "after",
}


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _tokens(text: str | None) -> set[str]:
    tokens = set(_normalize_text(text).split()) - _STOPWORDS
    return {_ALIASES.get(token, token) for token in tokens}


def _market_tokens(market: dict[str, Any]) -> set[str]:
    cached = market.get("_lag_tokens")
    if isinstance(cached, set):
        return cached
    if isinstance(cached, (list, tuple)):
        return set(str(token) for token in cached)
    tokens = _tokens(_market_text(market))
    market["_lag_tokens"] = tokens
    return tokens


def _market_text(market: dict[str, Any]) -> str:
    return " ".join(
        str(part)
        for part in (
            market.get("title"),
            market.get("question"),
            market.get("slug"),
            market.get("reference_event"),
        )
        if part
    )


def _title(market: dict[str, Any]) -> str:
    return str(market.get("title") or market.get("question") or market.get("market_title") or "")


def _is_multi_outcome_title(title: str | None) -> bool:
    text = _normalize_text(title)
    return any(re.search(pattern, text) for pattern in _MULTI_OUTCOME_PATTERNS)


def _is_performer_type(text: str) -> bool:
    return any(re.search(p, text) for p in _PERFORMER_PATTERNS)


def _is_same_entity_cross_role(leader: dict[str, Any], related: dict[str, Any]) -> bool:
    leader_entity = _market_tokens(leader) - _POLITICAL_ROLE_TOKENS
    related_entity = _market_tokens(related) - _POLITICAL_ROLE_TOKENS
    return bool(leader_entity & related_entity)


# Known event keywords for fallback extraction
_KNOWN_WINNER_EVENTS = frozenset({
    "world cup", "fifa world cup", "soccer world cup",
    "super bowl", "nfl championship",
    "nba finals", "nba championship",
    "nhl finals", "nhl stanley cup",
    "premier league", "champions league",
    "wimbledon", "us open", "french open", "australian open",
    "masters", "pga championship", "us pga championship",
    "final", "finals", "championship", "champion",
    "cup", "league", "tournament",
})

_BROAD_EVENT_KEYWORDS = (
    "fifa world cup",
    "world cup",
    "super bowl",
    "nba finals",
    "nba championship",
    "nhl finals",
    "stanley cup",
    "premier league",
    "champions league",
    "wimbledon",
    "us open",
    "french open",
    "australian open",
    "masters",
    "pga championship",
    "championship",
    "election",
    "nomination",
    "primary",
)


def _extract_option_set_key(title: str | None) -> tuple[str, str, str] | None:
    """Extract and normalize (event, year, market_type) from a winner market title.
    
    Uses regex patterns as primary extraction method, then fallback to known event keywords.
    
    Example:
    "Will Japan win the 2026 FIFA World Cup?" → ("fifa world cup", "2026", "winner")
    "Will France be the 2026 World Cup winner?" → ("world cup", "2026", "winner")
    "Will team X win the Super Bowl in 2025?" → ("super bowl", "2025", "winner")
    
    Returns None if this is not a winner market, but tries hard to extract from known keywords.
    """
    if not title:
        return None
    
    text = _normalize_text(title)
    original_title = str(title)  # Keep for debug logging
    
    # Detect if this looks like a winner market
    has_win_keyword = bool(re.search(r"\bwin(s|ning|ner)?\b", text))
    
    if not has_win_keyword:
        LOGGER.debug("extract_option_set_key no_win_keyword title=%r", original_title)
        return None
    
    event_part = None
    extraction_method = None
    
    # ===== PRIMARY EXTRACTION: Try regex patterns =====
    
    # Pattern 1: will X win [the] EVENT
    match = re.search(r"\bwill\s+(.+?)\s+win\s+(?:the\s+)?(.+?)(?:\?)?$", text)
    if match:
        event_part = match.group(2).strip()
        extraction_method = "pattern_will_x_win_event"
    
    # Pattern 2: will X be the EVENT winner
    if not event_part:
        match = re.search(r"\bwill\s+(.+?)\s+be\s+(?:the\s+)?(.+?)\s+winner(?:\?)?$", text)
        if match:
            event_part = match.group(2).strip()
            extraction_method = "pattern_will_x_be_winner"
    
    # Pattern 3: win the {year} {event}
    if not event_part:
        match = re.search(r"\bwin\s+the\s+(20\d{2}|19\d{2})\s+(.+?)(?:\?)?$", text)
        if match:
            year_prefix = match.group(1)
            event_part = f"{year_prefix} {match.group(2)}".strip()
            extraction_method = "pattern_win_the_year_event"
    
    # Pattern 4: win {event} {year}
    if not event_part:
        match = re.search(r"\bwin\s+(.+?)\s+(20\d{2}|19\d{2})(?:\?)?$", text)
        if match:
            event_part = f"{match.group(1)} {match.group(2)}".strip()
            extraction_method = "pattern_win_event_year"
    
    # ===== FALLBACK EXTRACTION: Look for known event keywords =====
    
    if not event_part and has_win_keyword:
        LOGGER.debug("extract_option_set_key trying fallback extraction title=%r", original_title)
        
        # Look for known event keywords in the text
        text_lower = text.lower()
        matched_events = []
        
        for event_keyword in _KNOWN_WINNER_EVENTS:
            if event_keyword in text_lower:
                matched_events.append(event_keyword)
        
        if matched_events:
            # Use the longest match (most specific)
            known_event = max(matched_events, key=len)
            event_part = known_event
            extraction_method = "fallback_known_keywords"
            LOGGER.debug(
                "extract_option_set_key fallback_matched title=%r event_keyword=%r",
                original_title,
                known_event,
            )
    
    # If still no event_part, we failed
    if not event_part:
        LOGGER.warning(
            "extract_option_set_key failed: no event extracted title=%r has_win=%s",
            original_title,
            has_win_keyword,
        )
        return None
    
    # ===== CLEAN UP EVENT PART =====
    
    event_part = re.sub(r"\s+winner\s*(?:\?)?$", "", event_part).strip()
    event_part = re.sub(r"\b(yes|no)\b", "", event_part).strip()
    
    if not event_part:
        LOGGER.warning("extract_option_set_key failed: event_part empty after cleanup title=%r", original_title)
        return None
    
    # ===== EXTRACT YEAR =====
    
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", event_part)
    year = year_match.group(1) if year_match else ""
    
    # ===== NORMALIZE EVENT =====
    
    # Remove year from event for normalization
    event_normalized = re.sub(r"\b(?:in|by)\s+(20\d{2}|19\d{2})\b", " ", event_part).strip()
    event_normalized = re.sub(r"\b(20\d{2}|19\d{2})\b", " ", event_normalized).strip()
    
    # Standardize common phrases
    event_normalized = re.sub(r"\bwinner\s+of\b", "", event_normalized)
    event_normalized = re.sub(r"\bwinner?\b", "", event_normalized)
    
    # Remove prepositions
    event_normalized = re.sub(r"\s+(in|by|at|of)\s+", " ", event_normalized)
    
    # Clean up whitespace
    event_normalized = re.sub(r"\s+", " ", event_normalized).strip()
    
    if not event_normalized:
        LOGGER.warning("extract_option_set_key failed: event_normalized empty title=%r", original_title)
        return None
    
    event_key = event_normalized.lower()
    
    result = (event_key, year, "winner")
    LOGGER.warning(
        "extract_option_set_key success title=%r method=%s result=%s",
        original_title,
        extraction_method,
        result,
    )
    return result


def _win_event_suffix(title: str | None) -> str | None:
    """Legacy function for backward compatibility. Extracts event suffix."""
    text = _normalize_text(title)
    match = re.search(r"\bwill\s+(.+?)\s+win\s+(?:the\s+)?(.+)$", text)
    if not match:
        return None
    suffix = match.group(2)
    suffix = re.sub(r"\b(yes|no)\b", " ", suffix)
    return " ".join(suffix.split()) or None


def _option_set_key_for_pair(leader: dict[str, Any], related: dict[str, Any]) -> str | None:
    """Return a string representation of matched option_set_key for reporting."""
    leader_key = _extract_option_set_key(_title(leader))
    related_key = _extract_option_set_key(_title(related))
    
    if leader_key and related_key and leader_key == related_key:
        # Return tuple as string for JSON serialization
        return str(leader_key)
    
    # Fallback to old behavior
    suffix = _win_event_suffix(_title(leader))
    if suffix:
        return re.sub(r"\s+", "_", suffix.strip())
    shared = _market_tokens(leader) & _market_tokens(related)
    return "_".join(sorted(shared)) if shared else None


def _leader_option_set_type(leader: dict[str, Any]) -> tuple[str | None, tuple[str, str, str] | None]:
    """Return (option_set_type, option_set_key) if this leader belongs to a multi-outcome option set.

    Winner markets (World Cup winner, election nominee, etc.) return ('winner', (event, year, market_type)).
    Performer markets (halftime show, etc.) return ('performer', None).
    All other leaders return (None, None) and are compared against the full universe.
    """
    title = _title(leader)
    text = _normalize_text(_market_text(leader))
    leader_repr = leader.get("market_key") or leader.get("market_id") or title[:50]
    
    # Check if this is a winner market
    if _is_multi_outcome_title(title):
        LOGGER.warning("leader_option_set_type is_multi_outcome_title title=%r", title[:80])
        winner_key = _extract_option_set_key(title)
        if winner_key:
            LOGGER.warning(
                "leader_option_set_type winner_extracted leader=%r option_set_key=%s",
                leader_repr,
                winner_key,
            )
            return "winner", winner_key
        else:
            LOGGER.warning(
                "leader_option_set_type is_multi_outcome_but_extraction_failed leader=%r title=%r",
                leader_repr,
                title[:80],
            )
    
    # Fallback: check if this looks like a winner market (has "win" + known keywords)
    if "win" in text:
        winner_key = _extract_option_set_key(title)
        if winner_key:
            LOGGER.warning(
                "leader_option_set_type fallback_winner_extracted leader=%r option_set_key=%s",
                leader_repr,
                winner_key,
            )
            return "winner", winner_key
    
    # Check if this is a performer market
    if _is_performer_type(text):
        LOGGER.warning("leader_option_set_type is_performer_type leader=%r", leader_repr)
        return "performer", None
    
    LOGGER.debug("leader_option_set_type no_option_set leader=%r title=%r", leader_repr, title[:80])
    return None, None


def _numbers_and_dates(text: str | None) -> set[str]:
    normalized = _normalize_text(text)
    numbers = set(re.findall(r"\b\d+(?:\.\d+)?(?:k|m|b|%|x)?\b", normalized))
    dates = {
        token
        for token in normalized.split()
        if token
        in {
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "2024",
            "2025",
            "2026",
            "2027",
            "2028",
            "2029",
            "2030",
        }
    }
    return numbers | dates


def _has_any(text: str | None, terms: set[str]) -> bool:
    normalized = _normalize_text(text)
    return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms)


def _has_binary_event_shape(market: dict[str, Any]) -> bool:
    text = _normalize_text(_market_text(market))
    if "?" not in _title(market):
        return False
    return (
        bool(re.search(r"\b(by|before|after)\b", text))
        or _has_any(text, _BINARY_EVENT_TERMS)
        or bool(re.search(r"\bwill\s+.+\s+(happen|occur|launch|resign|withdraw|capture|approve|airdrop)\b", text))
    )


def _shared_broad_event(left_text: str, right_text: str) -> str | None:
    for keyword in _BROAD_EVENT_KEYWORDS:
        if keyword in left_text and keyword in right_text:
            return keyword
    return None


def _option_subject(title: str | None) -> str | None:
    text = _normalize_text(title)
    if not text:
        return None
    patterns = (
        r"\bwill\s+(.+?)\s+win\s+(?:the\s+)?(.+?)(?:\?)?$",
        r"\bwill\s+(.+?)\s+be\s+(?:the\s+)?(.+?)\s+winner(?:\?)?$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            subject = re.sub(r"\b(yes|no)\b", " ", match.group(1))
            subject = re.sub(r"\s+", " ", subject).strip()
            return subject or None
    return None


def _subset_relationship_reason(leader_title: str | None, related_title: str | None) -> str | None:
    leader_subject = _option_subject(leader_title)
    related_subject = _option_subject(related_title)
    if not leader_subject or not related_subject:
        return None

    related_text = _normalize_text(related_subject)
    leader_text = _normalize_text(leader_title)
    related_full_text = _normalize_text(related_title)

    if leader_subject == "mrbeast":
        if "democratic nomination" not in leader_text and "democratic nomination" not in related_full_text:
            return None
        groups = ("democrat", "democrats", "democratic")
    else:
        groups = _ENTITY_GROUP_MAP.get(leader_subject)

    if not groups:
        return None

    for group in groups:
        if re.search(rf"\b{re.escape(group)}\b", related_text):
            return f"{leader_subject} is part of {group}"
    return None


def _is_subset_relationship(leader_title: str | None, related_title: str | None) -> bool:
    return _subset_relationship_reason(leader_title, related_title) is not None


def _resolved_or_bad_tick_candidate(
    leader: dict[str, Any],
    max_abs_leader_move: float = DEFAULT_MAX_ABS_LEADER_MOVE,
) -> bool:
    return (
        abs(_change(leader, "one_hour_price_change")) > max(0.0, float(max_abs_leader_move))
        or abs(_change(leader, "one_day_price_change")) > max(0.0, float(max_abs_leader_move))
    )


def _leader_current_price(leader: dict[str, Any]) -> float | None:
    return _to_float(
        _first_present(
            leader,
            (
                "current_price",
                "lasttradeprice",
                "lastTradePrice",
                "last_trade_price",
                "mid",
            ),
        )
    )


def _leader_prefilter_skip_reason(
    leader: dict[str, Any],
    *,
    max_abs_leader_move: float = DEFAULT_MAX_ABS_LEADER_MOVE,
) -> str | None:
    one_hour_change = _change(leader, "one_hour_price_change")
    one_day_change = _change(leader, "one_day_price_change")
    if _resolved_or_bad_tick_candidate(leader, max_abs_leader_move=max_abs_leader_move):
        return "resolved_or_bad_tick_candidate"
    if abs(one_hour_change) > 0.5 and abs(one_day_change) < 0.05:
        return "inconsistent_spike_candidate"
    if abs(one_day_change) < 0.01 and abs(one_hour_change) < 0.05:
        return "low_information_leader"
    current_price = _leader_current_price(leader)
    if current_price is not None and (current_price < 0.02 or current_price > 0.98):
        return "near_boundary_probability"
    return None


def _event_key_from_option_set_key(option_set_key: Any) -> tuple[str, str] | None:
    if option_set_key is None:
        return None
    parsed = option_set_key
    if isinstance(option_set_key, str):
        try:
            parsed = literal_eval(option_set_key)
        except (SyntaxError, ValueError):
            return None
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
        event = str(parsed[0] or "").strip()
        year = str(parsed[1] or "").strip()
        if event or year:
            return (event, year)
    return None


def _event_key_from_candidate(candidate: dict[str, Any]) -> tuple[str, str] | None:
    return _event_key_from_option_set_key(candidate.get("option_set_key"))


def _classify_relationship(
    leader: dict[str, Any], related: dict[str, Any], similarity: float
) -> tuple[str, str, str | None, str, str | None, bool, str | None]:
    leader_title = _title(leader)
    related_title = _title(related)
    leader_text = _normalize_text(_market_text(leader))
    related_text = _normalize_text(_market_text(related))
    leader_tokens = _market_tokens(leader)
    related_tokens = _market_tokens(related)
    overlap = leader_tokens & related_tokens

    # Check for same option_set_key (both winner markets with same event)
    leader_option_set_key = _extract_option_set_key(leader_title)
    related_option_set_key = _extract_option_set_key(related_title)
    
    if (
        leader_option_set_key 
        and related_option_set_key 
        and leader_option_set_key == related_option_set_key 
        and not _same_market(leader, related)
    ):
        subset_reason = _subset_relationship_reason(leader_title, related_title)
        if subset_reason:
            key_str = str(leader_option_set_key)
            return (
                REL_SAME_OPTION_SET_POSITIVE_CORRELATION, DIR_POSITIVE, None,
                f"same option set with subset/superset relationship: {subset_reason}",
                key_str, True, subset_reason,
            )

        key_str = str(leader_option_set_key)  # For JSON serialization
        LOGGER.warning(
            "classify_relationship same_option_set_key leader=%r related=%r key=%s",
            leader.get("title"),
            related.get("title"),
            key_str,
        )
        return (
            REL_SAME_OPTION_SET_NEGATIVE_CORRELATION, DIR_NEGATIVE, None,
            "same multi-outcome event: mutually exclusive competitors in the same option set",
            key_str, False, None,
        )

    # Cross-category or same-category market-type check (winner vs performer, or both same)
    if len(overlap) >= 2 and not _same_market(leader, related):
        leader_is_winner = _is_multi_outcome_title(leader_title)
        related_is_winner = _is_multi_outcome_title(related_title)
        leader_is_performer = _is_performer_type(leader_text)
        related_is_performer = _is_performer_type(related_text)

        if (leader_is_winner and related_is_performer) or (leader_is_performer and related_is_winner):
            return (
                REL_SHARED_EVENT_ONLY, DIR_UNCLEAR, None,
                "winner market paired with performer market; shared event keyword but different decision variables", None, False, None,
            )

        if leader_is_performer and related_is_performer:
            key = _option_set_key_for_pair(leader, related)
            return (
                REL_SAME_OPTION_SET_NEGATIVE_CORRELATION, DIR_NEGATIVE, None,
                "same option set: both markets select from the same performer pool", key, False, None,
            )

    # Cross-role political nominee check (presidential ↔ VP)
    if _is_president_vs_vice_president_nominee(leader_text, related_text):
        if _is_same_entity_cross_role(leader, related):
            return (
                REL_CROSS_ROLE_SAME_ENTITY, DIR_UNCLEAR, None,
                "same person appears in both presidential and VP nominee markets; exploratory cross-role relationship", None, False, None,
            )
        return (
            REL_SHARED_EVENT_ONLY, DIR_UNCLEAR, None,
            "presidential nominee and VP nominee markets share political context but different entities and roles", None, False, None,
        )

    shared_broad_event = _shared_broad_event(leader_text, related_text)
    if shared_broad_event and not _same_market(leader, related):
        return (
            REL_SAME_EVENT_OUTCOME, DIR_UNCLEAR, None,
            f"same broad event ({shared_broad_event}) but different decision variables", None, False, None,
        )

    leader_bucket_values = _numbers_and_dates(leader_title)
    related_bucket_values = _numbers_and_dates(related_title)
    if (
        similarity >= 0.45
        and leader_bucket_values
        and related_bucket_values
        and leader_bucket_values != related_bucket_values
        and (
            _has_any(leader_text, _BUCKET_TERMS)
            or _has_any(related_text, _BUCKET_TERMS)
            or "sentenced" in leader_text
            or "sentenced" in related_text
            or "sentence" in leader_text
            or "sentence" in related_text
        )
    ):
        return REL_DUPLICATE_BUCKET, DIR_UNCLEAR, None, "same question family with different numeric/date bucket", None, False, None

    template_name, template_direction, template_reason = _match_causal_template(leader_text, related_text)
    if template_name:
        if template_direction != DIR_UNCLEAR:
            return REL_CAUSALLY_LINKED, template_direction, template_name, template_reason or "", None, False, None
        return REL_CORRELATED_BUT_WEAK, DIR_UNCLEAR, template_name, template_reason or "", None, False, None

    if similarity >= 0.25 and overlap:
        return REL_CORRELATED_BUT_WEAK, DIR_UNCLEAR, None, "weak similarity only; no causal template matched", None, False, None

    return REL_UNRELATED, DIR_UNCLEAR, None, "insufficient causal or correlation evidence", None, False, None


def _is_president_vs_vice_president_nominee(left_text: str, right_text: str) -> bool:
    def is_presidential_nominee(text: str) -> bool:
        return bool(re.search(r"\b(president|presidential)\b", text)) and bool(re.search(r"\b(nominee|nomination)\b", text))

    def is_vp_nominee(text: str) -> bool:
        return bool(re.search(r"\b(vice president|vice presidential|vp)\b", text)) and bool(re.search(r"\b(nominee|nomination)\b", text))

    return (is_presidential_nominee(left_text) and is_vp_nominee(right_text)) or (
        is_presidential_nominee(right_text) and is_vp_nominee(left_text)
    )


def _match_causal_template(leader_text: str, related_text: str) -> tuple[str | None, str, str | None]:
    templates = (
        ("geo_ceasefire_hostage_release", {"ceasefire", "truce", "peace", "deal", "agreement"}, {"hostage", "hostages", "prisoner", "prisoners", "released", "release"}, DIR_POSITIVE),
        ("geo_ceasefire_sanction_imposition", {"ceasefire", "truce", "peace", "deal", "agreement"}, {"sanction", "sanctions"}, DIR_NEGATIVE),
        ("geo_ceasefire_withdrawal", {"ceasefire", "truce", "peace"}, {"withdraw", "withdrawal"}, DIR_POSITIVE),
        ("geo_ceasefire_territory_capture", {"ceasefire", "truce", "peace"}, {"capture", "control", "territory"}, DIR_NEGATIVE),
        ("geo_escalation_troop_involvement", {"invasion", "invade", "escalation", "attack"}, {"nato", "eu", "troop", "troops"}, DIR_POSITIVE),
        ("geo_sanctions_diplomatic_recognition", {"sanction", "sanctions"}, {"recognition", "recognize", "diplomatic"}, DIR_NEGATIVE),
        ("geo_election_called_leadership_transition", {"election", "called"}, {"resign", "transition", "leadership", "successor"}, DIR_POSITIVE),
        ("crypto_airdrop_token_launch", {"airdrop"}, {"token", "launch", "claim"}, DIR_POSITIVE),
        ("crypto_token_launch_fdv_market_cap", {"token", "launch"}, {"fdv", "market", "cap", "valuation"}, DIR_POSITIVE),
        ("crypto_exchange_listing_token_price", {"listing", "listed", "coinbase", "binance", "kraken"}, {"price", "market", "cap", "fdv"}, DIR_POSITIVE),
        ("crypto_protocol_exploit_tvl_token_impact", {"exploit", "hack", "hacked"}, {"tvl", "token", "price", "market", "cap"}, DIR_NEGATIVE),
        ("legal_conviction_sentencing_prison_time", {"conviction", "convicted", "guilty", "sentenced", "sentence"}, {"prison", "jail", "years", "sentenced", "sentence"}, DIR_POSITIVE),
        ("regulatory_approval_company_product_impact", {"approval", "approved", "approve"}, {"company", "product", "launch", "sales", "price", "stock", "etf"}, DIR_POSITIVE),
        ("regulatory_ban_investigation_company_product_impact", {"ban", "banned", "investigation", "investigate", "probe"}, {"company", "product", "sales", "price", "stock"}, DIR_NEGATIVE),
        ("rates_cut_mortgage_rate_impact", {"fed", "rate", "rates", "cut", "cuts"}, {"mortgage", "mortgages", "fall", "below"}, DIR_POSITIVE),
    )
    for name, left_terms, right_terms, direction in templates:
        if _has_any(leader_text, left_terms) and _has_any(related_text, right_terms):
            return name, direction, f"causal template matched: {name}"
        if _has_any(leader_text, right_terms) and _has_any(related_text, left_terms):
            return name, direction, f"causal template matched: {name}"
    return None, DIR_UNCLEAR, None


def _market_id(market: dict[str, Any]) -> str:
    return str(
        market.get("market_key")
        or market.get("market_id")
        or market.get("conditionId")
        or market.get("id")
        or market.get("slug")
        or ""
    )


def _same_market(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_id = _market_id(left)
    right_id = _market_id(right)
    if left_id and right_id and left_id == right_id:
        return True
    return bool(_normalize_text(left.get("title")) and _normalize_text(left.get("title")) == _normalize_text(right.get("title")))


def _outcome_bucket(market: dict[str, Any]) -> str:
    slug = _normalize_text(market.get("slug"))
    if slug:
        return slug
    return _normalize_text(market.get("title") or market.get("question"))


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _present(value: Any) -> Any:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    return value


def _first_present(market: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _present(market.get(key))
        if value is not None:
            return value
    return None


def _change(market: dict[str, Any], field: str) -> float:
    value = _to_float(market.get(field))
    if value is not None:
        return value
    if field == "one_day_price_change":
        value = _to_float(market.get("computed_mid_delta"))
        if value is not None:
            return value
    return 0.0


def _similarity_score(leader: dict[str, Any], related: dict[str, Any]) -> float:
    leader_tokens = _market_tokens(leader)
    related_tokens = _market_tokens(related)
    token_score = 0.0
    if leader_tokens and related_tokens:
        overlap = leader_tokens & related_tokens
        if overlap:
            union = leader_tokens | related_tokens
            jaccard = len(overlap) / len(union) if union else 0.0
            containment = len(overlap) / min(len(leader_tokens), len(related_tokens))
            token_score = max(jaccard, containment * 0.6)
    embedding_score = min(0.30, _hashed_text_embedding_similarity(_market_text(leader), _market_text(related)))
    score = max(token_score, embedding_score)
    return round(min(1.0, score), 4)


def _hashed_text_embedding(text: str | None, dimensions: int = 64) -> list[float]:
    normalized = _normalize_text(text)
    vector = [0.0] * dimensions
    if not normalized:
        return vector
    tokens = normalized.split()
    features = tokens[:]
    compact = " ".join(tokens)
    features.extend(compact[idx : idx + 4] for idx in range(max(0, len(compact) - 3)))
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=4).digest()
        bucket = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[bucket] += sign
    return vector


def _hashed_text_embedding_similarity(left: str | None, right: str | None) -> float:
    left_vec = _hashed_text_embedding(left)
    right_vec = _hashed_text_embedding(right)
    dot = sum(a * b for a, b in zip(left_vec, right_vec))
    left_norm = sum(a * a for a in left_vec) ** 0.5
    right_norm = sum(b * b for b in right_vec) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm))


def _divergence_score(
    similarity_score: float,
    leader_1h: float,
    related_1h: float,
    leader_24h: float,
    related_24h: float,
    expected_direction: str = DIR_POSITIVE,
) -> float:
    direction_multiplier = -1.0 if expected_direction == DIR_NEGATIVE else 1.0
    expected_1h = leader_1h * similarity_score * direction_multiplier
    expected_24h = leader_24h * similarity_score * direction_multiplier
    gap_1h = max(0.0, abs(expected_1h) - (related_1h * (1.0 if expected_1h >= 0 else -1.0)))
    gap_24h = max(0.0, abs(expected_24h) - (related_24h * (1.0 if expected_24h >= 0 else -1.0)))
    return round(max(gap_1h, gap_24h), 6)


def _suggested_trade_direction(leader_1h: float, leader_24h: float, expected_direction: str = DIR_POSITIVE) -> str:
    if expected_direction == DIR_UNCLEAR:
        return "review_only"
    leader_move = leader_24h if abs(leader_24h) >= abs(leader_1h) else leader_1h
    if expected_direction == DIR_NEGATIVE:
        leader_move = -leader_move
    if leader_move > 0:
        return "buy_yes_related"
    if leader_move < 0:
        return "buy_no_related"
    return "hold"


def _candidate_payload(
    leader: dict[str, Any],
    related: dict[str, Any],
    similarity: float,
    divergence: float,
    leader_1h: float,
    related_1h: float,
    leader_24h: float,
    related_24h: float,
    relationship_type: str,
    expected_direction: str,
    causal_template_matched: str | None,
    causal_link_reason: str | None,
    option_set_key: str | None = None,
    subset_relationship: bool = False,
    subset_relationship_reason: str | None = None,
    min_related_abs_move_24h: float = DEFAULT_MIN_RELATED_ABS_MOVE_24H,
    min_similarity_for_trade: float = DEFAULT_MIN_SIMILARITY_FOR_TRADE,
) -> dict[str, Any]:
    strong_match = (
        relationship_type in {
            REL_CAUSALLY_LINKED,
            REL_SAME_OPTION_SET_NEGATIVE_CORRELATION,
            REL_SAME_OPTION_SET_POSITIVE_CORRELATION,
        }
        and expected_direction != DIR_UNCLEAR
        and similarity >= STRONG_MIN_SIMILARITY
        and divergence >= STRONG_MIN_DIVERGENCE
    )
    exploratory_match = similarity >= DEFAULT_MIN_SIMILARITY and not strong_match
    low_related_activity = abs(related_24h) < max(0.0, float(min_related_abs_move_24h))
    below_similarity_threshold = similarity < max(0.0, float(min_similarity_for_trade))
    tradable_signal = strong_match and not low_related_activity and not below_similarity_threshold

    suggested = _suggested_trade_direction(leader_1h, leader_24h, expected_direction)
    if not tradable_signal:
        suggested = "review_only"
    # For same-option-set, leader falling → buy_yes is uncertain (option set may not be exhaustive)
    # Cross-role same-entity is always review_only until a directional rule is configured
    if relationship_type == REL_CROSS_ROLE_SAME_ENTITY:
        suggested = "review_only"
    risk = compute_execution_risk_details(
        related,
        venue=str(related.get("venue") or related.get("platform") or "polymarket"),
        divergence_stress=min(1.0, float(divergence or 0.0)),
    )
    liquidity_score = min(1.0, float(related.get("depth_top5") or related.get("liquidity_proxy") or 0.0) / 100_000.0)
    execution_risk_penalty = float(risk["risk_score"])
    trade_rank_score = round(max(0.0, float(divergence or 0.0) - (0.50 * execution_risk_penalty) + (0.20 * liquidity_score)), 6)
    trade_bucket = "primary" if tradable_signal and risk["risk_label"] != "HIGH" else "review"

    return {
        "leader_market_title": leader.get("title"),
        "leader_market_id": _market_id(leader),
        "related_market_title": related.get("title") or related.get("question"),
        "related_market_id": _market_id(related),
        "similarity_score": similarity,
        "relationship_type": relationship_type,
        "expected_correlation_direction": expected_direction,
        "causal_template_matched": causal_template_matched,
        "causal_link_reason": causal_link_reason,
        "option_set_key": option_set_key,
        "subset_relationship": subset_relationship,
        "subset_relationship_reason": subset_relationship_reason,
        "resolved_or_bad_tick_candidate": False,
        "low_related_activity": low_related_activity,
        "below_similarity_threshold": below_similarity_threshold,
        "tradable_signal": tradable_signal,
        "leader_price_change_1h": leader_1h,
        "related_price_change_1h": related_1h,
        "leader_price_change_24h": leader_24h,
        "related_price_change_24h": related_24h,
        "divergence_score": divergence,
        "execution_risk_score": risk["risk_score"],
        "execution_risk_label": risk["risk_label"],
        "execution_risk_drivers": risk["drivers"],
        "liquidity_score": round(liquidity_score, 6),
        "trade_rank_score": trade_rank_score,
        "trade_bucket": trade_bucket,
        "suggested_trade_direction": suggested,
        "exploratory_match": exploratory_match,
        "strong_match": strong_match,
        "explanation": (
            f"Leader moved {leader_1h:+.4f} over 1h and {leader_24h:+.4f} over 24h; "
            f"related market moved {related_1h:+.4f} over 1h and {related_24h:+.4f} over 24h "
            f"with {relationship_type} / {expected_direction} relationship and similarity {similarity:.4f}."
        ),
    }


def _leader_candidate_evaluations(
    leader: dict[str, Any],
    polymarket_universe: list[dict[str, Any]],
    *,
    min_similarity: float,
    min_divergence: float,
    max_universe: int,
    max_candidates_per_leader: int | None = None,
    exploratory: bool = False,
    include_weak_similarity: bool = False,
    min_related_abs_move_24h: float = DEFAULT_MIN_RELATED_ABS_MOVE_24H,
    min_similarity_for_trade: float = DEFAULT_MIN_SIMILARITY_FOR_TRADE,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    scoring_start = time.monotonic()
    leader_title = _title(leader)
    leader_1h = _change(leader, "one_hour_price_change")
    leader_24h = _change(leader, "one_day_price_change")
    leader_has_move = leader_1h != 0.0 or leader_24h != 0.0
    
    # ===== SAFETY CHECK: If leader looks like a winner market but extraction fails, skip it =====
    leader_option_set_type, leader_option_set_key = _leader_option_set_type(leader)
    
    # If it looks like a winner market but we couldn't extract a key, skip instead of doing global search
    if leader_option_set_type == "winner" and leader_option_set_key is None:
        LOGGER.warning(
            "leader_candidate_evaluations SKIP winner market with failed extraction title=%r",
            leader_title[:80],
        )
        return []
    
    seen_buckets: set[str] = set()
    evaluations: list[dict[str, Any]] = []
    universe = _prefilter_universe(leader, polymarket_universe, max_universe=max_universe)
    prefilter_done = time.monotonic()

    for related in universe:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if max_candidates_per_leader is not None and len(evaluations) >= max_candidates_per_leader:
            break
        is_self_market = _same_market(leader, related)
        bucket = _outcome_bucket(related)
        duplicate_bucket = bool(bucket and bucket in seen_buckets and not is_self_market)
        if bucket and not is_self_market and not duplicate_bucket:
            seen_buckets.add(bucket)

        similarity = _similarity_score(leader, related)
        (
            relationship_type,
            expected_direction,
            causal_template_matched,
            relationship_reason,
            option_set_key,
            subset_relationship,
            subset_relationship_reason,
        ) = _classify_relationship(leader, related, similarity)
        related_1h = _change(related, "one_hour_price_change")
        related_24h = _change(related, "one_day_price_change")
        if abs(leader_24h) <= NEAR_ZERO_MOVE and abs(related_24h) <= NEAR_ZERO_MOVE:
            continue
        divergence = _divergence_score(
            similarity,
            leader_1h,
            related_1h,
            leader_24h,
            related_24h,
            expected_direction,
        )

        rejection_reasons: list[str] = []
        if not leader_has_move:
            rejection_reasons.append("leader has no 1h or 24h move")
        if max(abs(leader_1h), abs(leader_24h)) < MIN_LEADER_MOVE:
            rejection_reasons.append(f"leader move below threshold {MIN_LEADER_MOVE:.4f}")
        if is_self_market:
            rejection_reasons.append("self-market")
        if duplicate_bucket:
            rejection_reasons.append("duplicate outcome bucket")
        
        # Strict: Always reject invalid relationship types
        if relationship_type in {REL_SAME_EVENT_OUTCOME, REL_DUPLICATE_BUCKET, REL_UNRELATED, REL_SHARED_EVENT_ONLY}:
            rejection_reasons.append(f"always_invalid: {relationship_type}")
        
        # CORRELATED_BUT_WEAK: Only allowed if include_weak_similarity=true (exploratory does NOT override)
        if relationship_type == REL_CORRELATED_BUT_WEAK:
            if not include_weak_similarity:
                rejection_reasons.append("weak similarity rejected (correlated_but_weak requires include_weak_similarity=true)")
        
        # Expected direction must be clear (except for allowed unclear types)
        if expected_direction == DIR_UNCLEAR:
            unclear_allowed = relationship_type == REL_CROSS_ROLE_SAME_ENTITY or (
                include_weak_similarity and relationship_type == REL_CORRELATED_BUT_WEAK
            )
            if not unclear_allowed:
                rejection_reasons.append("expected correlation direction unclear")
        
        if similarity < min_similarity:
            rejection_reasons.append(f"similarity below threshold {min_similarity:.4f}")
        if divergence < min_divergence:
            rejection_reasons.append(f"divergence below threshold {min_divergence:.4f}")

        evaluations.append(
            {
                **_candidate_payload(
                    leader,
                    related,
                    similarity,
                    divergence,
                    leader_1h,
                    related_1h,
                    leader_24h,
                    related_24h,
                    relationship_type,
                    expected_direction,
                    causal_template_matched,
                    relationship_reason,
                    option_set_key,
                    subset_relationship,
                    subset_relationship_reason,
                    min_related_abs_move_24h,
                    min_similarity_for_trade,
                ),
                "excluded_as_self_market": is_self_market,
                "excluded_as_duplicate_bucket": duplicate_bucket,
                "excluded_from_normal_output": bool(rejection_reasons),
                "relationship_reason": relationship_reason,
                "rejected_reason": "; ".join(rejection_reasons) if rejection_reasons else None,
            }
        )

    evaluations.sort(key=lambda item: (item["similarity_score"], item["divergence_score"]), reverse=True)
    LOGGER.warning(
        "lag candidate similarity scoring leader=%r prefiltered=%d evaluated=%d prefilter_s=%.4f scoring_s=%.4f",
        leader.get("title"),
        len(universe),
        len(evaluations),
        prefilter_done - scoring_start,
        time.monotonic() - prefilter_done,
    )
    return evaluations


def _prefilter_universe(
    leader: dict[str, Any],
    polymarket_universe: list[dict[str, Any]],
    *,
    max_universe: int,
) -> list[dict[str, Any]]:
    """Prefilter universe by token overlap and optional option_set_key matching.
    
    If the leader has an option_set_key (winner/performer markets), it is considered multi-outcome.
    In that case, only compare against markets with the same option_set_key to avoid
    cross-category comparisons like "World Cup winner" → "halftime performer".
    """
    leader_tokens = _market_tokens(leader)
    leader_option_set_type, leader_option_set_key = _leader_option_set_type(leader)
    
    scored: list[tuple[int, int, bool, dict[str, Any]]] = []
    
    # If leader has option_set_key constraint, filter by matching option_set_key
    if leader_option_set_key:
        LOGGER.warning(
            "lag candidate prefilter leader=%r has option_set_key=%r type=%s prioritizing same set",
            leader.get("title"),
            leader_option_set_key,
            leader_option_set_type,
        )
        for market in polymarket_universe:
            market_option_set_type, market_option_set_key = _leader_option_set_type(market)
            market_tokens = _market_tokens(market)
            overlap_count = len(leader_tokens & market_tokens)
            is_self = _same_market(leader, market)
            same_option_set = market_option_set_key == leader_option_set_key
            if overlap_count <= 0 and not is_self:
                continue
            scored.append((1 if same_option_set else 0, overlap_count, is_self, market))
        LOGGER.warning(
            "lag candidate prefilter option_set_key=%r matched=%d from universe=%d",
            leader_option_set_key,
            len(scored),
            len(polymarket_universe),
        )
    else:
        # No option_set constraint; normal token-based prefiltering
        if not leader_tokens:
            return polymarket_universe[: max(1, int(max_universe))]
        for market in polymarket_universe:
            market_tokens = _market_tokens(market)
            overlap_count = len(leader_tokens & market_tokens)
            is_self = _same_market(leader, market)
            if overlap_count <= 0 and not is_self:
                continue
            scored.append((0, overlap_count, is_self, market))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    capped = [market for _same_option_set, _overlap, _is_self, market in scored[: max(1, int(max_universe))]]
    LOGGER.warning(
        "lag candidate prefilter leader=%r leader_option_set_key=%r universe=%d matched=%d capped=%d max_universe=%d",
        leader.get("title"),
        leader_option_set_key,
        len(polymarket_universe),
        len(scored),
        len(capped),
        max_universe,
    )
    return capped


def normalize_polymarket_universe(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        if market.get("venue") == "polymarket" and market.get("market_key"):
            normalized.append(dict(market))
        else:
            normalized.append(normalize_polymarket_market(market))
    return normalized


def _normalize_parquet_market(row: dict[str, Any]) -> dict[str, Any]:
    market_id = _first_present(row, ("market_id", "market_key", "conditionid", "conditionId", "id"))
    title = _first_present(row, ("market_title", "question", "title", "event_title"))
    best_bid = _to_float(_first_present(row, ("bestbid", "bestBid", "best_yes_bid")))
    best_ask = _to_float(_first_present(row, ("bestask", "bestAsk", "best_yes_ask")))
    current_price = _to_float(_first_present(row, ("current_price", "lasttradeprice", "lastTradePrice", "mid")))
    if current_price is None and best_bid is not None and best_ask is not None:
        current_price = (best_bid + best_ask) / 2.0

    spread = _to_float(_first_present(row, ("spread", "spread_reported")))
    if spread is None and best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid

    normalized = {
        "venue": "polymarket",
        "market_id": str(market_id) if market_id is not None else "",
        "market_key": str(market_id) if market_id is not None else "",
        "market_title": str(title) if title is not None else None,
        "title": str(title) if title is not None else None,
        "question": _first_present(row, ("question", "market_title", "title", "event_title")),
        "slug": _first_present(row, ("slug", "event_slug")),
        "current_price": current_price,
        "mid": current_price,
        "spread": spread,
        "spread_reported": spread,
        "volume_proxy": _to_float(_first_present(row, ("volume24hr", "volume24hrclob", "volumenum", "volume"))),
        "liquidity_proxy": _to_float(_first_present(row, ("liquiditynum", "liquidity", "liquidityclob"))) or 0.0,
        "one_hour_price_change": _to_float(_first_present(row, ("price_change_1h", "onehourpricechange", "oneHourPriceChange"))),
        "one_day_price_change": _to_float(_first_present(row, ("price_change_24h", "onedaypricechange", "oneDayPriceChange"))),
        "last_trade_price": _to_float(_first_present(row, ("lasttradeprice", "lastTradePrice"))),
    }
    normalized["_lag_tokens"] = _tokens(_market_text(normalized))
    return normalized


def _vardr1_markets_path() -> Path:
    return Path(os.getenv("VARDR1_MARKETS_PATH", DEFAULT_VARDR1_MARKETS_PATH).strip())


@lru_cache(maxsize=4)
def _load_vardr1_parquet_universe_cached(path_str: str, mtime_ns: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(path_str)
    try:
        import pandas as pd

        desired_columns = {
            "active",
            "closed",
            "archived",
            "market_id",
            "market_key",
            "conditionid",
            "conditionId",
            "id",
            "market_title",
            "question",
            "title",
            "event_title",
            "slug",
            "event_slug",
            "current_price",
            "lasttradeprice",
            "lastTradePrice",
            "mid",
            "bestbid",
            "bestBid",
            "best_yes_bid",
            "bestask",
            "bestAsk",
            "best_yes_ask",
            "spread",
            "spread_reported",
            "volume24hr",
            "volume24hrclob",
            "volumenum",
            "volume",
            "liquiditynum",
            "liquidity",
            "liquidityclob",
            "price_change_1h",
            "onehourpricechange",
            "oneHourPriceChange",
            "price_change_24h",
            "onedaypricechange",
            "oneDayPriceChange",
        }
        try:
            import pyarrow.parquet as pq

            available_columns = set(pq.read_schema(path).names)
            columns = sorted(desired_columns & available_columns)
        except Exception:
            columns = None

        df = pd.read_parquet(path, columns=columns)
        raw_rows = len(df)
        if {"active", "closed"}.issubset(df.columns):
            df = df[(df["active"] == True) & (df["closed"] != True)]  # noqa: E712
        if "archived" in df.columns:
            df = df[df["archived"] != True]  # noqa: E712
        rows = df.to_dict(orient="records")
        normalized = [
            market
            for row in rows
            if (market := _normalize_parquet_market(row)).get("market_key") and market.get("title")
        ]
        return normalized, {
            "source": "vardr1_parquet",
            "source_path": str(path),
            "raw_row_count": raw_rows,
            "row_count": len(normalized),
        }
    except Exception as exc:
        LOGGER.warning("failed to load Vardr-1 Polymarket parquet universe from %s: %s", path, exc)
        return [], {
            "source": "vardr1_parquet_error",
            "source_path": str(path),
            "raw_row_count": 0,
            "row_count": 0,
        }


def _load_vardr1_parquet_universe(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return _load_vardr1_parquet_universe_cached(str(path), stat.st_mtime_ns)


def _load_csv_universe(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(path)
        rows = df.to_dict(orient="records")
        normalized = [
            market
            for row in rows
            if (market := _normalize_parquet_market(row)).get("market_key") and market.get("title")
        ]
        return normalized, {
            "source": "csv",
            "source_path": str(path),
            "raw_row_count": len(rows),
            "row_count": len(normalized),
        }
    except Exception as exc:
        LOGGER.warning("failed to load CSV Polymarket universe from %s: %s", path, exc)
        return None


def load_local_polymarket_universe_with_metadata(root_dir: str | Path = ".") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parquet_path = _vardr1_markets_path()
    parquet_result = _load_vardr1_parquet_universe(parquet_path)
    if parquet_result is not None:
        return parquet_result

    root = Path(root_dir)
    csv_candidates = [
        Path(os.getenv("VARDR_MARKETS_CSV_PATH", "")).expanduser() if os.getenv("VARDR_MARKETS_CSV_PATH") else None,
        root / "data" / "raw" / "polymarket_markets.csv",
        root / "data" / "polymarket_markets.csv",
    ]
    for csv_path in csv_candidates:
        if csv_path is None:
            continue
        csv_result = _load_csv_universe(csv_path)
        if csv_result is not None:
            return csv_result

    latest_snapshot = root / "data" / "latest_snapshot.json"
    if latest_snapshot.exists():
        try:
            payload = json.loads(latest_snapshot.read_text(encoding="utf-8"))
            snapshots = payload.get("polymarket_snapshots") if isinstance(payload, dict) else None
            if isinstance(snapshots, list):
                markets = normalize_polymarket_universe([item for item in snapshots if isinstance(item, dict)])
                return markets, {
                    "source": "latest_snapshot",
                    "source_path": str(latest_snapshot),
                    "raw_row_count": len(snapshots),
                    "row_count": len(markets),
                }
        except Exception as exc:
            LOGGER.warning("failed to load local Polymarket snapshot universe from %s: %s", latest_snapshot, exc)

    fixture_path = root / "fixtures" / "polymarket_markets_open.json"
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("failed to load local Polymarket fixture universe from %s: %s", fixture_path, exc)
        return [], {
            "source": "none",
            "source_path": str(parquet_path),
            "raw_row_count": 0,
            "row_count": 0,
        }

    if isinstance(payload, dict):
        payload = payload.get("markets", [])
    if not isinstance(payload, list):
        return [], {
            "source": "fixture",
            "source_path": str(fixture_path),
            "raw_row_count": 0,
            "row_count": 0,
        }
    markets = normalize_polymarket_universe([item for item in payload if isinstance(item, dict)])
    return markets, {
        "source": "fixture",
        "source_path": str(fixture_path),
        "raw_row_count": len(payload),
        "row_count": len(markets),
    }


def load_local_polymarket_universe(root_dir: str | Path = ".") -> list[dict[str, Any]]:
    markets, _metadata = load_local_polymarket_universe_with_metadata(root_dir)
    return markets


def build_lag_candidates_with_metadata(
    leader_markets: list[dict[str, Any]],
    polymarket_universe: list[dict[str, Any]],
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_divergence: float = DEFAULT_MIN_DIVERGENCE,
    limit: int = 50,
    max_leaders_evaluated: int = DEFAULT_MAX_LEADERS_EVALUATED,
    min_valid_candidates: int = DEFAULT_MIN_VALID_CANDIDATES,
    max_universe: int = 5000,
    max_candidates_per_leader: int = 25,
    max_candidates_per_leader_output: int = DEFAULT_MAX_CANDIDATES_PER_LEADER_OUTPUT,
    max_candidates_per_event_output: int = DEFAULT_MAX_CANDIDATES_PER_EVENT_OUTPUT,
    include_review_only: bool = False,
    max_abs_leader_move: float = DEFAULT_MAX_ABS_LEADER_MOVE,
    exploratory: bool = False,
    include_weak_similarity: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    min_leaders_before_timeout: int = 5,
    min_related_abs_move_24h: float = DEFAULT_MIN_RELATED_ABS_MOVE_24H,
    min_similarity_for_trade: float = DEFAULT_MIN_SIMILARITY_FOR_TRADE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate leaders sequentially, skipping those with no valid lag candidates.

    Valid candidates are those with:
    - No rejection reasons (passed all thresholds and filters)
    - Relationship type in STRICTLY_VALID_RELATIONSHIPS, or optionally
      CROSS_ROLE_SAME_ENTITY if exploratory=true

    Returns (candidates, metadata) where metadata describes how many leaders were
    fetched, evaluated, skipped, and why.
    """
    candidates: list[dict[str, Any]] = []
    leaders_evaluated = 0
    leaders_skipped = 0
    resolved_or_bad_tick_leaders_skipped = 0
    resolved_or_bad_tick_skip_examples: list[dict[str, Any]] = []
    candidates_suppressed_by_leader_cap = 0
    candidates_suppressed_by_event_cap = 0
    review_only_candidates_suppressed = 0
    leaders_filtered_before_evaluation = 0
    leader_prefilter_skip_reasons_count: dict[str, int] = {}
    event_output_counts: dict[tuple[str, str], int] = {}
    skip_reasons: dict[str, int] = {}
    partial_timeout = False
    started_at = time.monotonic()
    deadline = started_at + max(0.1, float(timeout_seconds))
    min_leaders_before_timeout = max(0, int(min_leaders_before_timeout))
    minimum_leaders_to_scan = min(
        len(leader_markets),
        max(1, int(max_leaders_evaluated)),
        min_leaders_before_timeout,
    )
    leaders_evaluated_before_timeout = 0

    for leader in leader_markets[: max(1, int(max_leaders_evaluated))]:
        if time.monotonic() >= deadline and leaders_evaluated >= minimum_leaders_to_scan:
            partial_timeout = True
            leaders_evaluated_before_timeout = leaders_evaluated
            break
        leaders_evaluated += 1
        enforce_deadline_for_leader = leaders_evaluated > min_leaders_before_timeout

        prefilter_reason = _leader_prefilter_skip_reason(leader, max_abs_leader_move=max_abs_leader_move)
        if prefilter_reason is not None:
            leaders_skipped += 1
            leaders_filtered_before_evaluation += 1
            skip_reasons[prefilter_reason] = skip_reasons.get(prefilter_reason, 0) + 1
            leader_prefilter_skip_reasons_count[prefilter_reason] = (
                leader_prefilter_skip_reasons_count.get(prefilter_reason, 0) + 1
            )
            if prefilter_reason == "resolved_or_bad_tick_candidate":
                resolved_or_bad_tick_leaders_skipped += 1
            if prefilter_reason == "resolved_or_bad_tick_candidate" and len(resolved_or_bad_tick_skip_examples) < 5:
                resolved_or_bad_tick_skip_examples.append(
                    {
                        "leader_market_title": leader.get("title"),
                        "leader_market_id": _market_id(leader),
                        "leader_price_change_1h": _change(leader, "one_hour_price_change"),
                        "leader_price_change_24h": _change(leader, "one_day_price_change"),
                    }
                )
            LOGGER.warning(
                "lag_candidates skip prefiltered leader=%r reason=%s 1h=%s 24h=%s current_price=%s",
                leader.get("title"),
                prefilter_reason,
                _change(leader, "one_hour_price_change"),
                _change(leader, "one_day_price_change"),
                _leader_current_price(leader),
            )
            continue

        evaluations = _leader_candidate_evaluations(
            leader,
            polymarket_universe,
            min_similarity=min_similarity,
            min_divergence=min_divergence,
            max_universe=max_universe,
            exploratory=exploratory,
            include_weak_similarity=include_weak_similarity,
            min_related_abs_move_24h=min_related_abs_move_24h,
            min_similarity_for_trade=min_similarity_for_trade,
            deadline=deadline if enforce_deadline_for_leader else None,
        )

        # Count only VALID candidates for leader skipping decision
        # Valid = no rejection reasons AND relationship type is acceptable
        valid_evaluations = []
        for e in evaluations:
            if e["rejected_reason"] is not None:
                continue
            rel_type = e["relationship_type"]
            if not _is_valid_relationship_type(
                rel_type,
                exploratory=exploratory,
                include_weak_similarity=include_weak_similarity,
            ):
                continue
            valid_evaluations.append(e)

        if len(valid_evaluations) < min_valid_candidates:
            leaders_skipped += 1
            reason = "no_valid_candidates"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            LOGGER.warning(
                "lag_candidates skip leader=%r valid=%d required=%d",
                leader.get("title"),
                len(valid_evaluations),
                min_valid_candidates,
            )
            continue

        eligible_for_output = [
            evaluation
            for evaluation in valid_evaluations
            if include_review_only or evaluation.get("tradable_signal", False)
        ]
        review_only_candidates_suppressed += sum(
            1
            for evaluation in valid_evaluations
            if not evaluation.get("tradable_signal", False) and not include_review_only
        )
        leader_output_limit = max(1, int(max_candidates_per_leader_output))
        event_output_limit = max(1, int(max_candidates_per_event_output))
        added_for_leader = 0
        base_leader_output_limit = min(2, leader_output_limit)
        for evaluation in eligible_for_output:
            if added_for_leader >= leader_output_limit:
                candidates_suppressed_by_leader_cap += 1
                continue
            if added_for_leader >= base_leader_output_limit and not evaluation.get("tradable_signal", False):
                candidates_suppressed_by_leader_cap += 1
                continue
            event_key = _event_key_from_candidate(evaluation)
            if event_key is not None and event_output_counts.get(event_key, 0) >= event_output_limit:
                candidates_suppressed_by_event_cap += 1
                continue
            candidates.append(
                {
                    key: value
                    for key, value in evaluation.items()
                    if key not in {"excluded_as_self_market", "excluded_as_duplicate_bucket", "rejected_reason"}
                }
            )
            added_for_leader += 1
            if event_key is not None:
                event_output_counts[event_key] = event_output_counts.get(event_key, 0) + 1

        if len(candidates) >= limit and leaders_evaluated >= minimum_leaders_to_scan:
            break

    candidates.sort(key=lambda item: item.get("trade_rank_score", item["divergence_score"]), reverse=True)
    candidates = candidates[: max(1, int(limit))]
    output_event_keys = {
        event_key
        for candidate in candidates
        for event_key in [_event_key_from_candidate(candidate)]
        if event_key is not None
    }
    output_leader_ids = {
        candidate.get("leader_market_id")
        for candidate in candidates
        if candidate.get("leader_market_id") is not None
    }
    elapsed_time_total = time.monotonic() - started_at
    if partial_timeout and leaders_evaluated_before_timeout == 0:
        leaders_evaluated_before_timeout = leaders_evaluated

    metadata: dict[str, Any] = {
        "leaders_fetched": len(leader_markets),
        "leaders_evaluated": leaders_evaluated,
        "remaining_leaders_not_evaluated": max(0, len(leader_markets) - leaders_evaluated),
        "leaders_skipped": leaders_skipped,
        "skip_reasons_count": skip_reasons,
        "valid_candidates_found": len(candidates),
        "review_only_candidates_suppressed": review_only_candidates_suppressed,
        "non_tradable_filtered_count": review_only_candidates_suppressed,
        "candidates_suppressed_by_leader_cap": candidates_suppressed_by_leader_cap,
        "candidates_suppressed_by_event_cap": candidates_suppressed_by_event_cap,
        "max_candidates_per_leader_output": max_candidates_per_leader_output,
        "max_candidates_per_event_output": max_candidates_per_event_output,
        "max_abs_leader_move": max_abs_leader_move,
        "include_review_only": include_review_only,
        "resolved_or_bad_tick_leaders_skipped": resolved_or_bad_tick_leaders_skipped,
        "resolved_or_bad_tick_skip_examples": resolved_or_bad_tick_skip_examples,
        "leaders_filtered_before_evaluation": leaders_filtered_before_evaluation,
        "leader_prefilter_skip_reasons_count": leader_prefilter_skip_reasons_count,
        "events_represented_count": len(output_event_keys),
        "unique_leaders_in_output": len(output_leader_ids),
        "tradable_signals_count": sum(1 for candidate in candidates if candidate.get("tradable_signal", False)),
        "partial_results_due_to_timeout": partial_timeout,
        "elapsed_time_total": elapsed_time_total,
        "leaders_evaluated_before_timeout": leaders_evaluated_before_timeout,
        "min_leaders_before_timeout": min_leaders_before_timeout,
        "min_related_abs_move_24h": min_related_abs_move_24h,
        "min_similarity_for_trade": min_similarity_for_trade,
    }
    return candidates, metadata


def build_lag_candidates(
    leader_markets: list[dict[str, Any]],
    polymarket_universe: list[dict[str, Any]],
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_divergence: float = DEFAULT_MIN_DIVERGENCE,
    limit: int = 50,
    max_leaders: int = 3,
    max_leaders_evaluated: int | None = None,
    min_valid_candidates: int = DEFAULT_MIN_VALID_CANDIDATES,
    max_universe: int = 5000,
    max_candidates_per_leader: int = 25,
    max_candidates_per_leader_output: int = DEFAULT_MAX_CANDIDATES_PER_LEADER_OUTPUT,
    max_candidates_per_event_output: int = DEFAULT_MAX_CANDIDATES_PER_EVENT_OUTPUT,
    include_review_only: bool = False,
    max_abs_leader_move: float = DEFAULT_MAX_ABS_LEADER_MOVE,
    exploratory: bool = False,
    include_weak_similarity: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    min_leaders_before_timeout: int = 5,
    min_related_abs_move_24h: float = DEFAULT_MIN_RELATED_ABS_MOVE_24H,
    min_similarity_for_trade: float = DEFAULT_MIN_SIMILARITY_FOR_TRADE,
) -> list[dict[str, Any]]:
    """Return lag candidates. Skips leaders that produce no valid candidates and
    continues evaluating the next leader until the limit is reached or timeout fires.

    max_leaders_evaluated overrides max_leaders when provided.
    """
    effective_max = max_leaders_evaluated if max_leaders_evaluated is not None else max_leaders
    candidates, _ = build_lag_candidates_with_metadata(
        leader_markets=leader_markets,
        polymarket_universe=polymarket_universe,
        min_similarity=min_similarity,
        min_divergence=min_divergence,
        limit=limit,
        max_leaders_evaluated=effective_max,
        min_valid_candidates=min_valid_candidates,
        max_universe=max_universe,
        max_candidates_per_leader=max_candidates_per_leader,
        max_candidates_per_leader_output=max_candidates_per_leader_output,
        max_candidates_per_event_output=max_candidates_per_event_output,
        include_review_only=include_review_only,
        max_abs_leader_move=max_abs_leader_move,
        exploratory=exploratory,
        include_weak_similarity=include_weak_similarity,
        timeout_seconds=timeout_seconds,
        min_leaders_before_timeout=min_leaders_before_timeout,
        min_related_abs_move_24h=min_related_abs_move_24h,
        min_similarity_for_trade=min_similarity_for_trade,
    )
    return candidates


def build_lag_candidates_debug(
    leader_markets: list[dict[str, Any]],
    polymarket_universe: list[dict[str, Any]],
    *,
    leader_limit: int = 5,
    top_candidates: int = 10,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_divergence: float = DEFAULT_MIN_DIVERGENCE,
    min_valid_candidates: int = DEFAULT_MIN_VALID_CANDIDATES,
    max_universe: int = 5000,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    exploratory: bool = False,
    include_weak_similarity: bool = False,
    universe_source_path: str | None = None,
    universe_row_count: int | None = None,
    min_leaders_before_timeout: int = 5,
    max_abs_leader_move: float = DEFAULT_MAX_ABS_LEADER_MOVE,
    min_related_abs_move_24h: float = DEFAULT_MIN_RELATED_ABS_MOVE_24H,
    min_similarity_for_trade: float = DEFAULT_MIN_SIMILARITY_FOR_TRADE,
) -> list[dict[str, Any]]:
    debug_rows: list[dict[str, Any]] = []
    started_at = time.monotonic()
    deadline = started_at + max(0.1, float(timeout_seconds))
    min_leaders_before_timeout = max(0, int(min_leaders_before_timeout))
    leaders_evaluated = 0
    leaders_evaluated_before_timeout = 0
    partial_timeout = False
    for leader in leader_markets[: max(1, int(leader_limit))]:
        if time.monotonic() >= deadline and leaders_evaluated >= min_leaders_before_timeout:
            partial_timeout = True
            leaders_evaluated_before_timeout = leaders_evaluated
            break
        leaders_evaluated += 1
        enforce_deadline_for_leader = leaders_evaluated > min_leaders_before_timeout
        option_set_type, leader_option_set_key = _leader_option_set_type(leader)
        prefilter_reason = _leader_prefilter_skip_reason(leader, max_abs_leader_move=max_abs_leader_move)
        if prefilter_reason is not None:
            debug_rows.append(
                {
                    "leader_market_title": leader.get("title"),
                    "option_set_type": option_set_type,
                    "leader_option_set_key": leader_option_set_key,
                    "leader_skipped": True,
                    "skip_reason": prefilter_reason,
                    "leader_prefilter_skip_reason": prefilter_reason,
                    "resolved_or_bad_tick_candidate": prefilter_reason == "resolved_or_bad_tick_candidate",
                    "leader_price_change_1h": _change(leader, "one_hour_price_change"),
                    "leader_price_change_24h": _change(leader, "one_day_price_change"),
                    "leader_current_price": _leader_current_price(leader),
                    "valid_candidate_count": 0,
                    "rejected_candidate_count": 0,
                    "top_rejection_reasons": [],
                    "universe_source_path": universe_source_path,
                    "universe_row_count": universe_row_count if universe_row_count is not None else len(polymarket_universe),
                    "number_of_local_universe_markets_searched": len(polymarket_universe),
                    "prefiltered_universe_count": 0,
                    "max_universe": max_universe,
                    "best_valid_candidates": [],
                    "top_related_candidates_before_threshold_filtering": [],
                }
            )
            continue
        prefiltered_universe = _prefilter_universe(leader, polymarket_universe, max_universe=max_universe)
        evaluations = _leader_candidate_evaluations(
            leader,
            prefiltered_universe,
            min_similarity=min_similarity,
            min_divergence=min_divergence,
            max_universe=max_universe,
            max_candidates_per_leader=max(1, int(top_candidates)),
            exploratory=exploratory,
            include_weak_similarity=include_weak_similarity,
            min_related_abs_move_24h=min_related_abs_move_24h,
            min_similarity_for_trade=min_similarity_for_trade,
            deadline=deadline if enforce_deadline_for_leader else None,
        )

        # Count only VALID candidates for skip decision
        valid_evaluations = []
        for e in evaluations:
            if e["rejected_reason"] is not None:
                continue
            rel_type = e["relationship_type"]
            if not _is_valid_relationship_type(
                rel_type,
                exploratory=exploratory,
                include_weak_similarity=include_weak_similarity,
            ):
                continue
            valid_evaluations.append(e)
        
        rejected_evaluations = [e for e in evaluations if e["rejected_reason"] is not None]
        leader_skipped = len(valid_evaluations) < min_valid_candidates

        rejection_tally: dict[str, int] = {}
        for e in rejected_evaluations:
            reason = e["rejected_reason"] or "unknown"
            rejection_tally[reason] = rejection_tally.get(reason, 0) + 1
        top_rejection_reasons = sorted(
            (f"{reason}: {count}" for reason, count in rejection_tally.items()),
            key=lambda s: int(s.split(": ")[-1]),
            reverse=True,
        )[:5]

        debug_rows.append(
            {
                "leader_market_title": leader.get("title"),
                "option_set_type": option_set_type,
                "leader_option_set_key": leader_option_set_key,
                "leader_skipped": leader_skipped,
                "skip_reason": "no_valid_candidates" if leader_skipped else None,
                "leader_prefilter_skip_reason": None,
                "resolved_or_bad_tick_candidate": False,
                "leader_price_change_1h": _change(leader, "one_hour_price_change"),
                "leader_price_change_24h": _change(leader, "one_day_price_change"),
                "leader_current_price": _leader_current_price(leader),
                "valid_candidate_count": len(valid_evaluations),
                "rejected_candidate_count": len(rejected_evaluations),
                "top_rejection_reasons": top_rejection_reasons,
                "universe_source_path": universe_source_path,
                "universe_row_count": universe_row_count if universe_row_count is not None else len(polymarket_universe),
                "number_of_local_universe_markets_searched": len(polymarket_universe),
                "prefiltered_universe_count": len(prefiltered_universe),
                "max_universe": max_universe,
                "best_valid_candidates": [
                    {
                    "related_market_title": e["related_market_title"],
                    "relationship_type": e["relationship_type"],
                    "expected_correlation_direction": e["expected_correlation_direction"],
                    "suggested_trade_direction": e["suggested_trade_direction"],
                    "tradable_signal": e["tradable_signal"],
                    "low_related_activity": e["low_related_activity"],
                    "below_similarity_threshold": e["below_similarity_threshold"],
                    "similarity_score": e["similarity_score"],
                        "divergence_score": e["divergence_score"],
                        "option_set_key": e["option_set_key"],
                        "subset_relationship": e["subset_relationship"],
                        "subset_relationship_reason": e["subset_relationship_reason"],
                        "resolved_or_bad_tick_candidate": e["resolved_or_bad_tick_candidate"],
                    }
                    for e in valid_evaluations[:5]
                ],
                "top_related_candidates_before_threshold_filtering": [
                    {
                        "related_market_title": evaluation["related_market_title"],
                        "similarity_score": evaluation["similarity_score"],
                        "relationship_type": evaluation["relationship_type"],
                        "expected_correlation_direction": evaluation["expected_correlation_direction"],
                        "suggested_trade_direction": evaluation["suggested_trade_direction"],
                        "tradable_signal": evaluation["tradable_signal"],
                        "low_related_activity": evaluation["low_related_activity"],
                        "below_similarity_threshold": evaluation["below_similarity_threshold"],
                        "causal_template_matched": evaluation["causal_template_matched"],
                        "causal_link_reason": evaluation["causal_link_reason"],
                        "option_set_key": evaluation["option_set_key"],
                        "subset_relationship": evaluation["subset_relationship"],
                        "subset_relationship_reason": evaluation["subset_relationship_reason"],
                        "resolved_or_bad_tick_candidate": evaluation["resolved_or_bad_tick_candidate"],
                        "relationship_reason": evaluation["relationship_reason"],
                        "rejected_reason": evaluation["rejected_reason"],
                        "excluded_from_normal_output": evaluation["excluded_from_normal_output"],
                        "excluded_as_self_market": evaluation["excluded_as_self_market"],
                        "excluded_as_duplicate_bucket": evaluation["excluded_as_duplicate_bucket"],
                    }
                    for evaluation in evaluations[: max(1, int(top_candidates))]
                ],
            }
        )
    elapsed_time_total = time.monotonic() - started_at
    if partial_timeout and leaders_evaluated_before_timeout == 0:
        leaders_evaluated_before_timeout = leaders_evaluated
    for row in debug_rows:
        row["elapsed_time_total"] = elapsed_time_total
        row["leaders_evaluated_before_timeout"] = leaders_evaluated_before_timeout
        row["partial_results_due_to_timeout"] = partial_timeout
    return debug_rows
