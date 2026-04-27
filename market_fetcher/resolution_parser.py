from __future__ import annotations

import re
from datetime import datetime, timezone

_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_MONTH_PATTERN = "|".join(_MONTHS)


def _evidence_text(market: dict) -> tuple[str, str]:
    rules_parts = [
        market.get("rules"),
        market.get("description"),
        market.get("resolution_source"),
    ]
    title_parts = [market.get("title"), market.get("question"), market.get("slug")]
    rules_text = " ".join(str(part) for part in rules_parts if part)
    title_text = " ".join(str(part) for part in title_parts if part)
    return rules_text, title_text


def _parse_date(text: str) -> str | None:
    match = re.search(
        rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            rf"\b({_MONTH_PATTERN})\s+(\d{{4}})\b",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        month, year = match.groups()
        date_text = f"{month} 1 {year}"
    else:
        month, day, year = match.groups()
        date_text = f"{month} {day} {year}"

    try:
        parsed = datetime.strptime(date_text.title(), "%B %d %Y")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _reference_event(text: str) -> str | None:
    if re.search(r"\bgta\s*(vi|6)\b", text, flags=re.IGNORECASE):
        return "GTA VI"
    before = re.search(r"\bbefore\s+([^?.;,]+)", text, flags=re.IGNORECASE)
    if before:
        event = before.group(1).strip()
        if event and not _parse_date(event):
            return event[:80]
    return None


def _close_time(market: dict) -> str | None:
    return (
        market.get("close_time_utc")
        or market.get("close_time")
        or market.get("closedTime")
        or market.get("endDate")
        or market.get("expiration_time")
        or market.get("end_date")
    )


def parse_resolution_metadata(market: dict) -> dict:
    rules_text, title_text = _evidence_text(market)
    combined = f"{rules_text} {title_text}".strip()
    rules_lower = rules_text.lower()
    combined_lower = combined.lower()

    explicit_date = _parse_date(rules_text) or _parse_date(title_text)
    reference_event = _reference_event(combined)
    raw_evidence = combined

    if reference_event == "GTA VI" and "before gta" in combined_lower:
        return {
            "resolution_deadline_utc": explicit_date,
            "reference_event": "GTA VI",
            "rule_type": "before_reference_event",
            "deadline_source": "rules_text" if rules_text else "unknown",
            "deadline_confidence": 0.85 if rules_text else 0.55,
            "raw_evidence": raw_evidence,
        }

    if reference_event == "GTA VI" and re.search(r"\bgta\s*(vi|6)\s+released\b|\breleased\s+.*\bgta\s*(vi|6)\b", combined_lower):
        return {
            "resolution_deadline_utc": explicit_date,
            "reference_event": "GTA VI",
            "rule_type": "release_timing",
            "deadline_source": "rules_text" if rules_text else "unknown",
            "deadline_confidence": 0.85 if rules_text and explicit_date else 0.6 if explicit_date else 0.4,
            "raw_evidence": raw_evidence,
        }

    if explicit_date:
        return {
            "resolution_deadline_utc": explicit_date,
            "reference_event": reference_event,
            "rule_type": "fixed_date",
            "deadline_source": "rules_text" if rules_text and _parse_date(rules_text) else "unknown",
            "deadline_confidence": 0.85 if rules_text and _parse_date(rules_text) else 0.65,
            "raw_evidence": raw_evidence,
        }

    if re.search(r"\bwill\b|\bif\b|\bwhether\b", combined_lower):
        rule_type = "event_occurrence"
    else:
        rule_type = "unknown"

    close_time = _close_time(market)
    return {
        "resolution_deadline_utc": close_time,
        "reference_event": reference_event,
        "rule_type": rule_type,
        "deadline_source": "close_time" if close_time else "unknown",
        "deadline_confidence": 0.3 if close_time else 0.0,
        "raw_evidence": raw_evidence,
    }
