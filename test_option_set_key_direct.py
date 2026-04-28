#!/usr/bin/env python3
"""Test option_set_key extraction - direct function test."""

import re

def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())

def _extract_option_set_key(title: str | None) -> tuple[str, str, str] | None:
    """Extract and normalize (event, year, market_type) from a winner market title."""
    if not title:
        return None
    
    text = _normalize_text(title)
    
    # Try multiple patterns:
    # 1. "will [entity] win ... [event]"
    # 2. "will [entity] be the ... [event] winner"
    
    match = None
    event_part = None
    
    # Pattern 1: will X win [the] EVENT
    match = re.search(r"\bwill\s+(.+?)\s+win\s+(?:the\s+)?(.+?)(?:\?)?$", text)
    if match:
        event_part = match.group(2).strip()
    else:
        # Pattern 2: will X be the EVENT winner
        match = re.search(r"\bwill\s+(.+?)\s+be\s+(?:the\s+)?(.+?)\s+winner(?:\?)?$", text)
        if match:
            event_part = match.group(2).strip()
    
    if not event_part:
        return None
    
    # Remove trailing "winner" or "winner?"
    event_part = re.sub(r"\s+winner\s*(?:\?)?$", "", event_part).strip()
    event_part = re.sub(r"\b(yes|no)\b", "", event_part).strip()
    
    if not event_part:
        return None
    
    # Extract year (4 digits) - appears anywhere in the event part
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", event_part)
    year = year_match.group(1) if year_match else ""
    
    # Remove year from event for normalization (including "in YEAR", "by YEAR", etc.)
    event_normalized = re.sub(r"\b(?:in|by)\s+(20\d{2}|19\d{2})\b", " ", event_part).strip()
    event_normalized = re.sub(r"\b(20\d{2}|19\d{2})\b", " ", event_normalized).strip()
    
    # Standardize common phrases
    event_normalized = re.sub(r"\bwinner\s+of\b", "", event_normalized)
    event_normalized = re.sub(r"\bwinner?\b", "", event_normalized)
    
    # Remove prepositions that shouldn't be part of event name
    event_normalized = re.sub(r"\s+(in|by|at|of)\s+", " ", event_normalized)
    
    # Clean up whitespace
    event_normalized = re.sub(r"\s+", " ", event_normalized).strip()
    
    # Extract main event keywords
    event_keywords = event_normalized.split()
    if not event_keywords:
        return None
    
    event_key = " ".join(event_keywords).lower()
    
    # Final cleanup
    event_key = re.sub(r"\s+", " ", event_key).strip()
    
    if not event_key:
        return None
    
    # Always use "winner" as market_type for this function
    return (event_key, year, "winner")

# Test cases
test_cases = [
    ("Will Japan win the 2026 FIFA World Cup?", ("fifa world cup", "2026", "winner")),
    ("Will France be the 2026 World Cup winner?", ("world cup", "2026", "winner")),
    ("Will Brazil win the 2026 World Cup?", ("world cup", "2026", "winner")),
    ("Will team X win the Super Bowl in 2025?", ("super bowl", "2025", "winner")),
]

print("Testing _extract_option_set_key():")
print("=" * 80)

passed = 0
failed = 0

for title, expected in test_cases:
    result = _extract_option_set_key(title)
    
    if result is None:
        print(f"✗ FAILED: {title}")
        print(f"  Expected: {expected}")
        print(f"  Got: None")
        failed += 1
    else:
        event, year, market_type = result
        exp_event, exp_year, exp_type = expected
        
        if event == exp_event and year == exp_year and market_type == exp_type:
            print(f"✓ PASSED: {title}")
            print(f"  Result: {result}")
            passed += 1
        else:
            print(f"✗ FAILED: {title}")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
            failed += 1

print("=" * 80)
print(f"Results: {passed} passed, {failed} failed")

# Test that same events produce same keys
print("\n" + "=" * 80)
print("Testing that same events produce same keys:")
print("=" * 80)

key1 = _extract_option_set_key("Will Japan win the 2026 FIFA World Cup?")
key2 = _extract_option_set_key("Will Brazil win the 2026 FIFA World Cup?")

print(f"Japan key:   {key1}")
print(f"Brazil key:  {key2}")

if key1 == key2 and key1 is not None:
    print(f"✓ PASSED: Both have the same key")
else:
    print(f"✗ FAILED: Keys don't match")

# Test that different events produce different keys
print("\n" + "=" * 80)
print("Testing that different years produce different keys:")
print("=" * 80)

key2025 = _extract_option_set_key("Will Japan win the 2025 FIFA World Cup?")
key2026 = _extract_option_set_key("Will Japan win the 2026 FIFA World Cup?")

print(f"2025 key: {key2025}")
print(f"2026 key: {key2026}")

if key2025 != key2026:
    print(f"✓ PASSED: Different years produce different keys")
else:
    print(f"✗ FAILED: Different years should produce different keys")
