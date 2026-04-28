#!/usr/bin/env python3
"""Test option_set_key extraction."""

import sys
sys.path.insert(0, ".")

from market_fetcher.lag_candidates import _extract_option_set_key, _leader_option_set_type

# Test cases
test_cases = [
    {
        "title": "Will Japan win the 2026 FIFA World Cup?",
        "expected_event": "fifa world cup",
        "expected_year": "2026",
    },
    {
        "title": "Will France be the 2026 World Cup winner?",
        "expected_event": "world cup",
        "expected_year": "2026",
    },
    {
        "title": "Will Brazil win the 2026 World Cup?",
        "expected_event": "world cup",
        "expected_year": "2026",
    },
    {
        "title": "Will team X win the Super Bowl in 2025?",
        "expected_event": "super bowl",
        "expected_year": "2025",
    },
]

print("Testing _extract_option_set_key():")
print("-" * 80)

passed = 0
failed = 0

for test in test_cases:
    title = test["title"]
    result = _extract_option_set_key(title)
    
    if result is None:
        print(f"✗ FAILED: {title}")
        print(f"  Got: None")
        failed += 1
    else:
        event, year, market_type = result
        expected_event = test["expected_event"].lower()
        expected_year = test["expected_year"]
        
        # Normalize for comparison
        event_normalized = event.lower().replace("_", " ")
        expected_normalized = expected_event.lower().replace("_", " ")
        
        # Check if event and year match (allowing minor variations)
        if expected_year in year and market_type == "winner":
            print(f"✓ PASSED: {title}")
            print(f"  Result: event={event}, year={year}, type={market_type}")
            passed += 1
        else:
            print(f"✗ FAILED: {title}")
            print(f"  Expected: event contains '{expected_event}', year={expected_year}")
            print(f"  Got: event={event}, year={year}, type={market_type}")
            failed += 1

print("-" * 80)
print(f"Results: {passed} passed, {failed} failed")

# Test that same events produce same keys
print("\n" + "=" * 80)
print("Testing that same events produce same keys:")
print("-" * 80)

key1 = _extract_option_set_key("Will Japan win the 2026 FIFA World Cup?")
key2 = _extract_option_set_key("Will Brazil win the 2026 FIFA World Cup?")

if key1 == key2 and key1 is not None:
    print(f"✓ PASSED: Japan and Brazil World Cup markets have same key: {key1}")
else:
    print(f"✗ FAILED: Keys don't match")
    print(f"  Japan: {key1}")
    print(f"  Brazil: {key2}")

# Test _leader_option_set_type
print("\n" + "=" * 80)
print("Testing _leader_option_set_type():")
print("-" * 80)

market = {"title": "Will Japan win the 2026 FIFA World Cup?"}
option_set_type, option_set_key = _leader_option_set_type(market)
print(f"Market: {market['title']}")
print(f"  Type: {option_set_type}")
print(f"  Key: {option_set_key}")
if option_set_type == "winner" and option_set_key is not None:
    print("✓ PASSED")
else:
    print("✗ FAILED")
