from market_fetcher.resolution_parser import parse_resolution_metadata


def test_explicit_date_parsing_works():
    meta = parse_resolution_metadata({
        "title": "Will X happen?",
        "rules": "This market resolves YES if X occurs by June 30, 2026.",
    })

    assert meta["resolution_deadline_utc"] == "2026-06-30T00:00:00Z"
    assert meta["rule_type"] == "fixed_date"
    assert meta["deadline_source"] == "rules_text"
    assert meta["deadline_confidence"] >= 0.8


def test_before_gta_vi_detected_as_reference_event():
    meta = parse_resolution_metadata({
        "title": "Will X happen before GTA VI?",
        "rules": "This market resolves YES if X happens before GTA VI is released.",
    })

    assert meta["reference_event"] == "GTA VI"
    assert meta["rule_type"] == "before_reference_event"
    assert meta["deadline_confidence"] >= 0.8


def test_missing_rules_falls_back_to_close_time_with_low_confidence():
    meta = parse_resolution_metadata({
        "title": "Will X happen?",
        "close_time_utc": "2026-07-01T00:00:00Z",
    })

    assert meta["resolution_deadline_utc"] == "2026-07-01T00:00:00Z"
    assert meta["deadline_source"] == "close_time"
    assert meta["deadline_confidence"] <= 0.3
