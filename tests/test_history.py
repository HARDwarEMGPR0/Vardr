import json
import tempfile
from pathlib import Path

from market_fetcher.history import load_latest_previous_snapshot, compute_market_deltas


# ---------------------------------------------------------------------------
# load_latest_previous_snapshot
# ---------------------------------------------------------------------------

def test_load_returns_none_for_none_path():
    assert load_latest_previous_snapshot(None) is None


def test_load_returns_none_for_missing_file():
    assert load_latest_previous_snapshot("/nonexistent/path/history.jsonl") is None


def test_load_returns_none_for_empty_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp = f.name
    try:
        assert load_latest_previous_snapshot(tmp) is None
    finally:
        Path(tmp).unlink()


def test_load_returns_last_line_of_jsonl():
    records = [{"run_id": "first", "v": 1}, {"run_id": "second", "v": 2}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        tmp = f.name
    try:
        result = load_latest_previous_snapshot(tmp)
        assert result == records[-1]
        assert result["run_id"] == "second"
    finally:
        Path(tmp).unlink()


def test_load_returns_none_for_corrupt_last_line():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "good"}) + "\n")
        f.write("not-valid-json\n")
        tmp = f.name
    try:
        assert load_latest_previous_snapshot(tmp) is None
    finally:
        Path(tmp).unlink()


# ---------------------------------------------------------------------------
# compute_market_deltas
# ---------------------------------------------------------------------------

def test_compute_deltas_none_previous_adds_null_fields():
    markets = [{"market_key": "m1", "mid": 0.5, "title": "Test"}]
    result = compute_market_deltas(markets, None)
    assert len(result) == 1
    assert result[0]["previous_mid"] is None
    assert result[0]["computed_mid_delta"] is None
    assert result[0]["title"] == "Test"  # original fields preserved


def test_compute_deltas_with_matching_market():
    previous_payload = {
        "polymarket_snapshots": [
            {"market_key": "m1", "mid": 0.40},
        ]
    }
    current = [{"market_key": "m1", "mid": 0.45}]
    result = compute_market_deltas(current, previous_payload)
    assert result[0]["previous_mid"] == 0.40
    assert abs(result[0]["computed_mid_delta"] - 0.05) < 1e-9


def test_compute_deltas_no_matching_market():
    previous_payload = {
        "polymarket_snapshots": [{"market_key": "other", "mid": 0.50}]
    }
    current = [{"market_key": "m1", "mid": 0.45}]
    result = compute_market_deltas(current, previous_payload)
    assert result[0]["previous_mid"] is None
    assert result[0]["computed_mid_delta"] is None


def test_compute_deltas_does_not_mutate_input():
    original = [{"market_key": "m1", "mid": 0.5}]
    original_copy = [dict(m) for m in original]
    compute_market_deltas(original, None)
    assert original == original_copy


def test_compute_deltas_preserves_existing_fields():
    previous_payload = {
        "polymarket_snapshots": [{"market_key": "m1", "mid": 0.40}]
    }
    current = [
        {
            "market_key": "m1",
            "mid": 0.45,
            "one_hour_price_change": 0.02,
            "spread": 0.01,
        }
    ]
    result = compute_market_deltas(current, previous_payload)
    assert result[0]["one_hour_price_change"] == 0.02
    assert result[0]["spread"] == 0.01
    assert result[0]["computed_mid_delta"] is not None
