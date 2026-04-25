from __future__ import annotations

import json
from pathlib import Path


def load_latest_previous_snapshot(history_path: str | None) -> dict | None:
    """Return the last JSON payload from a JSONL history file, or None."""
    if history_path is None:
        return None
    path = Path(history_path)
    if not path.exists():
        return None
    last_line: str | None = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if last_line is None:
        return None
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return None


def compute_market_deltas(
    current_markets: list[dict],
    previous_payload: dict | None,
) -> list[dict]:
    """Return copies of current_markets with previous_mid and computed_mid_delta added.

    Matches by market_key.  Fields are None when no prior data exists.
    Input list is never mutated.
    """
    if previous_payload is None:
        return [
            {**m, "previous_mid": None, "computed_mid_delta": None}
            for m in current_markets
        ]

    prev_by_key: dict[str, dict] = {}
    for m in previous_payload.get("polymarket_snapshots", []):
        key = m.get("market_key")
        if key is not None:
            prev_by_key[str(key)] = m

    result: list[dict] = []
    for m in current_markets:
        key = str(m.get("market_key", ""))
        prev = prev_by_key.get(key)
        cur_mid = m.get("mid")
        if prev is not None and cur_mid is not None and prev.get("mid") is not None:
            previous_mid: float | None = prev["mid"]
            delta: float | None = cur_mid - previous_mid
        else:
            previous_mid = None
            delta = None
        result.append({**m, "previous_mid": previous_mid, "computed_mid_delta": delta})
    return result
