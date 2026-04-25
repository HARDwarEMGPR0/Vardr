from __future__ import annotations

import json
from pathlib import Path


def load_leader_markets(source: str | None = None) -> list[dict]:
    """Return a list of leader-market dicts.

    source=None  → empty list (no external leaders configured)
    source=path  → load and return the list from the JSON file at that path
    """
    if source is None:
        return []

    path = Path(source)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Leader markets file must contain a JSON array, got {type(data).__name__}")
    return data
