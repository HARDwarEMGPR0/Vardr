"""Build flat snapshot tables from data/snapshots.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required: pip install pandas pyarrow") from exc


ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOTS_PATH = ROOT_DIR / "data" / "snapshots.json"
CSV_PATH = ROOT_DIR / "data" / "snapshots.csv"
PARQUET_PATH = ROOT_DIR / "data" / "snapshots.parquet"

REQUIRED_FIELDS = [
    "ts_utc",
    "venue",
    "market_key",
    "title",
    "question",
    "slug",
    "close_time_utc",
    "mid",
    "spread",
    "spread_reported",
    "depth_top5",
    "depth_yes_top5",
    "depth_no_top5",
    "best_yes_bid",
    "best_yes_ask",
    "best_no_bid",
    "best_no_ask",
    "volume_proxy",
    "liquidity_proxy",
    "one_hour_price_change",
    "one_day_price_change",
    "last_trade_price",
    "trades_count_sample",
    "minute_features",
]


def _load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, list) else []
        except json.JSONDecodeError:
            return []

    runs: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            runs.append(parsed)
    return runs


def _coerce_snapshot(snapshot: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"run_id": run.get("run_id"), "run_ts_utc": run.get("ts_utc")}
    for field in REQUIRED_FIELDS:
        row[field] = snapshot.get(field)
    row["mode"] = run.get("mode")
    row["run_status"] = run.get("run_status")
    return row


def main() -> int:
    runs = _load_runs(SNAPSHOTS_PATH)
    rows: list[dict[str, Any]] = []

    for run in runs:
        for snap in run.get("polymarket_snapshots", []) or []:
            if isinstance(snap, dict):
                rows.append(_coerce_snapshot(snap, run))
        for snap in run.get("kalshi_snapshots", []) or []:
            if isinstance(snap, dict):
                rows.append(_coerce_snapshot(snap, run))

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(CSV_PATH, index=False)
        df.to_parquet(PARQUET_PATH, index=False)
    else:
        df.to_csv(CSV_PATH, index=False)
        df.to_parquet(PARQUET_PATH, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
