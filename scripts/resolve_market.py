"""CLI entrypoint for deterministic market resolver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.market_resolver import TradeIntent, resolve_trade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve trade intent to best Kalshi/Polymarket markets")
    parser.add_argument("query", type=str, help="Natural-language trade intent query")
    parser.add_argument("--side", choices=["YES", "NO"], default="YES")
    parser.add_argument("--notional", type=float, default=100.0)
    parser.add_argument("--mode", choices=["live", "fixture"], default="live")
    parser.add_argument("--fixtures-dir", type=str, default="fixtures")
    parser.add_argument("--allow-fallback", action="store_true", default=True)
    parser.add_argument("--no-fallback", dest="allow_fallback", action="store_false")
    parser.add_argument("--category-hint", type=str, default=None)
    parser.add_argument("--event-date", type=str, default=None)
    parser.add_argument("--venue-preference", type=str, default=None)
    parser.add_argument("--include-mve", action="store_true")
    parser.add_argument("--trade-id", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    intent = TradeIntent(
        trade_id=args.trade_id,
        query=args.query,
        side=args.side,
        notional_usd=args.notional,
        category_hint=args.category_hint,
        event_date=args.event_date,
        venue_preference=args.venue_preference,
    )

    resolved = resolve_trade(
        intent=intent,
        mode=args.mode,
        fixtures_dir=args.fixtures_dir,
        allow_fallback=args.allow_fallback,
        include_mve=args.include_mve,
    )

    print(json.dumps(resolved.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
