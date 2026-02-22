"""CLI entry point for the market fetcher project.

Usage examples:
    python main.py --source polymarket
    python main.py --source kalshi
    python main.py --source all --top-n 10 --pretty
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import requests

from market_fetcher.config import get_settings
from market_fetcher.fetch_data import fetch_all
from market_fetcher.fetch_kalshi import fetch_kalshi_snapshot
from market_fetcher.fetch_polymarket import fetch_polymarket_snapshots
from market_fetcher.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Fetch market snapshots from Kalshi and Polymarket")
    parser.add_argument(
        "--source",
        choices=["all", "kalshi", "polymarket"],
        default="all",
        help="Which data source to fetch.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top Polymarket markets to fetch (used by polymarket/all).",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser


def run(source: str, top_n: int) -> dict[str, Any] | list[dict[str, Any]]:
    """Dispatch fetching based on selected source."""

    if top_n <= 0:
        raise ValueError("--top-n must be greater than 0")

    if source == "polymarket":
        return fetch_polymarket_snapshots(top_n=top_n)
    if source == "kalshi":
        return fetch_kalshi_snapshot()
    return fetch_all(polymarket_top_n=top_n)


def main() -> int:
    """Application entrypoint returning process exit code."""

    args = build_parser().parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        result = run(source=args.source, top_n=args.top_n)
        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))
        return 0
    except requests.RequestException as exc:
        LOGGER.error("Network/API error: %s", exc)
        return 1
    except ValueError as exc:
        LOGGER.error("Invalid input: %s", exc)
        return 2
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Unexpected error: %s", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
