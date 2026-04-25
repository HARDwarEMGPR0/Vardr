"""CLI for fetching Kalshi + Polymarket snapshots and risk scores."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from market_fetcher.history import load_latest_previous_snapshot, compute_market_deltas
from market_fetcher.intelligence import (
    find_opportunities,
    detect_inconsistencies,
    detect_reference_event_clusters,
    detect_lagging_correlated_markets,
)
from market_fetcher.leader_markets import load_leader_markets


from src.connectors.kalshi_public import fetch_markets as fetch_kalshi_markets  # noqa: E402
from src.connectors.polymarket_gamma import fetch_markets as fetch_polymarket_markets  # noqa: E402
from src.normalize import normalize_kalshi_market, normalize_polymarket_market, utc_now_iso  # noqa: E402
from src.scoring import compute_divergence, execution_risk_index  # noqa: E402

LOGGER = logging.getLogger(__name__)
STOPWORDS = {
    "will",
    "the",
    "a",
    "an",
    "in",
    "on",
    "by",
    "of",
    "to",
    "win",
    "2026",
}
TOPIC_KEYWORDS = {
    "world_cup": {"world", "cup", "fifa", "group", "winner", "soccer"},
    "fed_rates": {"fed", "rates", "cut", "hike", "fomc"},
    "inflation": {"cpi", "inflation", "yoy", "core"},
    "crypto": {"btc", "bitcoin", "eth", "ethereum", "crypto"},
    "elections": {"election", "mayor", "president", "senate", "vote"},
    "sports": {"playoffs", "nfl", "nba", "mlb", "nhl", "chiefs"},
}


def load_fixture(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_fixture_atomic(path: str | Path, obj: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)


def classify_network_error(exc: Exception) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout, requests.exceptions.SSLError)):
        return True
    text = str(exc)
    return (
        "WinError 10013" in text
        or "Failed to establish a new connection" in text
        or "Connection refused" in text
        or "Name or service not known" in text
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch market snapshots from Kalshi and Polymarket")
    parser.add_argument("--mode", choices=["live", "fixture"], default="live")
    parser.add_argument("--fixtures-dir", type=str, default="fixtures")
    parser.add_argument("--poly-limit", type=int, default=200)
    parser.add_argument("--kalshi-limit", type=int, default=200)
    parser.add_argument("--n-markets", type=int, default=5, help="Default market count for both venues")
    parser.add_argument("--n-kalshi", type=int, default=None, help="Override Kalshi count")
    parser.add_argument("--n-polymarket", type=int, default=None, help="Override Polymarket count")
    parser.add_argument("--include-mve", action="store_true", help="Include KXMVESPORTSMULTIGAMEEXTENDED markets")
    parser.add_argument("--mapping", type=str, default=None)
    parser.add_argument("--save-fixtures", action="store_true")
    parser.add_argument("--leader-markets-json", type=str, default=None, dest="leader_markets_json")
    parser.add_argument("--history-jsonl", type=str, default=None, dest="history_jsonl")
    parser.add_argument("--debug-intelligence", action="store_true", dest="debug_intelligence")
    parser.add_argument("--summary", action="store_true")

    def _parse_bool(value: str | None) -> bool:
        if value is None:
            return True
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")

    fb = parser.add_mutually_exclusive_group()
    fb.add_argument("--allow-fallback", dest="allow_fallback", nargs="?", const=True, type=_parse_bool, default=None)
    fb.add_argument("--no-fallback", dest="allow_fallback", action="store_false")
    return parser.parse_args()


def _effective_count(args: argparse.Namespace, venue: str) -> int:
    n_kalshi = getattr(args, "n_kalshi", None)
    n_polymarket = getattr(args, "n_polymarket", None)
    n_markets = getattr(args, "n_markets", 5)

    if venue == "kalshi" and n_kalshi is not None:
        return max(1, int(n_kalshi))
    if venue == "polymarket" and n_polymarket is not None:
        return max(1, int(n_polymarket))
    return max(1, int(n_markets))


def _load_mapping(path: str | None) -> tuple[list[tuple[str, str]], str | None]:
    if not path:
        return [], None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"mapping_error: {exc}"
    pairs = data.get("pairs", []) if isinstance(data, dict) else []
    out: list[tuple[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        k = pair.get("kalshi_ticker")
        p = pair.get("polymarket_conditionId")
        if k and p:
            out.append((str(k), str(p)))
    return out, None


def _append_run_to_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _append_run_to_file(record: dict[str, Any]) -> None:
    path = ROOT_DIR / "data" / "snapshots.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                history = existing
        except json.JSONDecodeError:
            history = []
    history.append(record)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _run_status(venue_status: dict[str, dict[str, Any]]) -> str:
    sources = {venue_status["polymarket"]["source"], venue_status["kalshi"]["source"]}
    if sources == {"live"}:
        return "live"
    if sources == {"fixture"}:
        return "fixture"
    return "mixed"


def _normalize_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    lowered = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return {t for t in lowered.split() if t and t not in STOPWORDS}


def _topic_tags(tokens: set[str]) -> set[str]:
    tags: set[str] = set()
    for tag, words in TOPIC_KEYWORDS.items():
        if tokens & words:
            tags.add(tag)
    return tags


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _number_bonus(a: set[str], b: set[str]) -> float:
    nums_a = {t for t in a if t.isdigit()}
    nums_b = {t for t in b if t.isdigit()}
    return 0.15 if nums_a and (nums_a & nums_b) else 0.0


def _match_score(k_title: str | None, p_title: str | None) -> float:
    k_tokens = _normalize_tokens(k_title)
    p_tokens = _normalize_tokens(p_title)
    k_topics = _topic_tags(k_tokens)
    p_topics = _topic_tags(p_tokens)

    if k_topics and p_topics and not (k_topics & p_topics):
        return 0.0

    return min(1.0, _jaccard(k_tokens, p_tokens) + _number_bonus(k_tokens, p_tokens))


def _example_k(s: dict[str, Any]) -> dict[str, Any]:
    return {
        "kalshi_ticker": s.get("market_key"),
        "kalshi_title": s.get("title") or s.get("question"),
    }


def _example_p(s: dict[str, Any]) -> dict[str, Any]:
    return {
        "polymarket_market_key": s.get("market_key") or s.get("slug"),
        "polymarket_title": s.get("title") or s.get("question"),
    }


def _pad_examples(items: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    out = items[:2]
    while len(out) < 2:
        out.append(_example_k({}) if kind == "k" else _example_p({}))
    return out


def build_cross_venue(
    kalshi: list[dict[str, Any]],
    poly: list[dict[str, Any]],
    threshold: float = 0.35,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for k in kalshi:
        for p in poly:
            score = _match_score(k.get("title") or k.get("question"), p.get("title") or p.get("question"))
            if score >= threshold:
                candidates.append((score, k, p))
    candidates.sort(key=lambda x: x[0], reverse=True)

    used_k: set[str] = set()
    used_p: set[str] = set()
    matched: list[dict[str, Any]] = []

    for score, k, p in candidates:
        k_key = str(k.get("market_key"))
        p_key = str(p.get("market_key"))
        if k_key in used_k or p_key in used_p:
            continue
        matched.append(
            {
                "kalshi_ticker": k.get("market_key"),
                "kalshi_title": k.get("title") or k.get("question"),
                "polymarket_market_key": p.get("market_key") or p.get("slug"),
                "polymarket_title": p.get("title") or p.get("question"),
                "match_score": round(score, 4),
            }
        )
        used_k.add(k_key)
        used_p.add(p_key)
        if len(matched) == 2:
            break

    k_only = [_example_k(x) for x in kalshi if str(x.get("market_key")) not in used_k]
    p_only = [_example_p(x) for x in poly if str(x.get("market_key")) not in used_p]

    return matched, _pad_examples(k_only, "k"), _pad_examples(p_only, "p")


def _selection_summary_block(debug: dict[str, Any]) -> str:
    return (
        f"fetched={debug.get('fetched_candidates', 0)} "
        f"eligible={debug.get('eligible_two_sided', 0)} "
        f"selected={debug.get('selected', 0)} "
        f"excluded={debug.get('excluded', 0)} "
        f"reasons={debug.get('exclusion_reasons', {})}"
    )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if not hasattr(args, "n_kalshi"):
        setattr(args, "n_kalshi", None)
    if not hasattr(args, "n_polymarket"):
        setattr(args, "n_polymarket", None)

    ts = utc_now_iso()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    fixtures_dir = Path(args.fixtures_dir)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    mode = getattr(args, "mode", "live")
    allow_fallback = getattr(args, "allow_fallback", None)
    if allow_fallback is None:
        allow_fallback = True if mode == "live" else False
    include_mve = getattr(args, "include_mve", False)

    n_poly = _effective_count(args, "polymarket")
    n_kalshi = _effective_count(args, "kalshi")

    poly_markets, poly_source, poly_error, poly_debug = fetch_polymarket_markets(
        mode=args.mode,
        allow_fallback=allow_fallback,
        fixtures_dir=fixtures_dir,
        save_fixtures=getattr(args, "save_fixtures", False) and mode == "live",
        limit=getattr(args, "poly_limit", 200),
        offset=0,
        n_markets=n_poly,
        loader=load_fixture,
        saver=save_fixture_atomic,
        classify_network_error=classify_network_error,
    )
    polymarket_snapshots = [normalize_polymarket_market(market, ts_utc=ts) for market in poly_markets]

    history_path = getattr(args, "history_jsonl", None)
    previous_payload = load_latest_previous_snapshot(history_path)
    polymarket_snapshots = compute_market_deltas(polymarket_snapshots, previous_payload)

    kalshi_markets, kalshi_source, kalshi_error, kalshi_debug = fetch_kalshi_markets(
        mode=args.mode,
        allow_fallback=allow_fallback,
        fixtures_dir=fixtures_dir,
        save_fixtures=getattr(args, "save_fixtures", False) and mode == "live",
        limit=getattr(args, "kalshi_limit", 200),
        n_markets=n_kalshi,
        include_mve=include_mve,
        loader=load_fixture,
        saver=save_fixture_atomic,
        classify_network_error=classify_network_error,
    )
    kalshi_snapshots = [
        normalize_kalshi_market(
            ticker=str(market.get("ticker")),
            market_json=market,
            orderbook_json=market.get("orderbook", {}),
            trades_json=market.get("trades", {}),
            ts_utc=ts,
            minute_features={
                "selection_meta": {
                    "depth_top5": market.get("depth_top5"),
                    "depth_yes_top5": market.get("depth_yes_top5"),
                    "depth_no_top5": market.get("depth_no_top5"),
                    "trades_count_sample": market.get("trades_count_sample"),
                }
            },
        )
        for market in kalshi_markets
    ]

    mapping_pairs, mapping_error = _load_mapping(getattr(args, "mapping", None))

    matched_pairs, kalshi_only_examples, polymarket_only_examples = build_cross_venue(
        kalshi_snapshots,
        polymarket_snapshots,
    )

    divergence_by_key: dict[str, float] = {}
    poly_by_key = {str(s.get("market_key")): s for s in polymarket_snapshots}
    kalshi_by_key = {str(s.get("market_key")): s for s in kalshi_snapshots}

    for k_ticker, p_id in mapping_pairs:
        if k_ticker in kalshi_by_key and p_id in poly_by_key:
            d = compute_divergence(kalshi_by_key[k_ticker], poly_by_key[p_id])
            if d is not None:
                divergence_by_key[k_ticker] = d
                divergence_by_key[p_id] = d

    for pair in matched_pairs:
        k_key = str(pair.get("kalshi_ticker"))
        p_key = str(pair.get("polymarket_market_key"))
        if k_key in kalshi_by_key and p_key in poly_by_key:
            d = compute_divergence(kalshi_by_key[k_key], poly_by_key[p_key])
            if d is not None:
                divergence_by_key[k_key] = d
                divergence_by_key[p_key] = d

    opportunities = find_opportunities(polymarket_snapshots)
    reference_event_clusters = detect_reference_event_clusters(polymarket_snapshots)
    inconsistencies = detect_inconsistencies(polymarket_snapshots)
    loaded_leaders = load_leader_markets(getattr(args, "leader_markets_json", None))
    lag_signals = detect_lagging_correlated_markets(
        markets=polymarket_snapshots,
        leader_markets=loaded_leaders if loaded_leaders else polymarket_snapshots,
    )

    poly_scores = []
    for snap in polymarket_snapshots:
        risk = execution_risk_index(snap, divergence_stress=divergence_by_key.get(str(snap.get("market_key")), 0.0))
        poly_scores.append({"market_key": snap.get("market_key"), **risk})

    kalshi_scores = []
    for snap in kalshi_snapshots:
        risk = execution_risk_index(snap, divergence_stress=divergence_by_key.get(str(snap.get("market_key")), 0.0))
        kalshi_scores.append({"market_key": snap.get("market_key"), **risk})

    venue_status = {
        "polymarket": {"ok": poly_source in {"live", "fixture"}, "source": poly_source, "error": poly_error},
        "kalshi": {"ok": kalshi_source in {"live", "fixture"}, "source": kalshi_source, "error": kalshi_error},
    }

    errors: list[dict[str, str]] = []
    if poly_error:
        errors.append({"venue": "polymarket", "error": poly_error})
    if kalshi_error:
        errors.append({"venue": "kalshi", "error": kalshi_error})
    if mapping_error:
        errors.append({"venue": "mapping", "error": mapping_error})

    payload = {
        "run_id": run_id,
        "ts_utc": ts,
        "mode": mode,
        "run_status": _run_status(venue_status),
        "venue_status": venue_status,
        "selection_debug": {
            "polymarket": poly_debug,
            "kalshi": kalshi_debug,
        },
        "polymarket_snapshots": polymarket_snapshots,
        "kalshi_snapshots": kalshi_snapshots,
        "kalshi_snapshot": kalshi_snapshots[0] if kalshi_snapshots else None,
        "cross_venue": {
            "matched_pairs": matched_pairs,
            "kalshi_only_examples": kalshi_only_examples,
            "polymarket_only_examples": polymarket_only_examples,
        },
        "errors": errors,
        "scores": {
            "polymarket": poly_scores,
            "kalshi_markets": kalshi_scores,
            "kalshi": kalshi_scores[0] if kalshi_scores else None,
            "divergence": [{"market_key": k, "divergence": v} for k, v in divergence_by_key.items()],
        },
        "intelligence": {
            "opportunities": opportunities,
            "reference_event_clusters": reference_event_clusters,
            "inconsistencies": inconsistencies,
            "lag_signals": lag_signals,
        },
    }
    return payload


def determine_exit_code(payload: dict[str, Any], mode: str) -> int:
    venue_status = payload.get("venue_status", {})
    ok_poly = bool(venue_status.get("polymarket", {}).get("ok"))
    ok_kalshi = bool(venue_status.get("kalshi", {}).get("ok"))
    if mode == "fixture":
        return 0 if (ok_poly and ok_kalshi) else 1
    return 0 if (ok_poly or ok_kalshi) else 1


def _enable_debug_intelligence() -> None:
    intel_logger = logging.getLogger("market_fetcher.intelligence")
    intel_logger.setLevel(logging.DEBUG)
    if not intel_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        intel_logger.addHandler(handler)
        intel_logger.propagate = False


def _print_intelligence_summary(payload: dict[str, Any], file: Any = None) -> None:
    if file is None:
        file = sys.stderr
    intel = payload.get("intelligence", {})
    lag = intel.get("lag_signals", [])
    lag_sorted = sorted(lag, key=lambda s: s.get("signal_strength", 0.0), reverse=True)
    print("\nINTELLIGENCE SUMMARY", file=file)
    print(f"  Opportunities:            {len(intel.get('opportunities', []))}", file=file)
    print(f"  Reference-event clusters: {len(intel.get('reference_event_clusters', []))}", file=file)
    print(f"  Lag signals:              {len(lag)}", file=file)
    if lag_sorted:
        print("  Top lag signals (by signal_strength):", file=file)
        for s in lag_sorted[:5]:
            leader = (s.get("leader_title") or "?")[:40]
            laggard = (s.get("laggard_title") or "?")[:40]
            print(
                f"    [{s.get('signal_strength', 0.0):.3f}] {s.get('action', '?')}"
                f" | leader: {leader} → laggard: {laggard}",
                file=file,
            )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if getattr(args, "debug_intelligence", False):
        _enable_debug_intelligence()

    try:
        payload = run_pipeline(args)
        exit_code = determine_exit_code(payload, args.mode)
    except Exception as exc:  # pragma: no cover
        payload = {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "ts_utc": utc_now_iso(),
            "mode": args.mode,
            "run_status": "mixed",
            "venue_status": {
                "polymarket": {"ok": False, "source": "none", "error": None},
                "kalshi": {"ok": False, "source": "none", "error": None},
            },
            "selection_debug": {
                "polymarket": {"fetched_candidates": 0, "eligible_two_sided": 0, "selected": 0, "excluded": 0, "exclusion_reasons": {}},
                "kalshi": {"fetched_candidates": 0, "eligible_two_sided": 0, "selected": 0, "excluded": 0, "exclusion_reasons": {}},
            },
            "polymarket_snapshots": [],
            "kalshi_snapshots": [],
            "kalshi_snapshot": None,
            "cross_venue": {
                "matched_pairs": [],
                "kalshi_only_examples": [_example_k({}), _example_k({})],
                "polymarket_only_examples": [_example_p({}), _example_p({})],
            },
            "errors": [{"venue": "pipeline", "error": f"pipeline_error: {exc}"}],
            "scores": {"polymarket": [], "kalshi_markets": [], "kalshi": None, "divergence": []},
        }
        exit_code = 1

    print(json.dumps(payload, indent=2))
    _append_run_to_file(payload)

    history_path = getattr(args, "history_jsonl", None)
    if history_path:
        _append_run_to_jsonl(history_path, payload)

    poly_dbg = payload.get("selection_debug", {}).get("polymarket", {})
    kalshi_dbg = payload.get("selection_debug", {}).get("kalshi", {})
    print("SELECTION SUMMARY", file=sys.stderr)
    print(f"  Polymarket: {_selection_summary_block(poly_dbg)}", file=sys.stderr)
    print(f"  Kalshi: {_selection_summary_block(kalshi_dbg)}", file=sys.stderr)

    if getattr(args, "summary", False):
        _print_intelligence_summary(payload)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
