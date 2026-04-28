"""Generate a structured LLM prompt from a fetch_snapshots.py output JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LLM_INSTRUCTION = (
    "You are a prediction-market trade idea analyst. Use only the provided data. "
    "Do not invent facts. Lag signals are the primary source of trade ideas. "
    "Review candidates are not trade recommendations. Do not convert review "
    "candidates into trades unless the causal direction is clearly supported. "
    "If only review candidates exist, say no strong trade exists. Evaluate whether "
    "leader movement causally implies same-direction, opposite-direction, or no "
    "clear movement in the laggard. "
    "Reference-event clusters are context, not standalone trade recommendations. "
    "Produce ranked trade ideas with trade, action, confidence, evidence, "
    "risks, invalidation criteria, and priority."
)


def load_snapshot(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sort_lag_signals(signals: list[dict]) -> list[dict]:
    return sorted(signals, key=lambda s: s.get("signal_strength", 0.0), reverse=True)


def get_execution_risk(market_key: str | None, poly_scores: list[dict]) -> dict | None:
    """Return the execution-risk fields for a market_key, or None if not found."""
    if not market_key:
        return None
    target = str(market_key)
    for score in poly_scores:
        if str(score.get("market_key")) == target:
            return {k: v for k, v in score.items() if k != "market_key"}
    return None


def get_market_context(market_key: str | None, snapshots: list[dict]) -> dict | None:
    """Return spread, depth_top5, and mid for a market_key from the snapshot list."""
    if not market_key:
        return None
    target = str(market_key)
    for snap in snapshots:
        if str(snap.get("market_key")) == target:
            return {
                "mid": snap.get("mid"),
                "spread": snap.get("spread"),
                "depth_top5": snap.get("depth_top5"),
            }
    return None


def build_structured_input(snapshot: dict[str, Any]) -> dict[str, Any]:
    if isinstance(snapshot.get("claude_input"), dict):
        claude_input = snapshot["claude_input"]
        if {"primary_trades", "review_candidates", "reasoning_context"}.issubset(claude_input):
            return claude_input
    if {"primary_trades", "review_candidates", "reasoning_context"}.issubset(snapshot):
        return {
            "primary_trades": snapshot.get("primary_trades", []),
            "review_candidates": snapshot.get("review_candidates", []),
            "reasoning_context": snapshot.get("reasoning_context", []),
        }
    intel = snapshot.get("intelligence", {})
    lag_signals = sort_lag_signals(intel.get("lag_signals", []))
    review_candidates = sorted(
        intel.get("review_candidates", []),
        key=lambda s: s.get("relationship_score", 0.0),
        reverse=True,
    )
    return {
        "primary_trades": lag_signals,
        "review_candidates": review_candidates,
        "reasoning_context": [
            {
                "type": "execution_intelligence_constraint",
                "message": (
                    "Vardr ranks execution opportunities from leader-lag divergence, "
                    "relationship direction, liquidity, and execution risk; it does "
                    "not predict event outcomes."
                ),
            },
            {
                "type": "causality_guardrail",
                "message": "Do not promote review candidates or speculative correlations to primary trades.",
            },
        ],
    }


def build_prompt(snapshot: dict[str, Any]) -> str:
    intel = snapshot.get("intelligence", {})
    lag_signals = sort_lag_signals(intel.get("lag_signals", []))
    review_candidates = sorted(
        intel.get("review_candidates", []),
        key=lambda s: s.get("relationship_score", 0.0),
        reverse=True,
    )
    opportunities = intel.get("opportunities", [])
    ref_clusters = intel.get("reference_event_clusters", [])
    poly_scores = snapshot.get("scores", {}).get("polymarket", [])
    poly_snapshots = snapshot.get("polymarket_snapshots", [])

    lines: list[str] = []

    lines.append("=== SYSTEM INSTRUCTION ===")
    lines.append(LLM_INSTRUCTION)
    lines.append("")

    lines.append("=== STRUCTURED CLAUDE INPUT ===")
    lines.append(json.dumps(build_structured_input(snapshot), indent=2, sort_keys=True))
    lines.append("")

    lines.append("=== PRIMARY TRADE IDEAS (lag_signals) ===")
    if not lag_signals:
        lines.append("No lag signals detected. No strong trade ideas at this time.")
    else:
        for i, sig in enumerate(lag_signals, 1):
            laggard_key = sig.get("laggard_market_key")
            risk = get_execution_risk(laggard_key, poly_scores)
            ctx = get_market_context(laggard_key, poly_snapshots)
            laggard_move = sig.get("laggard_move") or 0.0

            lines.append(f"Signal {i} [strength={sig.get('signal_strength', 0.0):.4f}]")
            lines.append(f"  Action:           {sig.get('action', '?')}")
            lines.append(
                f"  Leader:           {sig.get('leader_title', '?')}"
                f" (moved {sig.get('leader_move', 0.0):+.4f})"
            )
            lines.append(
                f"  Laggard:          {sig.get('laggard_title', '?')}"
                f" (moved {laggard_move:+.4f})"
            )
            lines.append(
                f"  Relationship:     {sig.get('relationship', '?')}"
                f" (score={sig.get('relationship_score', 0.0):.2f})"
            )
            if ctx:
                lines.append(
                    f"  Laggard market:   mid={ctx['mid']},"
                    f" spread={ctx['spread']},"
                    f" depth_top5={ctx['depth_top5']}"
                )
            if risk:
                risk_str = "  ".join(f"{k}={v}" for k, v in sorted(risk.items()))
                lines.append(f"  Execution risk:   {risk_str}")
            lines.append(f"  Reason:           {sig.get('reason', '?')}")
            lines.append("")

    lines.append("")

    lines.append("=== REVIEW CANDIDATES - not trade recommendations ===")
    if not review_candidates:
        lines.append("No review candidates detected.")
    else:
        for i, candidate in enumerate(review_candidates, 1):
            laggard_move = candidate.get("laggard_move") or 0.0
            lines.append(f"Review candidate {i}")
            lines.append(
                f"  Leader:             {candidate.get('leader_title', '?')}"
                f" (moved {candidate.get('leader_move', 0.0):+.4f})"
            )
            lines.append(
                f"  Laggard:            {candidate.get('laggard_title', '?')}"
                f" (moved {laggard_move:+.4f})"
            )
            lines.append(
                f"  Relationship score: {candidate.get('relationship_score', 0.0):.2f}"
            )
            lines.append(f"  Reason:             {candidate.get('reason', '?')}")
            lines.append(f"  Review question:    {candidate.get('review_question', '?')}")
            lines.append("")

    lines.append("")

    if ref_clusters:
        lines.append(
            "=== REFERENCE-EVENT CLUSTERS"
            " (context only, not standalone trade recommendations) ==="
        )
        for cluster in ref_clusters:
            n = len(cluster.get("markets", []))
            lines.append(
                f"Event: {cluster.get('reference_event', '?')}"
                f" | {n} markets"
                f" | price_range={cluster.get('price_range', 0):.4f}"
            )
            lines.append(f"  Context: {cluster.get('reason', '?')}")
            for m in cluster.get("markets", []):
                lines.append(f"    - {m.get('title', '?')}: mid={m.get('mid')}")
        lines.append("")

    if opportunities:
        lines.append("=== OPPORTUNITIES (watchlist) ===")
        for opp in opportunities:
            lines.append(
                f"  - {opp.get('title', '?')}"
                f" | mid={opp.get('mid')}"
                f" spread={opp.get('spread')}"
                f" depth_top5={opp.get('depth_top5')}"
            )
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a structured trade idea prompt from a snapshot JSON"
    )
    parser.add_argument("--snapshot", required=True, help="Path to snapshot JSON file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = load_snapshot(args.snapshot)
    print(build_prompt(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
