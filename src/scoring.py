"""Execution risk and divergence scoring."""

from __future__ import annotations

from typing import Any

KALSHI_DEPTH_REF = 500.0
POLY_DEPTH_REF = 1000.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _spread_component(venue: str, spread: float | None) -> float:
    if spread is None:
        return 0.0

    if venue == "kalshi" or spread > 1.5:
        return _clamp01(spread / 10.0)

    # Polymarket spread is usually in probability units [0,1].
    return _clamp01(spread / 0.10)


def _depth_component(venue: str, depth_top5: float | None) -> float:
    depth = float(depth_top5 or 0.0)
    depth_ref = KALSHI_DEPTH_REF if venue == "kalshi" else POLY_DEPTH_REF
    return 1.0 - min(1.0, depth / max(1.0, depth_ref))


def liquidity_stress(snapshot: dict[str, Any]) -> float:
    spread_c = _spread_component(snapshot.get("venue", ""), snapshot.get("spread"))
    depth_c = _depth_component(snapshot.get("venue", ""), snapshot.get("depth_top5"))
    return _clamp01((spread_c + depth_c) / 2.0)


def flow_stress(snapshot: dict[str, Any], churn: float | None = None) -> float:
    venue = snapshot.get("venue")
    if venue != "kalshi":
        return 0.2

    trades_count = int(snapshot.get("trades_count_sample") or 0)
    if trades_count > 0:
        return _clamp01(1.0 - min(1.0, trades_count / 50.0))

    if churn is not None:
        return _clamp01(churn)

    return 0.2


def compute_divergence(kalshi_snapshot: dict[str, Any], polymarket_snapshot: dict[str, Any]) -> float | None:
    """Compute normalized cross-venue mid divergence in [0, 1]."""

    kalshi_mid = kalshi_snapshot.get("mid")
    poly_mid = polymarket_snapshot.get("mid")

    if kalshi_mid is None or poly_mid is None:
        return None

    try:
        kalshi_scaled = float(kalshi_mid) / 100.0
        poly_scaled = float(poly_mid)
    except (TypeError, ValueError):
        return None

    return _clamp01(abs(kalshi_scaled - poly_scaled))


def execution_risk_index(
    snapshot: dict[str, Any],
    divergence_stress: float = 0.0,
    churn: float | None = None,
) -> dict[str, Any]:
    """Compute ExecutionRiskIndex, regime, and component drivers for a snapshot."""

    liq = liquidity_stress(snapshot)
    flow = flow_stress(snapshot, churn=churn)
    div = _clamp01(divergence_stress)

    eri = _clamp01(0.45 * liq + 0.35 * flow + 0.20 * div)

    if eri < 0.33:
        regime = "LOW"
    elif eri < 0.66:
        regime = "ELEVATED"
    else:
        regime = "HIGH"

    return {
        "execution_risk_index": eri,
        "regime": regime,
        "drivers": {
            "liquidity_stress": liq,
            "flow_stress": flow,
            "divergence_stress": div,
        },
    }


def compute_execution_risk(
    snapshot: dict[str, Any],
    venue: str | None = None,
    divergence_stress: float = 0.0,
    churn: float | None = None,
) -> float:
    """Return scalar execution risk index for a snapshot.

    `venue` is accepted for API compatibility and may override snapshot venue.
    """

    enriched = dict(snapshot)
    if venue:
        enriched["venue"] = venue
    return execution_risk_index(enriched, divergence_stress=divergence_stress, churn=churn)["execution_risk_index"]


def compute_regime(
    execution_risk: float,
) -> str:
    """Return regime label from scalar execution risk."""

    if execution_risk < 0.33:
        return "LOW"
    if execution_risk < 0.66:
        return "ELEVATED"
    return "HIGH"


def compute_drivers(
    snapshot: dict[str, Any],
    venue: str | None = None,
    divergence_stress: float = 0.0,
    churn: float | None = None,
) -> dict[str, float]:
    """Return decomposed stress drivers for a snapshot."""

    enriched = dict(snapshot)
    if venue:
        enriched["venue"] = venue
    return execution_risk_index(enriched, divergence_stress=divergence_stress, churn=churn)["drivers"]
