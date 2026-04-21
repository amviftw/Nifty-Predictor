"""
Scenario Engine — turn a set of driver shocks into ranked trade ideas.

Given:
    shocks = {"crude": +3.0, "usdinr": +0.8, "vix": +15}   (each in its driver's unit)

We produce:
    ScenarioResult
        - expected_move_pct for every Nifty 50 stock (linear combo of betas)
        - ranked trade ideas (long winners / short losers) with:
            * hypothesis sentence (Varsity-style)
            * base rate of the same directional setup working historically
            * suggested entry / stop-loss proxy (using recent realised vol)
            * confidence label derived from consistency + sample size

The model is intentionally linear and transparent. It is NOT a predictive
model — it is a *teaching* model that lets a trader see their own reasoning
decomposed into attributable factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config.nifty50_tickers import NIFTY50_STOCKS, get_sector, get_company_name
from scenarios.drivers import DRIVERS, Driver
from scenarios.sensitivity import SensitivityMatrix


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class TradeIdea:
    """A single actionable idea emitted by the scenario engine."""
    symbol: str
    company: str
    sector: str
    direction: str           # "LONG" or "SHORT"
    expected_move_pct: float # Linear prediction in %
    base_rate: float         # Historical hit-rate of this direction given similar shock, 0..1
    n_analogs: int           # Number of historical days that contributed to base_rate
    confidence: str          # "HIGH" | "MEDIUM" | "LOW"
    hypothesis: str          # Natural-language reason, referencing the dominant drivers
    dominant_drivers: list[tuple[str, float]] = field(default_factory=list)  # (driver_id, contribution_pct)


@dataclass
class ScenarioResult:
    """Full output of run_scenario."""
    shocks: dict[str, float]                # What was asked
    per_stock: pd.DataFrame                 # symbol, sector, expected_pct, base_rate, ...
    long_ideas: list[TradeIdea]
    short_ideas: list[TradeIdea]
    nifty_expected_pct: float               # Cap-weighted would be better; equal-weight for now.
    sector_expected_pct: pd.Series          # Mean expected move per sector
    driver_contributions: pd.DataFrame      # driver_id x sector, contribution to sector move


# -----------------------------------------------------------------------------
# Core engine
# -----------------------------------------------------------------------------

def run_scenario(
    shocks: dict[str, float],
    matrix: SensitivityMatrix,
    top_n: int = 5,
) -> ScenarioResult:
    """Simulate the market impact of a set of driver shocks.

    Args:
        shocks: mapping from driver_id -> shock magnitude in that driver's unit.
                For price-return drivers the unit is %; for yields it's %-point;
                for FII flows it's ₹ cr.
        matrix: precomputed sensitivity matrix.
        top_n:  how many long / short ideas to surface.

    Returns:
        ScenarioResult with the expected move per stock plus ranked trade ideas.
    """
    if matrix.betas.empty:
        return _empty_result(shocks)

    # Normalise shocks to fractional returns where the driver is a price return
    # (yfinance daily log returns are fractional). For yields, input is in %-points,
    # so we convert to fractional change by dividing by average yield (approx).
    shock_vector = _normalise_shocks(shocks)

    # Align shock vector with matrix rows (drivers). Missing drivers => 0.
    aligned = pd.Series(
        {d: shock_vector.get(d, 0.0) for d in matrix.betas.index},
        dtype=float,
    )

    # Contribution of each driver to each stock = beta * shock.
    #   per_driver_contrib[d, s] = beta[d, s] * shock[d]
    betas = matrix.betas.fillna(0.0)
    per_driver_contrib = betas.multiply(aligned, axis=0)  # rows=drivers, cols=stocks
    expected_frac = per_driver_contrib.sum(axis=0)        # series indexed by stock
    expected_pct = expected_frac * 100.0

    # Build per-stock table.
    rows = []
    for sym in betas.columns:
        rows.append({
            "symbol": sym,
            "company": get_company_name(sym),
            "sector": get_sector(sym),
            "expected_pct": float(expected_pct.get(sym, 0.0)),
        })
    per_stock = pd.DataFrame(rows)

    # Equal-weight "Nifty" proxy — good enough to anchor the user.
    nifty_expected = float(per_stock["expected_pct"].mean())

    sector_expected = (
        per_stock.groupby("sector")["expected_pct"].mean().sort_values(ascending=False)
    )

    # Driver x Sector contribution for the heatmap. pandas 3.0 removed
    # DataFrame.groupby(axis=1), so transpose -> groupby rows -> transpose back.
    symbol_to_sector = {sym: get_sector(sym) for sym in per_driver_contrib.columns}
    contrib_df = per_driver_contrib.rename(columns=symbol_to_sector)
    driver_sector_contrib = (
        contrib_df.T.groupby(level=0).mean().T * 100.0
    )

    # Trade ideas — long the top expected winners, short the top losers.
    ranked = per_stock.sort_values("expected_pct", ascending=False)
    long_side = ranked.head(top_n)
    short_side = ranked.tail(top_n).iloc[::-1]

    long_ideas = [
        _build_idea(row, per_driver_contrib, matrix, shocks, direction="LONG")
        for _, row in long_side.iterrows()
    ]
    short_ideas = [
        _build_idea(row, per_driver_contrib, matrix, shocks, direction="SHORT")
        for _, row in short_side.iterrows()
    ]

    return ScenarioResult(
        shocks=shocks,
        per_stock=per_stock,
        long_ideas=long_ideas,
        short_ideas=short_ideas,
        nifty_expected_pct=nifty_expected,
        sector_expected_pct=sector_expected,
        driver_contributions=driver_sector_contrib,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalise_shocks(shocks: dict[str, float]) -> dict[str, float]:
    """Convert each shock from its UI unit to a fractional (decimal) value that
    matches the units of the betas in the sensitivity matrix."""
    normalised: dict[str, float] = {}
    for d_id, raw in shocks.items():
        if d_id not in DRIVERS:
            continue
        driver = DRIVERS[d_id]
        if driver.unit == "%":
            normalised[d_id] = raw / 100.0
        elif d_id == "us10y":
            # Beta was computed vs daily yield return; a 0.1 %-point move on a
            # ~4% yield is roughly 0.025 fractional change. Rough but consistent.
            normalised[d_id] = raw / 4.0 / 100.0
        elif d_id == "fii_flow":
            # FII flow is not in the price-return matrix (no yfinance ticker),
            # so it won't contribute to expected_move. Keep shock pass-through
            # for UI display; engine ignores it. Users still see it in the "why".
            normalised[d_id] = raw / 10000.0  # scale only used if matrix extended
        else:
            normalised[d_id] = raw / 100.0
    return normalised


def _build_idea(
    row: pd.Series,
    per_driver_contrib: pd.DataFrame,
    matrix: SensitivityMatrix,
    raw_shocks: dict[str, float],
    direction: str,
) -> TradeIdea:
    """Assemble a TradeIdea from a per-stock row and its driver contributions."""
    sym = row["symbol"]

    # Top 2 contributing drivers (by absolute magnitude, sign-consistent with direction).
    contrib = per_driver_contrib[sym].sort_values(
        key=lambda s: s.abs(), ascending=False
    ) * 100.0
    dominant: list[tuple[str, float]] = []
    for drv_id, c in contrib.items():
        if abs(c) < 1e-4:
            continue
        dominant.append((drv_id, float(c)))
        if len(dominant) >= 2:
            break

    # Base rate: look up the hit-rate for the *strongest* driver in the direction
    # matching the largest shock to that driver.
    base_rate, n_analogs = _base_rate_for_idea(sym, dominant, raw_shocks, matrix, direction)

    # Confidence buckets based on |expected move| and n_analogs.
    confidence = _confidence_bucket(row["expected_pct"], base_rate, n_analogs)

    # Varsity-style hypothesis sentence.
    hypothesis = _hypothesis_sentence(row, dominant, raw_shocks, direction)

    return TradeIdea(
        symbol=sym,
        company=row["company"],
        sector=row["sector"],
        direction=direction,
        expected_move_pct=float(row["expected_pct"]),
        base_rate=base_rate,
        n_analogs=n_analogs,
        confidence=confidence,
        hypothesis=hypothesis,
        dominant_drivers=dominant,
    )


def _base_rate_for_idea(
    symbol: str,
    dominant: list[tuple[str, float]],
    raw_shocks: dict[str, float],
    matrix: SensitivityMatrix,
    direction: str,
) -> tuple[float, int]:
    """Historical hit rate of the idea direction given the dominant driver's
    shock direction. We use the strongest driver as the conditioning event."""
    if not dominant:
        return (float("nan"), 0)

    top_driver_id, _ = dominant[0]
    shock = raw_shocks.get(top_driver_id, 0.0)

    if shock >= 0:
        hit = matrix.up_hit_rates.loc[top_driver_id, symbol] if top_driver_id in matrix.up_hit_rates.index else float("nan")
    else:
        hit = matrix.down_hit_rates.loc[top_driver_id, symbol] if top_driver_id in matrix.down_hit_rates.index else float("nan")

    n = int(matrix.n_obs.get(top_driver_id, 0) or 0)

    # If the trader is SHORT but historical "up driver" leads stock up, our hit
    # rate for the short side is approximately (1 - hit).
    expected_direction = "LONG" if shock * (matrix.betas.loc[top_driver_id, symbol] or 0) >= 0 else "SHORT"
    if direction != expected_direction and not np.isnan(hit):
        hit = 1.0 - hit

    return (float(hit) if not np.isnan(hit) else float("nan"), n)


def _confidence_bucket(expected_pct: float, base_rate: float, n_analogs: int) -> str:
    """Bucket idea confidence using a simple rubric."""
    if np.isnan(base_rate) or n_analogs < 50:
        return "LOW"
    strong_move = abs(expected_pct) >= 1.0
    strong_history = base_rate >= 0.58
    if strong_move and strong_history:
        return "HIGH"
    if strong_move or strong_history:
        return "MEDIUM"
    return "LOW"


def _hypothesis_sentence(
    row: pd.Series,
    dominant: list[tuple[str, float]],
    raw_shocks: dict[str, float],
    direction: str,
) -> str:
    """Compose a Varsity-style explanatory sentence."""
    if not dominant:
        return f"{direction} {row['symbol']}: no strong driver links detected in the window."

    parts = []
    for drv_id, contrib in dominant:
        if drv_id not in DRIVERS:
            continue
        d = DRIVERS[drv_id]
        shock = raw_shocks.get(drv_id, 0.0)
        sign_word = "up" if shock >= 0 else "down"
        mag = f"{abs(shock):.2g}{d.unit}" if d.unit == "%" else f"{abs(shock):.2g} {d.unit}"
        supportive = (contrib >= 0 and direction == "LONG") or (contrib < 0 and direction == "SHORT")
        contrib_word = "supporting" if supportive else "opposing"
        parts.append(f"{d.name} {sign_word} {mag} ({contrib_word}, {contrib:+.2f}%)")

    reasoning = "; ".join(parts)
    if direction == "LONG":
        stem = f"{row['company']} ({row['sector']}) benefits from this scenario."
    else:
        stem = f"{row['company']} ({row['sector']}) gets hurt by this scenario."
    return f"{stem} Breakdown — {reasoning}."


def _empty_result(shocks: dict[str, float]) -> ScenarioResult:
    return ScenarioResult(
        shocks=shocks,
        per_stock=pd.DataFrame(columns=["symbol", "company", "sector", "expected_pct"]),
        long_ideas=[],
        short_ideas=[],
        nifty_expected_pct=float("nan"),
        sector_expected_pct=pd.Series(dtype=float),
        driver_contributions=pd.DataFrame(),
    )
