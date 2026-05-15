"""
Per-sector support/resistance levels + next-session directional bias.

The dashboard's sector chips historically reported only "what happened" (DoD,
WoW, MoM). This module derives a lightweight "what's likely next" overlay
from the daily-close history we already fetch in
`dashboard.components.sector_deep_dive._fetch_sector_history`:

  - Support (S)     = recent 20-day swing low
  - Resistance (R)  = recent 20-day swing high
  - Pivot (P)       = (H + L + C) / 3 of the latest session
  - RSI(14)         = Wilder's smoothing
  - Distance vs EMA-21 = momentum trend proxy
  - Bias score      = weighted blend → "Bullish / Neutral / Bearish"

The bias is intentionally simple and explainable — the chip caption shows
the underlying signals so a user can disagree on sight rather than trusting
a black-box label.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# Magic numbers, named.
_PIVOT_WINDOW = 20            # lookback for recent swing high / low
_RSI_PERIOD = 14
_EMA_TREND_SPAN = 21
_BIAS_BULL_THRESHOLD = 1.0
_BIAS_BEAR_THRESHOLD = -1.0


@dataclass
class SectorSignal:
    support: float
    resistance: float
    pivot: float
    rsi: float
    ema21_gap_pct: float       # close vs EMA-21, % (positive => above EMA)
    near_support: bool         # within 1.5% of support
    near_resistance: bool      # within 1.5% of resistance
    bias: str                  # "Bullish" | "Neutral" | "Bearish"
    bias_score: float          # raw score (signed); positive = bullish
    rationale: str             # one-liner: why this bias

    def as_dict(self) -> dict:
        return asdict(self)


def _rsi(closes: pd.Series, period: int = _RSI_PERIOD) -> float:
    """Wilder's RSI. Returns 50.0 if the series is too short for a stable read."""
    if len(closes) < period + 1:
        return 50.0
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return val if np.isfinite(val) else 50.0


def compute_signal(closes: pd.Series) -> SectorSignal | None:
    """Derive a SectorSignal from a daily-close series.

    Returns None when there isn't enough history to compute a stable read
    (caller should treat that as "no overlay available").
    """
    closes = closes.dropna()
    if len(closes) < max(_PIVOT_WINDOW, _RSI_PERIOD + 5):
        return None

    last_close = float(closes.iloc[-1])
    if last_close <= 0:
        return None

    recent = closes.tail(_PIVOT_WINDOW)
    support = float(recent.min())
    resistance = float(recent.max())

    # Pivot uses the latest day as a proxy for (H, L, C) because we only
    # store close. That's close enough for a directional bias; if intraday
    # high/low get plumbed through later this becomes a true pivot.
    pivot = (support + resistance + last_close) / 3.0

    rsi = _rsi(closes)

    ema21 = closes.ewm(span=_EMA_TREND_SPAN, adjust=False).mean()
    ema21_last = float(ema21.iloc[-1])
    ema21_gap_pct = ((last_close - ema21_last) / ema21_last * 100) if ema21_last else 0.0

    # Proximity flags — within 1.5% of either level
    near_support = abs(last_close - support) / last_close < 0.015 if last_close else False
    near_resistance = abs(resistance - last_close) / last_close < 0.015 if last_close else False

    # Weighted score: bounded contributions so no single signal dominates.
    score = 0.0
    rationale_bits = []

    # 1. Trend: above / below EMA-21
    if ema21_gap_pct > 0.5:
        score += 1.0
        rationale_bits.append(f"above EMA-21 ({ema21_gap_pct:+.1f}%)")
    elif ema21_gap_pct < -0.5:
        score -= 1.0
        rationale_bits.append(f"below EMA-21 ({ema21_gap_pct:+.1f}%)")

    # 2. RSI regime: <30 oversold (mean-revert bullish), >70 overbought
    if rsi < 35:
        score += 0.8
        rationale_bits.append(f"RSI oversold ({rsi:.0f})")
    elif rsi > 65:
        score -= 0.8
        rationale_bits.append(f"RSI overbought ({rsi:.0f})")
    elif rsi >= 55:
        score += 0.4
        rationale_bits.append(f"RSI firm ({rsi:.0f})")
    elif rsi <= 45:
        score -= 0.4
        rationale_bits.append(f"RSI soft ({rsi:.0f})")

    # 3. Proximity to S/R: a bounce off support skews bullish, rejection at
    #    resistance skews bearish.
    if near_support:
        score += 0.6
        rationale_bits.append("at support")
    elif near_resistance:
        score -= 0.6
        rationale_bits.append("at resistance")

    if score >= _BIAS_BULL_THRESHOLD:
        bias = "Bullish"
    elif score <= _BIAS_BEAR_THRESHOLD:
        bias = "Bearish"
    else:
        bias = "Neutral"

    rationale = "; ".join(rationale_bits) if rationale_bits else "mid-range, no edge"

    return SectorSignal(
        support=round(support, 2),
        resistance=round(resistance, 2),
        pivot=round(pivot, 2),
        rsi=round(rsi, 1),
        ema21_gap_pct=round(ema21_gap_pct, 2),
        near_support=near_support,
        near_resistance=near_resistance,
        bias=bias,
        bias_score=round(score, 2),
        rationale=rationale,
    )


def compute_sector_signals(
    histories: dict[str, pd.Series],
) -> dict[str, SectorSignal]:
    """Apply `compute_signal` to every sector with usable history."""
    out: dict[str, SectorSignal] = {}
    for name, closes in histories.items():
        sig = compute_signal(closes)
        if sig is not None:
            out[name] = sig
    return out
