"""
Hypothesis Validator — let the user express a conditional trading belief
and backtest it on historical close-to-close data.

Example hypothesis (expressed in UI):
    "When Brent drops more than 2% AND India VIX rises more than 5%,
     BUY HDFCBANK next day."

We translate that into:
    conditions = [
        ("crude", "<", -2.0),
        ("vix",  ">", +5.0),
    ]
    target = HDFCBANK
    direction = LONG
    horizon = 1 (next trading day close-to-close)

…and return hit rate, average return, and sample size over the history window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

try:
    import streamlit as st
    _CACHE = st.cache_data(ttl=1800, show_spinner=False)
except ImportError:  # pragma: no cover
    def _CACHE(fn):
        return fn

import yfinance as yf

from config.nifty50_tickers import NIFTY50_STOCKS
from scenarios.drivers import DRIVERS, price_based_drivers


Operator = Literal[">", "<", ">=", "<="]


@dataclass
class Condition:
    driver_id: str
    op: Operator
    threshold_pct: float   # In the driver's % unit.


@dataclass
class HypothesisResult:
    hypothesis: str
    n_trigger_days: int
    n_total_days: int
    hit_rate: float          # P(direction correct on trigger day's horizon)
    avg_return_pct: float    # Mean next-horizon return on trigger days
    median_return_pct: float
    stddev_pct: float
    returns: pd.Series       # Every conditional sample, for the histogram
    lookback_days: int


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def validate_hypothesis(
    conditions: list[Condition],
    target_symbol: str,
    direction: str = "LONG",
    horizon_days: int = 1,
    lookback_days: int = 750,
) -> HypothesisResult:
    """Backtest a conditional trade idea on historical data.

    Args:
        conditions: list of driver-threshold triggers that must ALL fire.
        target_symbol: NSE symbol (e.g. "HDFCBANK") to trade when triggers fire.
        direction: "LONG" or "SHORT". Hit rate is computed for this direction.
        horizon_days: close-to-close holding period, in trading days.
        lookback_days: how far back to search for trigger days.

    Returns:
        HypothesisResult. If data is missing, a result with n_trigger_days=0.
    """
    if target_symbol not in NIFTY50_STOCKS:
        return _empty_result(conditions, target_symbol, direction, lookback_days)

    target_ticker = NIFTY50_STOCKS[target_symbol][0]
    driver_tickers = {c.driver_id: DRIVERS[c.driver_id].ticker for c in conditions if c.driver_id in DRIVERS}

    # Filter to drivers we can pull from yfinance.
    driver_tickers = {d: t for d, t in driver_tickers.items() if t}
    if not driver_tickers:
        return _empty_result(conditions, target_symbol, direction, lookback_days)

    tickers = tuple([target_ticker] + list(driver_tickers.values()))
    closes = _fetch_history(tickers, lookback_days)
    if closes.empty or target_ticker not in closes.columns:
        return _empty_result(conditions, target_symbol, direction, lookback_days)

    target_rets = (closes[target_ticker].pct_change(horizon_days).shift(-horizon_days) * 100.0)

    # Compute each driver's daily return (% change, same-day), then build the
    # joint trigger mask.
    trigger_mask = pd.Series(True, index=closes.index)
    evaluated_conditions: list[Condition] = []
    for c in conditions:
        tkr = driver_tickers.get(c.driver_id)
        if not tkr or tkr not in closes.columns:
            continue
        drv_ret = closes[tkr].pct_change() * 100.0
        cond_mask = _apply_op(drv_ret, c.op, c.threshold_pct)
        trigger_mask = trigger_mask & cond_mask
        evaluated_conditions.append(c)

    joined = pd.concat([trigger_mask.rename("trigger"), target_rets.rename("ret")], axis=1).dropna()
    joined = joined.tail(lookback_days)
    n_total = int(len(joined))
    fired = joined[joined["trigger"]]
    n_trigger = int(len(fired))

    if n_trigger == 0:
        return HypothesisResult(
            hypothesis=_describe(evaluated_conditions, target_symbol, direction, horizon_days),
            n_trigger_days=0,
            n_total_days=n_total,
            hit_rate=float("nan"),
            avg_return_pct=float("nan"),
            median_return_pct=float("nan"),
            stddev_pct=float("nan"),
            returns=pd.Series(dtype=float),
            lookback_days=lookback_days,
        )

    rets = fired["ret"]
    direction_sign = +1 if direction == "LONG" else -1
    wins = (direction_sign * rets) > 0
    hit_rate = float(wins.mean())

    return HypothesisResult(
        hypothesis=_describe(evaluated_conditions, target_symbol, direction, horizon_days),
        n_trigger_days=n_trigger,
        n_total_days=n_total,
        hit_rate=hit_rate,
        avg_return_pct=float(rets.mean() * direction_sign),
        median_return_pct=float(rets.median() * direction_sign),
        stddev_pct=float(rets.std()),
        returns=rets * direction_sign,
        lookback_days=lookback_days,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _apply_op(series: pd.Series, op: Operator, threshold: float) -> pd.Series:
    if op == ">":
        return series > threshold
    if op == "<":
        return series < threshold
    if op == ">=":
        return series >= threshold
    if op == "<=":
        return series <= threshold
    raise ValueError(f"Unknown operator: {op}")


def _describe(
    conditions: list[Condition],
    target: str,
    direction: str,
    horizon: int,
) -> str:
    parts = []
    for c in conditions:
        d = DRIVERS.get(c.driver_id)
        if not d:
            continue
        parts.append(f"{d.name} {c.op} {c.threshold_pct}{d.unit}")
    if not parts:
        return f"{direction} {target} (no parseable conditions)"
    joined = " AND ".join(parts)
    horizon_txt = "next day" if horizon == 1 else f"next {horizon} days"
    return f"When {joined} → {direction} {target}, check {horizon_txt}"


def _empty_result(conditions, target, direction, lookback) -> HypothesisResult:
    return HypothesisResult(
        hypothesis=_describe(conditions, target, direction, 1),
        n_trigger_days=0,
        n_total_days=0,
        hit_rate=float("nan"),
        avg_return_pct=float("nan"),
        median_return_pct=float("nan"),
        stddev_pct=float("nan"),
        returns=pd.Series(dtype=float),
        lookback_days=lookback,
    )


@_CACHE
def _fetch_history(tickers: tuple[str, ...], lookback_days: int) -> pd.DataFrame:
    """Fetch close prices for the given tickers. Tuple so Streamlit can hash."""
    period = f"{max(lookback_days + 60, 400)}d"
    raw = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    closes: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                sub = raw[t]
                if "Close" in sub.columns:
                    closes[t] = sub["Close"].dropna()
    else:
        if "Close" in raw.columns and len(tickers) == 1:
            closes[tickers[0]] = raw["Close"].dropna()

    if not closes:
        return pd.DataFrame()

    df = pd.concat(closes, axis=1)
    df.index = pd.to_datetime(df.index)
    return df
