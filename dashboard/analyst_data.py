"""
Analyst data fetcher for the Target Hunter view.

Pulls analyst price targets and recent broker revisions from yfinance
for every Nifty 50 constituent. Cached separately from the market snapshot
since analyst data moves much more slowly (hours/days, not minutes).
"""

import sys
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
import streamlit as st
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.nifty50_tickers import NIFTY50_STOCKS
from dashboard.config import (
    ANALYST_CACHE_TTL_SECONDS,
    TARGET_REVISION_WINDOW_DAYS,
)

IST = timezone(timedelta(hours=5, minutes=30))

# yfinance Ticker.info keys we care about
_INFO_KEYS = (
    "currentPrice",
    "targetMeanPrice",
    "targetHighPrice",
    "targetLowPrice",
    "targetMedianPrice",
    "recommendationKey",
    "recommendationMean",
    "numberOfAnalystOpinions",
    "beta",
    "marketCap",
)


@dataclass
class AnalystSnapshot:
    """Per-symbol analyst coverage: targets, recommendations, recent revisions."""

    timestamp: datetime
    # per-symbol dict of scalar analyst fields
    targets: dict = field(default_factory=dict)
    # per-symbol DataFrame of recent revisions (firm, date, from_grade, to_grade, action)
    revisions: dict = field(default_factory=dict)
    # symbols where fetch failed
    failed: list = field(default_factory=list)


def _safe_float(value) -> float:
    try:
        if value is None:
            return 0.0
        f = float(value)
        if f != f:  # NaN guard
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def _fetch_analyst_targets(ticker: yf.Ticker) -> dict:
    """Pull the analyst-target fields from Ticker.info."""
    try:
        info = ticker.info or {}
    except Exception as e:
        logger.debug(f"Ticker.info failed: {e}")
        return {}

    out = {k: info.get(k) for k in _INFO_KEYS}
    # normalise numerics
    for k in (
        "currentPrice",
        "targetMeanPrice",
        "targetHighPrice",
        "targetLowPrice",
        "targetMedianPrice",
        "recommendationMean",
        "beta",
    ):
        out[k] = _safe_float(out.get(k))
    out["numberOfAnalystOpinions"] = int(out.get("numberOfAnalystOpinions") or 0)
    out["marketCap"] = _safe_float(out.get("marketCap"))
    out["recommendationKey"] = (out.get("recommendationKey") or "").lower()
    return out


def _fetch_recent_revisions(ticker: yf.Ticker, window_days: int) -> pd.DataFrame:
    """Pull recent upgrades/downgrades table, filter to the last window_days."""
    try:
        df = ticker.upgrades_downgrades
    except Exception as e:
        logger.debug(f"upgrades_downgrades failed: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Date is either the index or a 'GradeDate' column depending on yfinance version
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df.index = pd.to_datetime(df[date_col], errors="coerce")
            df = df.drop(columns=[date_col])

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    if df.empty:
        return pd.DataFrame()

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=window_days)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df[df.index >= cutoff]
    return df.sort_index(ascending=False)


@st.cache_data(ttl=ANALYST_CACHE_TTL_SECONDS, show_spinner="Fetching analyst targets...")
def load_analyst_snapshot() -> AnalystSnapshot:
    """Fetch analyst targets + recent revisions for every Nifty 50 stock."""
    snapshot = AnalystSnapshot(timestamp=datetime.now(IST))

    symbols = list(NIFTY50_STOCKS.items())
    for i, (symbol, (yahoo_ticker, _, _)) in enumerate(symbols):
        try:
            t = yf.Ticker(yahoo_ticker)
            targets = _fetch_analyst_targets(t)
            if not targets or targets.get("numberOfAnalystOpinions", 0) == 0:
                snapshot.failed.append(symbol)
                continue

            snapshot.targets[symbol] = targets
            revisions = _fetch_recent_revisions(t, TARGET_REVISION_WINDOW_DAYS)
            if not revisions.empty:
                snapshot.revisions[symbol] = revisions
        except Exception as e:
            logger.debug(f"Analyst fetch failed for {symbol}: {e}")
            snapshot.failed.append(symbol)

        # Gentle rate-limit against Yahoo
        if i < len(symbols) - 1:
            time.sleep(0.3)

    logger.info(
        f"Analyst snapshot: {len(snapshot.targets)} covered, "
        f"{len(snapshot.revisions)} with recent revisions, "
        f"{len(snapshot.failed)} uncovered"
    )
    return snapshot
