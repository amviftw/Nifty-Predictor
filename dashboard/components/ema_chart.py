"""EMA/SMA delta chart: shows Nifty 50 distance from key moving averages."""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from loguru import logger

from dashboard.config import CACHE_TTL_SECONDS
from dashboard.data_loader import _market_minute_bucket


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_nifty_history(period: str = "1y", _bucket: str = "") -> pd.DataFrame:
    """Fetch Nifty 50 index OHLCV history.

    `_bucket` is part of the cache key only.
    """
    del _bucket
    try:
        data = yf.download("^NSEI", period=period, interval="1d", progress=False)
        if data.empty:
            return pd.DataFrame()
        df = data[["Close"]].copy()
        df.columns = ["Close"]
        df = df.dropna()
        return df
    except Exception as e:
        logger.error(f"Nifty history fetch failed: {e}")
        return pd.DataFrame()


def _compute_ema_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute % distance from EMAs/SMAs. Returns DataFrame indexed by date."""
    close = df["Close"]
    result = pd.DataFrame(index=df.index)

    result["EMA 25"] = ((close - close.ewm(span=25, adjust=False).mean())
                         / close.ewm(span=25, adjust=False).mean() * 100)
    result["EMA 50"] = ((close - close.ewm(span=50, adjust=False).mean())
                         / close.ewm(span=50, adjust=False).mean() * 100)
    result["SMA 200"] = ((close - close.rolling(200).mean())
                          / close.rolling(200).mean() * 100)
    return result.dropna()


def render_ema_chart(view: str = "daily"):
    """Render Nifty 50 EMA/SMA delta chart."""
    st.markdown("#### Nifty 50 — Distance from Moving Averages")

    df = _fetch_nifty_history("2y", _bucket=_market_minute_bucket())
    if df.empty:
        st.info("Historical data unavailable for EMA chart")
        return

    deltas = _compute_ema_deltas(df)
    if deltas.empty:
        st.info("Not enough history to compute 200-day SMA")
        return

    if view == "weekly":
        deltas = deltas.resample("W-FRI").last().dropna()

    # Trim to a readable window
    display_window = 120 if view == "daily" else 52
    chart_data = deltas.tail(display_window)

    st.area_chart(chart_data, height=320, use_container_width=True)

    # Current snapshot
    latest = deltas.iloc[-1]
    cols = st.columns(3)
    for col, label in zip(cols, ["EMA 25", "EMA 50", "SMA 200"]):
        val = latest[label]
        with col:
            color = "green" if val > 0 else "red"
            st.markdown(f"**{label}**: :{color}[{val:+.1f}%]")

    if latest["SMA 200"] < 0 and latest["EMA 25"] > 0:
        st.caption("Short-term recovery above EMA 25 while still below 200-day SMA — watch for trend reversal confirmation")
    elif latest["SMA 200"] > 0 and latest["EMA 25"] < 0:
        st.caption("Short-term dip below EMA 25 in a long-term uptrend — potential buying opportunity if support holds")
    elif all(latest[c] > 0 for c in chart_data.columns):
        st.caption("Trading above all key averages — bullish structure intact")
    elif all(latest[c] < 0 for c in chart_data.columns):
        st.caption("Trading below all key averages — bearish structure, wait for reversal signals")
