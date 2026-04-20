"""
OHLCV data fetching and technical indicator computation for the Charts view.

Pulls intraday / daily candles from Yahoo Finance and computes common
indicators (EMA, SMA, Bollinger Bands, RSI, MACD, VWAP) in pure pandas
so no additional TA library is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# (period, interval) presets exposed in the UI.
# period is passed to yfinance; interval controls bar granularity.
TIMEFRAMES: dict[str, tuple[str, str]] = {
    "1D":  ("1d",  "5m"),
    "5D":  ("5d",  "15m"),
    "1M":  ("1mo", "60m"),
    "3M":  ("3mo", "1d"),
    "6M":  ("6mo", "1d"),
    "YTD": ("ytd", "1d"),
    "1Y":  ("1y",  "1d"),
    "5Y":  ("5y",  "1wk"),
    "Max": ("max", "1wk"),
}

INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


@dataclass
class ChartBundle:
    """Container for OHLCV + meta returned to the UI layer."""
    symbol: str
    df: pd.DataFrame            # columns: Open, High, Low, Close, Volume (+ indicators)
    last_price: float
    prev_close: float
    day_change_pct: float
    day_high: float
    day_low: float
    week52_high: float
    week52_low: float
    currency: str


@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV candles from Yahoo Finance. Cached 60s."""
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    # yfinance sometimes returns a MultiIndex on columns — flatten it.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data = data[keep].dropna(how="all")
    data.index = pd.to_datetime(data.index)
    return data


@st.cache_data(ttl=300, show_spinner=False)
def fetch_quote_meta(ticker: str) -> dict:
    """Fetch lightweight quote metadata: 52w high/low, currency, prev close."""
    try:
        info = yf.Ticker(ticker).fast_info
        return {
            "last_price": float(info.get("last_price") or float("nan")),
            "previous_close": float(info.get("previous_close") or float("nan")),
            "year_high": float(info.get("year_high") or float("nan")),
            "year_low": float(info.get("year_low") or float("nan")),
            "currency": str(info.get("currency") or "INR"),
        }
    except Exception:
        return {
            "last_price": float("nan"),
            "previous_close": float("nan"),
            "year_high": float("nan"),
            "year_low": float("nan"),
            "currency": "INR",
        }


# ---------------------------------------------------------------------------
# Indicators — pure pandas
# ---------------------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    return pd.DataFrame({
        "BB_Mid": mid,
        "BB_Upper": mid + num_std * std,
        "BB_Lower": mid - num_std * std,
    })


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Hist": hist})


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP (intraday only). Resets each trading day."""
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        return pd.Series(index=df.index, dtype=float)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = typical * df["Volume"]
    grouped = df.index.date
    return (
        pd.DataFrame({"pv": pv, "v": df["Volume"]}, index=df.index)
        .groupby(grouped)
        .cumsum()
        .pipe(lambda x: x["pv"] / x["v"])
    )


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------

def build_bundle(ticker: str, timeframe_key: str, override_interval: Optional[str] = None) -> ChartBundle:
    """Pull OHLCV, compute indicators, return a ChartBundle ready for plotting."""
    period, default_interval = TIMEFRAMES.get(timeframe_key, ("1mo", "60m"))
    interval = override_interval or default_interval

    df = fetch_ohlcv(ticker, period, interval)
    meta = fetch_quote_meta(ticker)

    if df.empty:
        return ChartBundle(
            symbol=ticker, df=df,
            last_price=float("nan"), prev_close=float("nan"),
            day_change_pct=float("nan"),
            day_high=float("nan"), day_low=float("nan"),
            week52_high=meta["year_high"], week52_low=meta["year_low"],
            currency=meta["currency"],
        )

    close = df["Close"]

    # Moving averages
    df["EMA_20"] = ema(close, 20)
    df["EMA_50"] = ema(close, 50)
    df["SMA_200"] = sma(close, 200)

    # Bollinger
    df = df.join(bollinger(close, 20, 2.0))

    # RSI
    df["RSI"] = rsi(close, 14)

    # MACD
    df = df.join(macd(close, 12, 26, 9))

    # VWAP (intraday only)
    if interval in INTRADAY_INTERVALS:
        df["VWAP"] = vwap(df)

    last_price = float(close.iloc[-1])
    prev_close = (
        float(meta["previous_close"])
        if not np.isnan(meta["previous_close"])
        else float(close.iloc[-2]) if len(close) >= 2 else last_price
    )
    day_change_pct = (last_price / prev_close - 1.0) * 100 if prev_close else 0.0

    today = df.index[-1].date()
    today_slice = df[df.index.date == today]
    day_high = float(today_slice["High"].max()) if not today_slice.empty else float(df["High"].iloc[-1])
    day_low = float(today_slice["Low"].min()) if not today_slice.empty else float(df["Low"].iloc[-1])

    return ChartBundle(
        symbol=ticker,
        df=df,
        last_price=last_price,
        prev_close=prev_close,
        day_change_pct=day_change_pct,
        day_high=day_high,
        day_low=day_low,
        week52_high=meta["year_high"],
        week52_low=meta["year_low"],
        currency=meta["currency"],
    )
