"""
OHLCV fetch + technical indicator math for the Charts view.

Pulls from Yahoo Finance (free, no key, works for NSE stocks and indices)
and computes indicators in pure pandas. The output is shaped for
TradingView's Lightweight Charts library: lists of
{time, open, high, low, close} and {time, value} dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# (yfinance period, yfinance interval) presets exposed in the UI.
TIMEFRAMES: dict[str, tuple[str, str]] = {
    "1D":  ("5d",  "5m"),    # 5 days of 5-min bars => ~390 bars, looks dense
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
    symbol: str
    df: pd.DataFrame
    last_price: float
    prev_close: float
    day_change_pct: float
    day_high: float
    day_low: float
    week52_high: float
    week52_low: float
    currency: str
    is_intraday: bool


# ---------------------------------------------------------------------------
# Fetchers (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, threads=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data = data[keep].dropna(how="all")
    data.index = pd.to_datetime(data.index)
    return data


@st.cache_data(ttl=300, show_spinner=False)
def fetch_quote_meta(ticker: str) -> dict:
    try:
        fi = yf.Ticker(ticker).fast_info
        return {
            "last_price": float(fi.get("last_price") or float("nan")),
            "previous_close": float(fi.get("previous_close") or float("nan")),
            "year_high": float(fi.get("year_high") or float("nan")),
            "year_low": float(fi.get("year_low") or float("nan")),
            "currency": str(fi.get("currency") or "INR"),
        }
    except Exception:
        return {
            "last_price": float("nan"), "previous_close": float("nan"),
            "year_high": float("nan"), "year_low": float("nan"),
            "currency": "INR",
        }


# ---------------------------------------------------------------------------
# Indicators
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
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Hist": hist})


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def build_bundle(ticker: str, timeframe_key: str, override_interval: Optional[str] = None) -> ChartBundle:
    period, default_interval = TIMEFRAMES.get(timeframe_key, ("3mo", "1d"))
    interval = override_interval or default_interval
    is_intraday = interval in INTRADAY_INTERVALS

    df = fetch_ohlcv(ticker, period, interval)
    meta = fetch_quote_meta(ticker)

    if df.empty:
        return ChartBundle(
            symbol=ticker, df=df,
            last_price=meta["last_price"], prev_close=meta["previous_close"],
            day_change_pct=float("nan"),
            day_high=float("nan"), day_low=float("nan"),
            week52_high=meta["year_high"], week52_low=meta["year_low"],
            currency=meta["currency"], is_intraday=is_intraday,
        )

    close = df["Close"]
    df["EMA_20"] = ema(close, 20)
    df["EMA_50"] = ema(close, 50)
    df["SMA_200"] = sma(close, 200)
    df = df.join(bollinger(close, 20, 2.0))
    df["RSI"] = rsi(close, 14)
    df = df.join(macd(close, 12, 26, 9))

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
        symbol=ticker, df=df,
        last_price=last_price, prev_close=prev_close,
        day_change_pct=day_change_pct,
        day_high=day_high, day_low=day_low,
        week52_high=meta["year_high"], week52_low=meta["year_low"],
        currency=meta["currency"], is_intraday=is_intraday,
    )


# ---------------------------------------------------------------------------
# Shape for lightweight-charts
# ---------------------------------------------------------------------------

def _time_key(ts: pd.Timestamp, is_intraday: bool):
    """Lightweight-charts wants UNIX seconds for intraday, YYYY-MM-DD for daily."""
    if is_intraday:
        return int(ts.timestamp())
    return ts.strftime("%Y-%m-%d")


def candles_payload(df: pd.DataFrame, is_intraday: bool) -> list[dict]:
    return [
        {
            "time": _time_key(idx, is_intraday),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low":  float(row["Low"]),
            "close": float(row["Close"]),
        }
        for idx, row in df.iterrows()
        if not (np.isnan(row["Open"]) or np.isnan(row["Close"]))
    ]


def volume_payload(df: pd.DataFrame, is_intraday: bool) -> list[dict]:
    up_color = "rgba(38, 166, 154, 0.55)"
    down_color = "rgba(239, 83, 80, 0.55)"
    return [
        {
            "time": _time_key(idx, is_intraday),
            "value": float(row["Volume"] or 0),
            "color": up_color if row["Close"] >= row["Open"] else down_color,
        }
        for idx, row in df.iterrows()
        if not np.isnan(row.get("Volume", np.nan))
    ]


def line_payload(series: pd.Series, is_intraday: bool) -> list[dict]:
    """Convert a numeric series (indexed by timestamps) to [{time,value}]."""
    return [
        {"time": _time_key(idx, is_intraday), "value": float(v)}
        for idx, v in series.dropna().items()
    ]


def histogram_payload(series: pd.Series, is_intraday: bool) -> list[dict]:
    out = []
    for idx, v in series.dropna().items():
        if np.isnan(v):
            continue
        out.append({
            "time": _time_key(idx, is_intraday),
            "value": float(v),
            "color": "rgba(38, 166, 154, 0.7)" if v >= 0 else "rgba(239, 83, 80, 0.7)",
        })
    return out
