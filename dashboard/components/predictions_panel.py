"""
ML predictions panel: shows BUY/SELL/HOLD signals if available from the
offline prediction pipeline, otherwise falls back to a live technical
signal summary computed from price data.
"""

import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.holidays import prev_trading_day, is_trading_day
from config.nifty50_tickers import NIFTY50_STOCKS, get_yahoo_tickers
from dashboard.config import CACHE_TTL_SECONDS

DB_PATH = Path(__file__).resolve().parents[2] / "storage" / "nifty_predictor.db"


def render_predictions_panel():
    """Try the ML pipeline DB first; fall back to live technical signals."""
    if _try_ml_predictions():
        return
    _render_technical_signals()


# ---------------------------------------------------------------------------
# Path 1: ML predictions from the offline pipeline DB
# ---------------------------------------------------------------------------

def _try_ml_predictions() -> bool:
    """Attempt to load ML predictions. Returns True if rendered."""
    try:
        from data.storage.db_manager import DBManager
        db = DBManager(db_path=str(DB_PATH))

        today = date.today()
        check_date = today if is_trading_day(today) else prev_trading_day(today)
        predictions = db.get_predictions(check_date.isoformat())

        if not predictions:
            predictions = db.get_predictions(prev_trading_day(check_date).isoformat())

        if not predictions:
            return False

        st.markdown("#### ML Trading Signals")
        pred_date = predictions[0].get("date", "")
        st.caption(f"From offline pipeline — predictions for {pred_date}")

        buys = [p for p in predictions if p.get("signal") == "BUY"]
        sells = [p for p in predictions if p.get("signal") == "SELL"]
        holds = [p for p in predictions if p.get("signal") == "HOLD"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("BUY", len(buys))
        with c2:
            st.metric("SELL", len(sells))
        with c3:
            st.metric("HOLD", len(holds))

        if buys:
            st.markdown("**:green[BUY Signals]**")
            _render_signal_table(buys)
        if sells:
            st.markdown("**:red[SELL Signals]**")
            _render_signal_table(sells)
        return True

    except Exception as e:
        logger.debug(f"ML predictions unavailable: {e}")
        return False


def _render_signal_table(signals: list[dict]):
    records = []
    for s in signals:
        symbol = s.get("symbol", "")
        company = NIFTY50_STOCKS.get(symbol, (None, symbol, "Unknown"))[1]
        records.append({
            "Symbol": symbol,
            "Company": company,
            "Confidence": round((s.get("confidence") or 0) * 100, 1),
            "P(Up)": round((s.get("prob_up") or 0) * 100, 1),
            "P(Down)": round((s.get("prob_down") or 0) * 100, 1),
            "P(Flat)": round((s.get("prob_flat") or 0) * 100, 1),
        })
    df = pd.DataFrame(records)
    st.dataframe(
        df,
        column_config={
            "Confidence": st.column_config.ProgressColumn("Conf %", min_value=0, max_value=100),
            "P(Up)": st.column_config.NumberColumn(format="%.1f%%"),
            "P(Down)": st.column_config.NumberColumn(format="%.1f%%"),
            "P(Flat)": st.column_config.NumberColumn(format="%.1f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Path 2: live technical signal scan (no ML models needed)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Computing technical signals...")
def _compute_tech_signals() -> pd.DataFrame:
    """Quick RSI + MACD + SMA scan for all Nifty 50 stocks."""
    tickers = " ".join(get_yahoo_tickers())
    try:
        data = yf.download(
            tickers, period="6mo", interval="1d",
            group_by="ticker", auto_adjust=True, progress=False, threads=True,
        )
    except Exception as e:
        logger.error(f"Tech signal fetch failed: {e}")
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    rows = []
    for symbol, (yahoo_ticker, company, sector) in NIFTY50_STOCKS.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if yahoo_ticker not in data.columns.get_level_values(0):
                    continue
                sub = data[yahoo_ticker]
            else:
                sub = data
            close = sub["Close"].dropna()
            if len(close) < 50:
                continue

            last = float(close.iloc[-1])

            # RSI 14
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = float((100 - 100 / (1 + rs)).iloc[-1])

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = float((macd - signal).iloc[-1])

            # SMA 50
            sma50 = float(close.rolling(50).mean().iloc[-1])
            above_sma50 = last > sma50

            # Derive signal
            if rsi < 35 and macd_hist > 0 and above_sma50:
                sig = "BUY"
            elif rsi > 70 and macd_hist < 0:
                sig = "SELL"
            elif rsi < 40 and macd_hist > 0:
                sig = "BUY"
            elif rsi > 65 and macd_hist < 0 and not above_sma50:
                sig = "SELL"
            else:
                sig = "HOLD"

            rows.append({
                "Symbol": symbol,
                "Company": company,
                "Sector": sector,
                "Close": round(last, 2),
                "RSI": round(rsi, 1),
                "MACD Hist": round(macd_hist, 2),
                "Above SMA50": above_sma50,
                "Signal": sig,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def _render_technical_signals():
    """Render a live technical signal scan as the fallback view."""
    st.markdown("#### Technical Signal Scanner")
    st.caption(
        "Live scan — RSI, MACD histogram, and SMA 50 for all Nifty 50 stocks. "
        "Signals are rule-based (not ML). For ML predictions, run "
        "`python scripts/daily_predict.py` locally."
    )

    df = _compute_tech_signals()
    if df.empty:
        st.info("Unable to compute signals — data fetch may have failed.")
        return

    buys = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]
    holds = df[df["Signal"] == "HOLD"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("BUY", len(buys))
    with c2:
        st.metric("SELL", len(sells))
    with c3:
        st.metric("HOLD", len(holds))

    tab_buy, tab_sell, tab_all = st.tabs(["BUY signals", "SELL signals", "All stocks"])

    col_config = {
        "Close": st.column_config.NumberColumn(format="%.2f"),
        "RSI": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        "MACD Hist": st.column_config.NumberColumn(format="%.2f"),
        "Above SMA50": st.column_config.CheckboxColumn("Above SMA50"),
    }

    with tab_buy:
        if buys.empty:
            st.caption("No BUY signals right now.")
        else:
            st.dataframe(buys.sort_values("RSI"), column_config=col_config,
                         use_container_width=True, hide_index=True)

    with tab_sell:
        if sells.empty:
            st.caption("No SELL signals right now.")
        else:
            st.dataframe(sells.sort_values("RSI", ascending=False), column_config=col_config,
                         use_container_width=True, hide_index=True)

    with tab_all:
        st.dataframe(df.sort_values("Signal"), column_config=col_config,
                     use_container_width=True, hide_index=True)
