"""
Target Hunter view: surface Nifty 50 stocks where analyst price targets are
materially above current price, validated by a transparent technical score,
with an independent POV on fair exit price and expected timeline.
"""

import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.nifty50_tickers import NIFTY50_STOCKS, get_yahoo_tickers
from dashboard.analyst_data import AnalystSnapshot, load_analyst_snapshot
from dashboard.config import (
    ANALYST_CACHE_TTL_SECONDS,
    SECTOR_SUPPLY_CHAIN,
    TARGET_HUNTER_MIN_ANALYSTS,
    TARGET_HUNTER_MIN_UPSIDE,
    TARGET_HUNTER_TECH_SCORE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Technical scoring
# ---------------------------------------------------------------------------

@st.cache_data(ttl=ANALYST_CACHE_TTL_SECONDS, show_spinner="Loading 1-year history for technicals...")
def _load_technical_history() -> dict:
    """Fetch 1-year daily OHLCV for all Nifty 50 stocks. Returns dict[symbol] -> DataFrame."""
    tickers = " ".join(get_yahoo_tickers())
    try:
        data = yf.download(
            tickers,
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error(f"Technical history fetch failed: {e}")
        return {}

    if data.empty:
        return {}

    result = {}
    for symbol, (yahoo_ticker, _, _) in NIFTY50_STOCKS.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if yahoo_ticker in data.columns.get_level_values(0):
                    sub = data[yahoo_ticker].dropna(how="all")
                else:
                    continue
            else:
                sub = data.dropna(how="all")
            if sub.empty or "Close" not in sub.columns:
                continue
            result[symbol] = sub
        except Exception:
            continue
    return result


def _rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _macd(close: pd.Series) -> tuple[float, float]:
    """Returns (macd_line_last, signal_line_last)."""
    if len(close) < 35:
        return 0.0, 0.0
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal.iloc[-1])


def _technical_breakdown(df: pd.DataFrame) -> dict:
    """Compute indicators + a 0–100 score. Returns dict of metrics."""
    if df is None or df.empty or "Close" not in df.columns:
        return {}
    close = df["Close"].dropna()
    if len(close) < 30:
        return {}

    last = float(close.iloc[-1])
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else last
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else last
    rsi = _rsi(close)
    macd_line, signal_line = _macd(close)

    vol_ratio = 1.0
    if "Volume" in df.columns:
        vol = df["Volume"].dropna()
        if len(vol) >= 20:
            vol_ratio = float(vol.iloc[-1] / max(vol.tail(20).mean(), 1))

    high_52w = float(close.tail(252).max()) if len(close) >= 20 else last
    low_52w = float(close.tail(252).min()) if len(close) >= 20 else last
    pct_of_52w = (last - low_52w) / (high_52w - low_52w) * 100 if high_52w > low_52w else 50.0

    signals = {
        "above_sma50": last > sma50,
        "above_sma200": last > sma200,
        "rsi_healthy": 40 <= rsi <= 65,
        "rsi_not_overbought": rsi < 70,
        "macd_bullish": macd_line > signal_line,
        "volume_support": vol_ratio > 0.8,
    }
    score = sum(signals.values()) * 100 / 6

    return {
        "close": last,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "macd": macd_line,
        "macd_signal": signal_line,
        "vol_ratio": vol_ratio,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_of_52w": pct_of_52w,
        "score": score,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Fair exit + timeline POV
# ---------------------------------------------------------------------------

def _compute_fair_exit(current: float, target_mean: float, target_high: float, tech_score: float) -> float:
    """Blend analyst mean with technical confidence. Capped at analyst high."""
    if current <= 0 or target_mean <= 0:
        return 0.0
    upside = (target_mean - current) / current
    tech_factor = tech_score / 100.0
    fair = current * (1 + upside * (0.5 + 0.5 * tech_factor))
    if target_high > 0:
        fair = min(fair, target_high)
    return fair


def _compute_timeline(beta: float, tech_score: float, macd_bullish: bool, rsi: float) -> str:
    """~6 / ~12 / ~18+ months based on momentum."""
    if tech_score < TARGET_HUNTER_TECH_SCORE_THRESHOLD:
        return "~18+ months"
    if beta >= 1.3 and macd_bullish and rsi < 70:
        return "~6 months"
    return "~12 months"


def _risk_flags(num_analysts: int, rsi: float, high: float, low: float, mean: float) -> list:
    flags = []
    if num_analysts < TARGET_HUNTER_MIN_ANALYSTS:
        flags.append("thin coverage")
    if rsi > 70:
        flags.append("overbought")
    if mean > 0 and (high - low) / mean > 0.30:
        flags.append("wide target range")
    return flags


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_target_hunter():
    """Main entry: render the Target Hunter view."""
    st.title("Target Hunter — Analyst Upside Finder")
    st.caption(
        "Nifty 50 stocks where the analyst mean target sits materially above the current price, "
        "validated against a transparent 6-factor technical score. "
        "Fair exit = blend of analyst mean and our technical confidence, capped at analyst high."
    )

    analyst = load_analyst_snapshot()
    tech_history = _load_technical_history()

    if not analyst.targets:
        st.warning(
            "No analyst coverage returned from Yahoo Finance. "
            "This can happen on weekends, during Yahoo API hiccups, or if the cache is empty. "
            "Try Refresh Data in the sidebar."
        )
        return

    # --- Sidebar filters ---
    st.sidebar.subheader("Target Hunter filters")
    min_upside = st.sidebar.slider(
        "Min upside (%)", 0, 50,
        int(TARGET_HUNTER_MIN_UPSIDE * 100), step=5,
    ) / 100
    min_analysts = st.sidebar.slider(
        "Min analyst count", 1, 30, TARGET_HUNTER_MIN_ANALYSTS,
    )
    min_tech_score = st.sidebar.slider(
        "Min technical score", 0, 100, TARGET_HUNTER_TECH_SCORE_THRESHOLD, step=10,
    )

    # --- Build rows ---
    rows = []
    detail_cache = {}

    for symbol, t in analyst.targets.items():
        current = t.get("currentPrice") or 0.0
        target_mean = t.get("targetMeanPrice") or 0.0
        target_high = t.get("targetHighPrice") or 0.0
        target_low = t.get("targetLowPrice") or 0.0
        num_analysts = t.get("numberOfAnalystOpinions") or 0

        if current <= 0 or target_mean <= 0:
            continue

        upside = (target_mean - current) / current

        tech = _technical_breakdown(tech_history.get(symbol))
        tech_score = tech.get("score", 0.0)
        rsi = tech.get("rsi", 50.0)
        macd_bullish = tech.get("signals", {}).get("macd_bullish", False)
        beta = t.get("beta") or 1.0

        fair_exit = _compute_fair_exit(current, target_mean, target_high, tech_score)
        timeline = _compute_timeline(beta, tech_score, macd_bullish, rsi)
        flags = _risk_flags(num_analysts, rsi, target_high, target_low, target_mean)

        _, company, sector = NIFTY50_STOCKS[symbol]

        row = {
            "Symbol": symbol,
            "Company": company,
            "Sector": sector,
            "Current": current,
            "Mean Target": target_mean,
            "High Target": target_high,
            "Upside %": round(upside * 100, 2),
            "Analysts": num_analysts,
            "Rec": (t.get("recommendationKey") or "").replace("_", " ").title() or "—",
            "Tech Score": round(tech_score, 0),
            "Fair Exit": round(fair_exit, 2),
            "Timeline": timeline,
            "Flags": ", ".join(flags) if flags else "",
        }
        rows.append(row)
        detail_cache[symbol] = {"tech": tech, "targets": t}

    if not rows:
        st.info("No stocks passed the minimum analyst-coverage bar.")
        return

    df = pd.DataFrame(rows)

    # Apply filters
    filtered = df[
        (df["Upside %"] >= min_upside * 100)
        & (df["Analysts"] >= min_analysts)
        & (df["Tech Score"] >= min_tech_score)
    ].sort_values("Upside %", ascending=False)

    st.markdown(
        f"**{len(filtered)} of {len(df)} Nifty 50 stocks** match your filters "
        f"(upside ≥ {int(min_upside*100)}%, analysts ≥ {min_analysts}, tech ≥ {min_tech_score})."
    )

    if filtered.empty:
        st.info("Loosen the filters to see more candidates.")
        return

    st.dataframe(
        filtered,
        column_config={
            "Current": st.column_config.NumberColumn(format="%.2f"),
            "Mean Target": st.column_config.NumberColumn(format="%.2f"),
            "High Target": st.column_config.NumberColumn(format="%.2f"),
            "Upside %": st.column_config.NumberColumn(format="%.2f%%"),
            "Fair Exit": st.column_config.NumberColumn(format="%.2f"),
            "Tech Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        },
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Deep dive")

    for symbol in filtered["Symbol"].tolist():
        d = detail_cache[symbol]
        t = d["targets"]
        tech = d["tech"]
        row = filtered[filtered["Symbol"] == symbol].iloc[0]

        header = (
            f"**{symbol}** — {row['Company']} ・ Upside "
            f"{row['Upside %']:+.1f}% ・ Tech {int(row['Tech Score'])} ・ Fair Exit ₹{row['Fair Exit']:,.2f}"
        )
        with st.expander(header):
            c1, c2 = st.columns([3, 2])

            with c1:
                # Recent broker revisions
                st.markdown("**Recent broker revisions (last 90 days)**")
                revisions = analyst.revisions.get(symbol)
                if revisions is not None and not revisions.empty:
                    rev_df = revisions.reset_index()
                    rev_df.columns = [str(c).replace("_", " ").title() for c in rev_df.columns]
                    st.dataframe(rev_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No recent revisions in the last 90 days.")

                # Analyst target spread
                st.markdown("**Analyst target spread**")
                spread_df = pd.DataFrame({
                    "Metric": ["Current", "Low", "Mean", "Median", "High", "Fair Exit (ours)"],
                    "Price": [
                        t.get("currentPrice") or 0,
                        t.get("targetLowPrice") or 0,
                        t.get("targetMeanPrice") or 0,
                        t.get("targetMedianPrice") or 0,
                        t.get("targetHighPrice") or 0,
                        row["Fair Exit"],
                    ],
                })
                st.dataframe(spread_df, use_container_width=True, hide_index=True,
                             column_config={"Price": st.column_config.NumberColumn(format="%.2f")})

            with c2:
                st.markdown("**Technical breakdown**")
                if tech:
                    st.metric("RSI (14)", f"{tech['rsi']:.1f}")
                    st.metric("Price vs SMA50",
                              f"{(tech['close']/tech['sma50']-1)*100:+.2f}%" if tech['sma50'] else "—")
                    st.metric("Price vs SMA200",
                              f"{(tech['close']/tech['sma200']-1)*100:+.2f}%" if tech['sma200'] else "—")
                    st.metric("% of 52W range", f"{tech['pct_of_52w']:.0f}%")
                    st.caption(
                        f"MACD {'bullish' if tech['signals']['macd_bullish'] else 'bearish'} ・ "
                        f"Volume {tech['vol_ratio']:.2f}× 20d avg"
                    )
                else:
                    st.caption("Technical data unavailable.")

                st.markdown("**Timeline & flags**")
                st.caption(f"Expected: **{row['Timeline']}**")
                if row["Flags"]:
                    st.caption(f":orange[Risk: {row['Flags']}]")

                # Sector supply chain context
                sc = SECTOR_SUPPLY_CHAIN.get(row["Sector"])
                if sc:
                    st.markdown("**Macro drivers**")
                    st.caption(f"Linked to: {', '.join(sc['factors'])}")
                    st.caption(sc["note"])

    # Coverage footer
    st.caption(
        f"Analyst snapshot from {analyst.timestamp.strftime('%I:%M %p IST')} ・ "
        f"{len(analyst.targets)} stocks covered, {len(analyst.failed)} without analyst data."
    )
