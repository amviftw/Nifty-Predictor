"""Top movers: chip-card layout with ATH gap, RSI, and volume context."""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from loguru import logger

from config.nifty50_tickers import NIFTY50_STOCKS, get_yahoo_tickers
from dashboard.data_loader import MarketSnapshot, _market_minute_bucket
from dashboard.config import CACHE_TTL_SECONDS


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_extended_metrics(_bucket: str = "") -> dict:
    """Batch-fetch 1Y data → RSI 14, volume ratio, 52-week high per stock.

    `_bucket` is part of the cache key only — pass `_market_minute_bucket()`
    so per-stock RSI/52W/volume metrics on the chip cards stay in sync with
    the price coming from `load_market_snapshot`.
    """
    del _bucket
    tickers = get_yahoo_tickers()
    try:
        data = yf.download(
            " ".join(tickers), period="1y", interval="1d",
            group_by="ticker", auto_adjust=True, progress=False, threads=True,
        )
    except Exception as e:
        logger.error(f"Extended metrics fetch failed: {e}")
        return {}

    if data.empty:
        return {}

    result = {}
    for symbol, (yahoo_ticker, _company, _sector) in NIFTY50_STOCKS.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if yahoo_ticker not in data.columns.get_level_values(0):
                    continue
                sub = data[yahoo_ticker]
            else:
                sub = data
            close = sub["Close"].dropna()
            volume = sub["Volume"].dropna()
            if len(close) < 20:
                continue

            current = float(close.iloc[-1])
            high_52w = float(close.max())
            ath_gap_pct = ((current - high_52w) / high_52w) * 100 if high_52w else 0

            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = float((100 - 100 / (1 + rs)).iloc[-1])

            vol_ratio = 0.0
            if len(volume) >= 21:
                avg_vol = float(volume.iloc[-21:-1].mean())
                vol_ratio = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 0

            result[symbol] = {
                "ath_gap_pct": round(ath_gap_pct, 1),
                "rsi": round(rsi, 1),
                "vol_ratio": round(vol_ratio, 2),
            }
        except Exception:
            continue
    return result


def render_top_movers(snapshot: MarketSnapshot):
    """Render top gainers and losers as interactive chip cards."""
    st.markdown("#### Top Movers")

    if not snapshot.top_gainers and not snapshot.top_losers:
        st.info("Stock data unavailable")
        return

    ext = _fetch_extended_metrics(_bucket=_market_minute_bucket())
    change_key = "dod_pct" if snapshot.view == "daily" else "wow_pct"
    change_label = "Day" if snapshot.view == "daily" else "Week"

    st.markdown(
        f'<div style="font-size:0.72rem; color:#7a8294; margin-bottom:4px;">'
        f'Sorted by {change_label} change &middot; '
        f'<span style="color:#c9cfd9;">vs 52W High</span> shows distance from peak &middot; '
        f'<span style="color:#c9cfd9;">Vol</span> = today vs 20-day avg</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**:green[Gainers]**")
    _render_chips(snapshot.top_gainers, ext, change_key, is_gainer=True)

    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)

    st.markdown("**:red[Losers]**")
    _render_chips(snapshot.top_losers, ext, change_key, is_gainer=False)


def _render_chips(stocks: list, ext: dict, change_key: str, is_gainer: bool):
    """Render a flex-wrap row of stock chip cards."""
    if not stocks:
        st.caption("No data")
        return

    chips = []
    for s in stocks:
        sym = s["symbol"]
        company = s["company"]
        sector = s.get("sector", "")
        close = s["close"]
        change = s[change_key]
        m = ext.get(sym, {})

        if is_gainer:
            chg_color, chg_bg, border = "#00d09c", "rgba(0,208,156,0.12)", "rgba(0,208,156,0.22)"
        else:
            chg_color, chg_bg, border = "#eb5757", "rgba(235,87,87,0.12)", "rgba(235,87,87,0.22)"

        # ATH gap
        ath_gap = m.get("ath_gap_pct")
        ath_html = ""
        if ath_gap is not None:
            if ath_gap > -3:
                ath_color, ath_txt = "#f5a623", "Near peak"
            elif ath_gap > -15:
                ath_color, ath_txt = "#c9cfd9", f"{ath_gap:.0f}%"
            else:
                ath_color, ath_txt = "#00d09c", f"{ath_gap:.0f}%"
            ath_html = (
                f'<div><span class="chip-sub">vs 52W High</span>'
                f'<br><span style="color:{ath_color};font-weight:500;">{ath_txt}</span></div>'
            )

        # RSI
        rsi = m.get("rsi")
        rsi_html = ""
        if rsi is not None:
            if rsi < 30:
                rsi_color = "#00d09c"
            elif rsi > 70:
                rsi_color = "#eb5757"
            else:
                rsi_color = "#c9cfd9"
            rsi_html = (
                f'<div><span class="chip-sub">RSI</span>'
                f'<br><span style="color:{rsi_color};font-weight:500;">{rsi:.0f}</span></div>'
            )

        # Volume ratio
        vol = m.get("vol_ratio")
        vol_html = ""
        if vol is not None and vol > 0:
            vol_color = "#00d09c" if vol > 1.5 else "#c9cfd9" if vol > 0.8 else "#7a8294"
            vol_html = (
                f'<div><span class="chip-sub">Vol</span>'
                f'<br><span style="color:{vol_color};font-weight:500;">{vol:.1f}x</span></div>'
            )

        # Insight line
        insight = _derive_insight(change, ath_gap, rsi, vol, is_gainer)
        insight_html = ""
        if insight:
            insight_html = (
                f'<div style="margin-top:8px;padding-top:7px;border-top:1px solid #232834;'
                f'font-size:0.66rem;color:#7a8294;line-height:1.3;">{insight}</div>'
            )

        chip = (
            f'<div style="background:#151922;border:1px solid {border};border-radius:12px;'
            f'padding:14px 16px;min-width:175px;flex:1 1 175px;max-width:230px;'
            f'transition:border-color 0.15s ease;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-weight:700;font-size:0.85rem;color:#e8ecf1;">{sym}</span>'
            f'<span style="background:{chg_bg};color:{chg_color};padding:2px 8px;'
            f'border-radius:4px;font-weight:600;font-size:0.78rem;">{change:+.2f}%</span>'
            f'</div>'
            f'<div style="font-size:0.68rem;color:#7a8294;margin-top:2px;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis;">{company}</div>'
            f'<div style="font-size:1.05rem;font-weight:600;color:#e8ecf1;margin-top:6px;">'
            f'&#8377;{close:,.2f}</div>'
            f'<div style="display:flex;gap:14px;margin-top:8px;font-size:0.72rem;">'
            f'{ath_html}{vol_html}{rsi_html}'
            f'</div>'
            f'<div style="margin-top:8px;"><span style="font-size:0.62rem;background:#232834;'
            f'padding:2px 6px;border-radius:3px;color:#7a8294;">{sector}</span></div>'
            f'{insight_html}'
            f'</div>'
        )
        chips.append(chip)

    st.markdown(
        '<style>.chip-sub{font-size:0.62rem;color:#7a8294;text-transform:uppercase;'
        'letter-spacing:0.04em;}</style>'
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


def _derive_insight(change, ath_gap, rsi, vol, is_gainer):
    """Generate a one-line actionable insight for the chip."""
    parts = []
    if is_gainer:
        if ath_gap is not None and ath_gap < -20 and rsi is not None and rsi < 60:
            parts.append("Well below peak, room to run")
        elif ath_gap is not None and ath_gap > -3:
            parts.append("Nearing all-time high")
        if vol is not None and vol > 2.0:
            parts.append("Heavy institutional volume")
        elif vol is not None and vol > 1.5:
            parts.append("Above-avg volume")
        if rsi is not None and rsi > 70:
            parts.append("Overbought zone")
    else:
        if ath_gap is not None and ath_gap < -25:
            parts.append("Deep correction from peak")
        elif ath_gap is not None and ath_gap < -15:
            parts.append("Significant pullback")
        if rsi is not None and rsi < 30:
            parts.append("Oversold — watch for bounce")
        if vol is not None and vol > 2.0:
            parts.append("High selling volume")

    return " · ".join(parts[:2]) if parts else ""
